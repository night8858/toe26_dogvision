#include <ros/ros.h>

#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "common_structs.h"
#include "detector.hpp"
#include "ocr_detect.hpp"

namespace fs = std::filesystem;

// 透视变换裁剪文本区域（4点框 → 矩形，竖排文本自动旋转90°使其横排）
static cv::Mat crop_text_region(const cv::Mat &src, const OCRBox &box)
{
    const float w = std::max(
        static_cast<float>(cv::norm(box.pts[0] - box.pts[1])),
        static_cast<float>(cv::norm(box.pts[2] - box.pts[3])));
    const float h = std::max(
        static_cast<float>(cv::norm(box.pts[0] - box.pts[3])),
        static_cast<float>(cv::norm(box.pts[1] - box.pts[2])));

    if (w < 1.0f || h < 1.0f)
        return {};

    cv::Point2f src_pts[4], dst_pts[4];
    for (int i = 0; i < 4; ++i)
        src_pts[i] = box.pts[i];
    dst_pts[0] = {0.0f,     0.0f    };
    dst_pts[1] = {w - 1.0f, 0.0f    };
    dst_pts[2] = {w - 1.0f, h - 1.0f};
    dst_pts[3] = {0.0f,     h - 1.0f};

    cv::Mat transform = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat crop;
    cv::warpPerspective(src, crop, transform,
                        cv::Size(static_cast<int>(w), static_cast<int>(h)));

    // 竖排文本（高 > 宽 * 1.5）旋转90°
    if (crop.rows > static_cast<int>(crop.cols * 1.5f))
        cv::rotate(crop, crop, cv::ROTATE_90_CLOCKWISE);

    return crop;
}

// 在图像上绘制4点检测框和识别文本
static void draw_ocr_result(cv::Mat &vis, const OCRBox &box, const OCRRecResult &rec)
{
    std::vector<cv::Point> poly;
    for (int i = 0; i < 4; ++i)
        poly.emplace_back(static_cast<int>(std::round(box.pts[i].x)),
                          static_cast<int>(std::round(box.pts[i].y)));
    cv::polylines(vis, poly, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    const int text_y = std::max(0, static_cast<int>(std::round(box.pts[0].y)) - 5);
    std::ostringstream label;
    label << rec.text << " " << std::fixed << std::setprecision(2) << rec.score;
    cv::putText(vis, label.str(),
                cv::Point(static_cast<int>(std::round(box.pts[0].x)), text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ppocr_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string image_path;
    std::string output_dir;
    std::string config_path;

    pnh.param<std::string>("image_path",  image_path,  "/home/toe/toe26_dogvision/src/dogvision26/src/data/img/image_143643394669487.png");
    pnh.param<std::string>("output_dir",  output_dir,  "/home/toe/toe26_dogvision/src/dogvision26/src/data/ocr_output");
    pnh.param<std::string>("config_path", config_path,
        "/home/toe/toe26_dogvision/src/dogvision26/src/detect/settings.json");

    if (image_path.empty())
    {
        ROS_ERROR("Missing required param ~image_path");
        return -1;
    }
    if (output_dir.empty())
    {
        ROS_ERROR("Missing required param ~output_dir");
        return -1;
    }

    // ── 1. 加载配置 ──────────────────────────────────────────────────────────
    Appconfig config;
    {
        detect_det_ppocr loader(nullptr);
        loader.load_config(config, config_path);
    }
    ROS_INFO_STREAM("Config      : " << config_path);
    ROS_INFO_STREAM("Det model   : " << config.detect_config.ppocr_det_model_path);
    ROS_INFO_STREAM("Rec model   : " << config.detect_config.ppocr_rec_model_path);
    ROS_INFO_STREAM("Dict        : " << config.detect_config.rec_char_dict_path);

    // ── 2. 初始化 det 检测模型 ───────────────────────────────────────────────
    detect_det_ppocr det(&config);
    det.load_model(config.detect_config.ppocr_det_model_path,
                   config.detect_config.det_device);

    // ── 3. 初始化 rec 识别模型 ───────────────────────────────────────────────
    detect_rec_ppocr rec(&config);
    rec.load_model(config.detect_config.ppocr_rec_model_path,
                   config.detect_config.rec_device);
    rec.loda_dict(config.detect_config.rec_char_dict_path);

    // max_wh_ratio 固定为配置中的 rec_img_w / rec_img_h（默认 320/48≈6.67）
    // 超出此比例的裁剪图会被缩放到最大宽度，无需动态 reshape 模型
    const float default_wh_ratio =
        (config.detect_config.rec_img_h > 0)
        ? static_cast<float>(config.detect_config.rec_img_w) /
          static_cast<float>(config.detect_config.rec_img_h)
        : 320.0f / 48.0f;
    rec.set_max_wh_ratio(default_wh_ratio);

    // ── 4. 读取输入图像 ──────────────────────────────────────────────────────
    cv::Mat img = cv::imread(image_path);
    if (img.empty())
    {
        ROS_ERROR_STREAM("Cannot read image: " << image_path);
        return -1;
    }
    ROS_INFO_STREAM("Image       : " << image_path
                    << " [" << img.cols << "x" << img.rows << "]");

    // ── 5. 文本检测（det）───────────────────────────────────────────────────
    det.preprocess(img);
    det.inference();
    det.postprocess();
    const std::vector<OCRBox> &boxes = det.ocr_det_out_;
    ROS_INFO_STREAM("Detected " << boxes.size() << " text region(s)");

    // ── 6. 裁剪 + 文本识别（rec）────────────────────────────────────────────
    std::vector<OCRItem> ocr_items;
    cv::Mat vis = img.clone();

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        cv::Mat crop = crop_text_region(img, boxes[i]);
        if (crop.empty())
            continue;

        rec.preprocess(crop);
        rec.inference();
        rec.postprocess();

        if (!rec.result.empty() && !rec.result[0].text.empty())
        {
            OCRItem item;
            item.box = boxes[i];
            item.rec = rec.result[0];
            ocr_items.push_back(item);
            draw_ocr_result(vis, boxes[i], rec.result[0]);
            ROS_INFO_STREAM("  [" << i << "] \""
                            << rec.result[0].text
                            << "\"  score=" << rec.result[0].score);
        }
    }

    // ── 7. 保存结果到指定目录 ────────────────────────────────────────────────
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec)
    {
        ROS_ERROR_STREAM("Cannot create output dir: "
                         << output_dir << " (" << ec.message() << ")");
        return -1;
    }

    const std::string out_img_path = output_dir + "/result.jpg";
    const std::string out_txt_path = output_dir + "/result.txt";

    if (!cv::imwrite(out_img_path, vis))
        ROS_WARN_STREAM("Failed to write image: " << out_img_path);
    else
        ROS_INFO_STREAM("Result image: " << out_img_path);

    {
        std::ofstream ofs(out_txt_path);
        if (!ofs.is_open())
        {
            ROS_WARN_STREAM("Failed to write text: " << out_txt_path);
        }
        else
        {
            for (size_t i = 0; i < ocr_items.size(); ++i)
            {
                ofs << "[" << i << "]\t"
                    << ocr_items[i].rec.text << "\t"
                    << std::fixed << std::setprecision(4)
                    << ocr_items[i].rec.score << "\n";
            }
            ROS_INFO_STREAM("Result text : " << out_txt_path
                            << " (" << ocr_items.size() << " items)");
        }
    }

    return 0;
}

