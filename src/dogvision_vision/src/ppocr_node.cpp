#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <dogvision_vision/common_structs.h>
#include <dogvision_vision/ocr_detect.hpp>
#include <dogvision_vision/ocr_utils.hpp>

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ppocr_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string image_path;
    std::string output_dir;
    std::string config_path;

    pnh.param<std::string>("image_path", image_path,
                           ros::package::getPath("dogvision_vision") + "/data/img/image_143643394669487.png");
    pnh.param<std::string>("output_dir", output_dir,
                           ros::package::getPath("dogvision_vision") + "/data/ocr_output");
    pnh.param<std::string>("config_path", config_path,
                           ros::package::getPath("dogvision_vision") + "/config/settings.json");

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

    // ── 4b. 定位白底算术题 ROI，截取后送入 det ──────────────────────────────
    cv::Mat det_input = img;           // 默认使用全图
    cv::Rect2f math_roi = find_math_proble(img);
    if (math_roi.area() > 0.f)
    {
        cv::Rect math_rect(
            static_cast<int>(math_roi.x),
            static_cast<int>(math_roi.y),
            static_cast<int>(math_roi.width),
            static_cast<int>(math_roi.height));
        // 安全 clamp（确保不越界）
        math_rect &= cv::Rect(0, 0, img.cols, img.rows);
        if (math_rect.area() > 0)
        {
            det_input = img(math_rect).clone();
            ROS_INFO("Math ROI    : [%d, %d, %d x %d]",
                     math_rect.x, math_rect.y, math_rect.width, math_rect.height);
        }
    }
    else
    {
        ROS_WARN("No math region detected, running OCR on full image.");
    }

    // ── 5. 文本检测（det）───────────────────────────────────────────────────
    det.preprocess(det_input);
    det.inference();
    det.postprocess();
    const std::vector<OCRBox> &boxes = det.ocr_det_out_;
    ROS_INFO_STREAM("Detected " << boxes.size() << " text region(s)");

    // ── 6. 裁剪 + 文本识别（rec）────────────────────────────────────────────
    std::vector<OCRItem> ocr_items;
    cv::Mat vis = det_input.clone();   // 可视化在 det_input 坐标系上绘制

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        cv::Mat crop = crop_text_region(det_input, boxes[i]);
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

    // ── 6b. 算术识别：汇总所有 OCR 文本，解析并计算 ─────────────────────────
    {
        std::string all_text;
        for (const auto &item : ocr_items)
            all_text += item.rec.text + " ";

        ROS_INFO_STREAM("All OCR text: \"" << all_text << "\"");

        double calc_result = 0.0;
        std::string expr_str;
        if (parse_simple_expr(all_text, calc_result, expr_str))
        {
            int int_result = static_cast<int>(std::round(calc_result));
            int mod_result = ((int_result % 4) + 4) % 4; // 保证非负
            ROS_INFO("Expr: %s  =>  %d  %%4 = %d",
                     expr_str.c_str(), int_result, mod_result);
            show_result_window(expr_str, mod_result);
        }
        else
        {
            ROS_WARN("No arithmetic expression found in OCR output.");
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
