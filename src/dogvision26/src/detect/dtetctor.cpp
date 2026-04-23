#include <iostream>
#include "detector.hpp"
#include "common_structs.h"
#include <fstream>
#include <jsoncpp/json/json.h>
#include <opencv2/calib3d.hpp>
#include <ros/package.h>
#include <string>

// ---------------------------------------------------------------
// 鱼眼去畸变配置区
// ---------------------------------------------------------------
// 是否启用鱼眼去畸变。关闭后 diatorion() 将直接返回原图拷贝。
constexpr bool kEnableFisheyeUndistort = true;

// balance 取值范围通常为 [0, 1]：
// 1) 越接近 0：裁切更多，黑边更少，视场更窄。
// 2) 越接近 1：保留更多原始视场，黑边可能更多。
constexpr double kFisheyeBalance = 0.0;

// fov_scale > 1 会扩大视场，< 1 会缩小视场；通常保持 1.0。
constexpr double kFisheyeFovScale = 1.0;

// --- 填充 K 矩阵 (3x3) ---
const cv::Mat K = (cv::Mat_<double>(3, 3) << 8.2631010840557929e+02, 0., 7.3508237365876721e+02,
                   0., 8.3234495506807673e+02, 5.6784864582942498e+02,
                   0., 0., 1.);

// --- 填充 D 矩阵 (4x1) ---
const cv::Mat D = (cv::Mat_<double>(4, 1) << 1.9474519085664992e-02,
                   2.2096711413330011e-02,
                   -4.1006640770500716e-02,
                   2.6220651979250005e-02);
// 用于处理畸变
cv::Mat map1, map2;
cv::Mat newCameraMatrix;
bool g_fisheye_map_ready = false;

void detector::load_config(Appconfig &config, std::string json_file_path)
{
    Json::Reader reader;
    Json::Value value;
    std::ifstream in(json_file_path, std::ios::binary);
    std::cout << "load json now..." << std::endl;
    if (!in.is_open())
    {
        std::cerr << "Failed to open file: " << json_file_path;
        exit(1);
    }
    if (reader.parse(in, value))
    {
        // 获取包路径用于将 JSON 中的相对路径转为绝对路径
        const std::string pkg = ros::package::getPath("dogvision26");
        auto resolve = [&](const std::string &p) -> std::string
        {
            if (p.empty() || p[0] == '/')
                return p; // 已是绝对路径则直接使用
            return pkg + "/" + p;
        };

        config.detect_config.bin_file_path = resolve(value["path"]["openvino_bin_file_path"].asString());
        config.detect_config.xml_file_path = resolve(value["path"]["openvino_xml_file_path"].asString());

        config.detect_config.ppocr_det_model_path = resolve(value["path"]["ppocr_det_model_path"].asString());
        config.detect_config.ppocr_rec_model_path = resolve(value["path"]["ppocr_rec_model_path"].asString());
        config.detect_config.ppocr_cls_model_path = resolve(value["path"]["ppocr_cls_model_path"].asString());
        config.detect_config.rec_char_dict_path = resolve(value["path"]["ppocr_dict_path"].asString());

        config.detect_config.batch_size = value["NCHW"]["batch_size"].asInt();
        config.detect_config.c = value["NCHW"]["C"].asInt();
        config.detect_config.w = value["NCHW"]["W"].asInt();
        config.detect_config.h = value["NCHW"]["H"].asInt();

        config.detect_config.type = value["img"]["type"].asInt();
        config.detect_config.width = value["img"]["width"].asInt();
        config.detect_config.height = value["img"]["height"].asInt();

        config.detect_config.nms_thresh = value["thresh"]["nms_thresh"].asFloat();
        config.detect_config.bbox_conf_thresh = value["thresh"]["bbox_conf_thresh"].asFloat();
        config.detect_config.merge_thresh = value["thresh"]["merge_thresh"].asFloat();

        config.detect_config.classes = value["nums"]["classes"].asInt();
        config.detect_config.class0 = value["nums"]["cls0"].asString();
        config.detect_config.class1 = value["nums"]["cls1"].asString();
        config.detect_config.class2 = value["nums"]["cls2"].asString();
        config.detect_config.class3 = value["nums"]["cls3"].asString();

        config.hikcamera_config.device_id = value["hikcamera"]["device_id"].asInt();
        config.hikcamera_config.exposure = value["hikcamera"]["exposure"].asInt();
        config.hikcamera_config.height = value["hikcamera"]["height"].asInt();
        config.hikcamera_config.width = value["hikcamera"]["width"].asInt();
        config.hikcamera_config.offset_x = value["hikcamera"]["offset_x"].asInt();
        config.hikcamera_config.offset_y = value["hikcamera"]["offset_y"].asInt();

        config.usbcamera_config[0].device_id = value["usbcamera0"]["device_id"].asInt();
        config.usbcamera_config[0].width = value["usbcamera0"]["width"].asInt();
        config.usbcamera_config[0].height = value["usbcamera0"]["height"].asInt();

#ifdef TWO_CAMERAS
        // 此处可补充多个相机的初始化
#endif
    }
    else
    {
        std::cerr << "Load Json Error!!!" << std::endl;
        exit(1);
    }
    std::cout << "load json success" << std::endl;

    // -----------------------------------------------------------
    // 鱼眼去畸变映射初始化（只需在配置加载后执行一次）
    // -----------------------------------------------------------
    // 这里使用 OpenCV 的 cv::fisheye 专用模型，而不是普通 pinhole 模型。
    // 原因：鱼眼镜头的畸变形式与普通镜头不同，使用 fisheye 模型更稳定。
    if (!kEnableFisheyeUndistort)
    {
        g_fisheye_map_ready = false;
        std::cout << "fisheye undistort disabled" << std::endl;
        return;
    }

    const cv::Size image_size(config.hikcamera_config.width, config.hikcamera_config.height);
    if (image_size.width <= 0 || image_size.height <= 0)
    {
        g_fisheye_map_ready = false;
        std::cerr << "fisheye init failed: invalid image size "
                  << image_size.width << "x" << image_size.height << std::endl;
        return;
    }

    // 1) 根据 K/D 与目标输出尺寸估计新的相机内参矩阵。
    // 2) R 使用单位阵（不做额外旋转）。
    // 3) balance 与 fov_scale 控制有效视场和黑边。
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        image_size,
        cv::Matx33d::eye(),
        newCameraMatrix,
        kFisheyeBalance,
        image_size,
        kFisheyeFovScale);

    // 预计算重映射表：运行时每帧只需要 remap，速度更稳定。
    // 这里使用 CV_16SC2，通常比 CV_32FC1 更省内存、更适合实时场景。
    cv::fisheye::initUndistortRectifyMap(
        K,
        D,
        cv::Matx33d::eye(),
        newCameraMatrix,
        image_size,
        CV_16SC2,
        map1,
        map2);

    g_fisheye_map_ready = !map1.empty() && !map2.empty();
    if (g_fisheye_map_ready)
    {
        std::cout << "fisheye distortion init success" << std::endl;
    }
    else
    {
        std::cerr << "fisheye distortion init failed: map is empty" << std::endl;
    }

}

detector::detector(Appconfig *config)
{
    hik_img_flag = 0;
    for (int i = 0; i < 4; ++i)
    {
        usb_img_flag[i] = 0;
    }

    if (config == nullptr)
    {
        return;
    }

    // 初始化模型路径参数
    detect_config_.xml_file_path = config->detect_config.xml_file_path;
    detect_config_.bin_file_path = config->detect_config.bin_file_path;

    // 初始化其他检测参数
    detect_config_.batch_size = config->detect_config.batch_size;
    detect_config_.h = config->detect_config.h;
    detect_config_.w = config->detect_config.w;
    detect_config_.c = config->detect_config.c;

    detect_config_.type = config->detect_config.type;
    detect_config_.width = config->detect_config.width;
    detect_config_.height = config->detect_config.height;

    detect_config_.nms_thresh = config->detect_config.nms_thresh;
    detect_config_.bbox_conf_thresh = config->detect_config.bbox_conf_thresh;
    detect_config_.merge_thresh = config->detect_config.merge_thresh;
    detect_config_.classes = config->detect_config.classes;

    detect_config_.ppocr_det_model_path = config->detect_config.ppocr_det_model_path;
    detect_config_.ppocr_rec_model_path = config->detect_config.ppocr_rec_model_path;
    detect_config_.ppocr_cls_model_path = config->detect_config.ppocr_cls_model_path;
    detect_config_.rec_char_dict_path = config->detect_config.rec_char_dict_path;

    detect_config_.class0 = config->detect_config.class0;
    detect_config_.class1 = config->detect_config.class1;
    detect_config_.class2 = config->detect_config.class2;
    detect_config_.class3 = config->detect_config.class3;
}

detector::~detector()
{
}

const std::vector<Detection> *detector::yolo_results_ptr() const
{
    return nullptr;
}

void detector::push_img(cv::Mat &grab_img, int cam_id)
{
    // cam_id: 0 = hikvision camera, 1-4 = usb camera
    if (cam_id == 0)
    {
        // Hikvision camera
        {
            // 自动加锁，离开作用域自动解锁
            std::lock_guard<std::mutex> lock(hik_img_mutex_);

            // Push to vector (maintain max_size_)
            if (input_imgs_hikvion.size() >= max_size_)
            {
                input_imgs_hikvion.erase(input_imgs_hikvion.begin());
            }
            input_imgs_hikvion.push_back(grab_img.clone());

            // Update single image buffer
            input_img_hik_ = grab_img.clone();
            hik_img_flag = 1; // Set flag indicating new image available
        }
    }
    // else if (cam_id >= 1 && cam_id <= 4)
    // {
    //     // USB camera (1-4)
    //     int usb_idx = cam_id - 1;
    //     {
    //         std::lock_guard<std::mutex> lock(usb_img_mutex_[usb_idx]);

    //         // Push to vector (maintain max_size_)
    //         if (input_imgs_usb[usb_idx].size() >= max_size_)
    //         {
    //             input_imgs_usb[usb_idx].erase(input_imgs_usb[usb_idx].begin());
    //         }
    //         input_imgs_usb[usb_idx].push_back(grab_img.clone());

    //         // Update single image buffer
    //         input_img_usb_[usb_idx] = grab_img.clone();
    //         usb_img_flag[usb_idx] = 1; // Set flag indicating new image available
    //     }
    // }
    // else
    // {
    //     std::cerr << "Invalid camera ID: " << cam_id << std::endl;
    // }
}

// 处理使用广角镜头后的畸变
cv::Mat detector::diatorion(cv::Mat &input_img)
{
    // 输入为空时直接返回空 Mat，避免后续 remap 触发异常。
    if (input_img.empty())
    {
        return cv::Mat();
    }

    // 未初始化成功时不做去畸变，返回原图拷贝，保证主流程可继续运行。
    if (!kEnableFisheyeUndistort || !g_fisheye_map_ready)
    {
        return input_img.clone();
    }

    // 运行期通过 remap 执行去畸变：
    // 1) map1/map2 在 load_config 中已按 fisheye 模型预计算。
    // 2) INTER_LINEAR 提供较平滑插值效果，适合视觉检测前处理。
    // 3) BORDER_CONSTANT 处理边缘空洞区域。
    cv::Mat undistorted_image;
    cv::remap(
        input_img,          // 输入：原始畸变图像
        undistorted_image,  // 输出：去畸变后的图像
        map1,               // 输入：x坐标映射图
        map2,               // 输入：y坐标映射图
        cv::INTER_LINEAR,   // 输入：插值方法
        cv::BORDER_CONSTANT // 输入：边界填充模式
    );
    return undistorted_image;
}

void detector::show_yolo_result(cv::Mat &show_img, const Detection &det)
{
    if (show_img.empty())
    {
        return;
    }

    // 在show_img上绘制检测结果det
    // 绘制边界框和类别标签

    // 提取边界框坐标 (假设bbox[4]为 x, y, width, height)
    int x = static_cast<int>(det.bbox[0]);
    int y = static_cast<int>(det.bbox[1]);
    int width = static_cast<int>(det.bbox[2]);
    int height = static_cast<int>(det.bbox[3]);

    // 计算右下角坐标
    int x2 = x + width;
    int y2 = y + height;

    // 确保坐标在图像范围内
    x = std::max(0, x);
    y = std::max(0, y);
    x2 = std::min(show_img.cols, x2);
    y2 = std::min(show_img.rows, y2);

    if (x2 <= x || y2 <= y)
    {
        return;
    }

    // 根据类别ID选择颜色
    cv::Scalar color;
    int class_id = static_cast<int>(det.class_id);
    switch (class_id % 5) // 5种颜色循环
    {
    case 0:
        color = cv::Scalar(0, 255, 0); // 绿色 (BGR格式)
        break;
    case 1:
        color = cv::Scalar(255, 0, 0); // 蓝色
        break;
    case 2:
        color = cv::Scalar(0, 0, 255); // 红色
        break;
    case 3:
        color = cv::Scalar(255, 255, 0); // 青色
        break;
    case 4:
        color = cv::Scalar(255, 0, 255); // 紫色
        break;
    default:
        color = cv::Scalar(0, 255, 255); // 黄色
        break;
    }

    // 绘制边界框
    int thickness = 2;
    cv::rectangle(show_img, cv::Point(x, y), cv::Point(x2, y2), color, thickness);

    // 准备标签文本 (类别 + 置信度)
    std::string label = "Class: " + std::to_string(class_id) +
                        " Conf: " + std::to_string(det.conf).substr(0, 4);

    // 获取文本大小以用于背景矩形
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5;
    int font_thickness = 1;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, font, font_scale, font_thickness, &baseline);

    // 绘制标签背景矩形
    const int text_top = std::max(0, y - text_size.height - 5);
    const int text_bottom = std::max(0, y);
    cv::rectangle(show_img,
                  cv::Point(x, text_top),
                  cv::Point(x + text_size.width, text_bottom),
                  color, -1); // 填充矩形

    // 绘制文本标签
    cv::putText(show_img, label, cv::Point(x, std::max(0, y - 5)),
                font, font_scale, cv::Scalar(255, 255, 255), font_thickness);
}

bool detector::yolo_run(cv::Mat &input_img, std::vector<Detection> &res)
{
    if (input_img.empty())
    {
        res.clear();
        return false;
    }

    std::lock_guard<std::mutex> lock(yolo_infer_mutex_);

    preprocess(input_img);
    inference();
    postprocess();

    const std::vector<Detection> *dets = yolo_results_ptr();
    if (dets == nullptr)
    {
        res.clear();
        return false;
    }

    res = *dets;
    return !res.empty();
}

void detector::show_ocr_result(void)
{
    // OCR结果显示函数
    // 这里可以实现对OCR结果的可视化，例如在图像上绘制识别的文本等
}

bool detector::get_ocr_result(void)
{
    // OCR结果处理函数
    // 这里可以实现对OCR结果的后处理，例如文本识别、结果过滤等
    return true; // 返回处理结果
}