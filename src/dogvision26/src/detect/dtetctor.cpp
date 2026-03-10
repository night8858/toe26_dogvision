#include <iostream>
#include "detector.hpp"
#include "common_structs.h"
#include <fstream>
#include <jsoncpp/json/json.h>
#include <string>
void load_config(Appconfig& config, std::string json_file_path)
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

        config.detect_config.ppocr_file_path = value["path"]["ppocr_file_path"].asString();
        config.detect_config.bin_file_path = value["path"]["openvino_bin_file_path"].asString();
        config.detect_config.xml_file_path = value["path"]["openvino_xml_file_path"].asString();
        
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


        config.hikcamera_config.device_id = value["camera_0"]["device_id"].asInt();
        config.hikcamera_config.exposure = value["camera_0"]["exposure"].asInt();
        config.hikcamera_config.height = value["camera_0"]["height"].asInt();
        config.hikcamera_config.width = value["camera_0"]["width"].asInt();
        config.hikcamera_config.offset_x = value["camera_0"]["offset_x"].asInt();
        config.hikcamera_config.offset_y = value["camera_0"]["offset_y"].asInt();


        config.usbcamera_config[0].device_id = value["usbcamera0"]["device_id"].asInt();
        config.usbcamera_config[0].width = value["usbcamera0"]["width"].asInt();
        config.usbcamera_config[0].height = value["usbcamera0"]["height"].asInt();

#ifdef TWO_CAMERAS
        //此处可补充多个相机的初始化
#endif
    }
    else
    {
        std::cerr << "Load Json Error!!!" << std::endl;
        exit(1);
    }
    std::cout << "load json success" << std::endl;
}


detector::detector(Appconfig* config)
{
    // 初始化模型参数
    detect_config_.xml_file_path = config->detect_config.xml_file_path;
    detect_config_.bin_file_path = config->detect_config.bin_file_path;
    detect_config_.ppocr_file_path = config->detect_config.ppocr_file_path;

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
}

detector::~detector()
{

}


void detector::push_img(cv::Mat &grab_img , int cam_id)
{
    // cam_id: 0 = hikvision camera, 1-4 = usb camera
    if (cam_id == 0)
    {
        // Hikvision camera
        {
            //自动加锁，离开作用域自动解锁
            std::lock_guard<std::mutex> lock(hik_img_mutex_);
            
            // Push to vector (maintain max_size_)
            if (input_imgs_hikvion.size() >= max_size_)
            {
                input_imgs_hikvion.erase(input_imgs_hikvion.begin());
            }
            input_imgs_hikvion.push_back(grab_img.clone());
            
            // Update single image buffer
            input_img_hik_ = grab_img.clone();
            hik_img_flag = 1;  // Set flag indicating new image available
        }
    }
    else if (cam_id >= 1 && cam_id <= 4)
    {
        // USB camera (1-4)
        int usb_idx = cam_id - 1;
        {
            std::lock_guard<std::mutex> lock(usb_img_mutex_[usb_idx]);
            
            // Push to vector (maintain max_size_)
            if (input_imgs_usb[usb_idx].size() >= max_size_)
            {
                input_imgs_usb[usb_idx].erase(input_imgs_usb[usb_idx].begin());
            }
            input_imgs_usb[usb_idx].push_back(grab_img.clone());
            
            // Update single image buffer
            input_img_usb_[usb_idx] = grab_img.clone();
            usb_img_flag[usb_idx] = 1;  // Set flag indicating new image available
        }
    }
    else
    {
        std::cerr << "Invalid camera ID: " << cam_id << std::endl;
    }
}


void detector::show_yolo_result(cv::Mat &show_img , Detection &det)
{
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
    
    // 根据类别ID选择颜色
    cv::Scalar color;
    int class_id = static_cast<int>(det.class_id);
    switch (class_id % 5)  // 5种颜色循环
    {
        case 0:
            color = cv::Scalar(0, 255, 0);      // 绿色 (BGR格式)
            break;
        case 1:
            color = cv::Scalar(255, 0, 0);      // 蓝色
            break;
        case 2:
            color = cv::Scalar(0, 0, 255);      // 红色
            break;
        case 3:
            color = cv::Scalar(255, 255, 0);    // 青色
            break;
        case 4:
            color = cv::Scalar(255, 0, 255);    // 紫色
            break;
        default:
            color = cv::Scalar(0, 255, 255);    // 黄色
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
    cv::rectangle(show_img, 
                  cv::Point(x, y - text_size.height - 5),
                  cv::Point(x + text_size.width, y),
                  color, -1);  // 填充矩形
    
    // 绘制文本标签
    cv::putText(show_img, label, cv::Point(x, y - 5), 
                font, font_scale, cv::Scalar(255, 255, 255), font_thickness);
}

bool detector::get_yolo_result(cv::Mat &input_img , std::vector<Detection> &res)
{
    // YOLO结果处理函数
    // 这里可以实现对YOLO结果的后处理，例如非极大值抑制、结果过滤等
    return true;  // 返回处理结果
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
    return true;  // 返回处理结果
}