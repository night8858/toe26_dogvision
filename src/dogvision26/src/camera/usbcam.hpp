#ifndef TOE_USBCAM_H
#define TOE_USBCAM_H

#include "usbcam.hpp"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

class usb_camera
{

public:
    double FPS = 0.0;

    int frame_width = 640;
    int frame_height = 480;

    usb_camera() = default;  // 构造函数
    ~usb_camera() = default; // 析构函数

    std::array<cv::Mat, 2> usb_frame_array;    // 存放USB相机帧
    std::array<std::mutex, 2> usb_mutex_array; // 存放USB相机帧互斥锁

    bool usb_camera_init(cv::VideoCapture &capture);
    void usb_camera_show_frame(void);
    bool usb_camera_get_frame(cv::VideoCapture &capture, cv::Mat &frame);
    // void usb_camera_detect(cv::Mat &frame, cv::Mat &result , const nlohmann::json &input_json);

private:
    cv::VideoCapture cap;
    std::mutex usb_frame_mutex;
    cv::Mat frame;
};


bool color_judge(cv::Mat &frame);
int rect_area_limit(int input, int limit_min, int limit_max);

extern std::array<cv::Mat, 2> usb_frame_array;    // 存放USB相机帧
extern std::array<std::mutex, 2> usb_mutex_array; // 存放USB相机帧互斥锁
#endif