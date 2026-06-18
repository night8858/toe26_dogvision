#ifndef TOE_USBCAM_H
#define TOE_USBCAM_H

#include <array>
#include <iostream>
#include <mutex>
#include <string>
#include <opencv2/opencv.hpp>

/**
 * @brief USB 相机封装类，基于 OpenCV VideoCapture 实现
 */
class usb_camera
{

public:
    double FPS = 0.0; ///< 实际帧率统计

    int frame_width = 640;  ///< 帧宽度（像素）
    int frame_height = 480; ///< 帧高度（像素）

    /**
     * @brief 构造 USB 相机封装对象。
     * @param 无
     * @retval 无
     */
    usb_camera() = default;  // 构造函数

    /**
     * @brief 析构 USB 相机封装对象。
     * @param 无
     * @retval 无
     */
    ~usb_camera() = default; // 析构函数

    std::array<cv::Mat, 2> usb_frame_array;    ///< 双缓冲帧数组（交替写入）
    std::array<std::mutex, 2> usb_mutex_array; ///< 对应帧缓冲的互斥锁

    /**
     * @brief 初始化 USB 相机采集对象。
     * @param capture 需要初始化的 OpenCV 采集对象。
     * @retval bool 初始化成功时返回 true。
     */
    bool usb_camera_init(cv::VideoCapture &capture);

    /**
     * @brief 显示最近一帧 USB 相机图像。
     * @param 无
     * @retval void
     */
    void usb_camera_show_frame(void);

    /**
     * @brief 从 USB 相机采集对象读取一帧图像。
     * @param capture OpenCV 采集对象。
     * @param frame 输出图像帧。
     * @retval bool 成功获取图像帧时返回 true。
     */
    bool usb_camera_get_frame(cv::VideoCapture &capture, cv::Mat &frame);

private:
    cv::VideoCapture cap;        ///< OpenCV 视频采集对象
    std::mutex usb_frame_mutex;  ///< 内部帧互斥锁
    cv::Mat frame;               ///< 内部帧暂存
};


/**
 * @brief 判断图像帧是否满足颜色规则。
 * @param frame 输入图像帧。
 * @retval bool 满足颜色规则时返回 true。
 */
bool color_judge(cv::Mat &frame);

/**
 * @brief 将面积值限制在指定范围内。
 * @param input 输入面积值。
 * @param limit_min 允许的最小值。
 * @param limit_max 允许的最大值。
 * @retval int 限幅后的面积值。
 */
int rect_area_limit(int input, int limit_min, int limit_max);

extern std::array<cv::Mat, 2> usb_frame_array;    // 存放USB相机帧
extern std::array<std::mutex, 2> usb_mutex_array; // 存放USB相机帧互斥锁
#endif
