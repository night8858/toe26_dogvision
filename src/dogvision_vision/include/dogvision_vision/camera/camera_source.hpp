#pragma once

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include <dogvision_vision/camera/hikvision.hpp>
#include <dogvision_vision/common_structs.h>

/**
 * @brief Runtime camera adapter selected by settings.json.
 */
class CameraSource
{
public:
    explicit CameraSource(const Appconfig& config);
    ~CameraSource();

    bool init();
    bool get_frame(cv::Mat& frame);
    bool recover(int max_retries = 5);
    void shutdown();

    const std::string& type_name() const { return camera_type_; }
    int device_id() const;
    int width() const;
    int height() const;
    int fps() const;

private:
    bool init_hik();
    bool init_usb();
    void shutdown_usb();

    std::string camera_type_;
    s_camera_params hik_params_{};
    s_usbcamera_params usb_params_{};
    std::unique_ptr<HikGrab> hik_;
    cv::VideoCapture usb_capture_;
    bool initialized_ = false;
};
