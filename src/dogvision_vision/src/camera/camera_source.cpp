#include <dogvision_vision/camera/camera_source.hpp>

#include <chrono>
#include <iostream>
#include <thread>

CameraSource::CameraSource(const Appconfig& config)
    : camera_type_(config.camera_type)
{
    hik_params_.device_id = config.hikcamera_config.device_id;
    hik_params_.width = config.hikcamera_config.width;
    hik_params_.height = config.hikcamera_config.height;
    hik_params_.offset_x = config.hikcamera_config.offset_x;
    hik_params_.offset_y = config.hikcamera_config.offset_y;
    hik_params_.exposure = config.hikcamera_config.exposure;

    const int usb_index = config.usb_camera_index;
    usb_params_ = config.usbcamera_config[usb_index];
}

CameraSource::~CameraSource()
{
    shutdown();
}

bool CameraSource::init()
{
    if (initialized_)
    {
        return true;
    }

    if (camera_type_ == "hik")
    {
        return init_hik();
    }
    if (camera_type_ == "usb")
    {
        return init_usb();
    }

    std::cerr << "Unsupported camera type: " << camera_type_ << std::endl;
    return false;
}

bool CameraSource::init_hik()
{
    hik_ = std::make_unique<HikGrab>(hik_params_);
    hik_->Hik_init();
    initialized_ = true;
    return true;
}

bool CameraSource::init_usb()
{
    usb_capture_.open(usb_params_.device_id);
    if (!usb_capture_.isOpened())
    {
        std::cerr << "Error: USB camera failed to open device "
                  << usb_params_.device_id << std::endl;
        return false;
    }

    usb_capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    usb_capture_.set(cv::CAP_PROP_FRAME_WIDTH, usb_params_.width);
    usb_capture_.set(cv::CAP_PROP_FRAME_HEIGHT, usb_params_.height);
    usb_capture_.set(cv::CAP_PROP_FPS, usb_params_.fps);

    cv::Mat test;
    if (!usb_capture_.read(test) || test.empty())
    {
        std::cerr << "Error: USB camera opened but failed to read a frame from device "
                  << usb_params_.device_id << std::endl;
        shutdown_usb();
        return false;
    }

    initialized_ = true;
    std::cout << "USB camera initialized successfully (device "
              << usb_params_.device_id << ", " << usb_params_.width << "x"
              << usb_params_.height << " @" << usb_params_.fps << " FPS)" << std::endl;
    return true;
}

bool CameraSource::get_frame(cv::Mat& frame)
{
    frame.release();
    if (!initialized_ && !init())
    {
        return false;
    }

    if (camera_type_ == "hik")
    {
        return hik_ && hik_->get_one_frame(frame, hik_params_.device_id) && !frame.empty();
    }

    if (camera_type_ == "usb")
    {
        return usb_capture_.isOpened() && usb_capture_.read(frame) && !frame.empty();
    }

    return false;
}

bool CameraSource::recover(int max_retries)
{
    for (int i = 0; i < max_retries; ++i)
    {
        cv::Mat test;
        if (get_frame(test))
        {
            return true;
        }

        shutdown();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        if (init())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (get_frame(test))
            {
                return true;
            }
        }
    }
    return false;
}

void CameraSource::shutdown()
{
    if (camera_type_ == "hik")
    {
        if (hik_)
        {
            hik_->Hik_end();
            hik_.reset();
        }
    }
    else if (camera_type_ == "usb")
    {
        shutdown_usb();
    }
    initialized_ = false;
}

void CameraSource::shutdown_usb()
{
    if (usb_capture_.isOpened())
    {
        usb_capture_.release();
    }
}

int CameraSource::device_id() const
{
    return camera_type_ == "usb" ? usb_params_.device_id : hik_params_.device_id;
}

int CameraSource::width() const
{
    return camera_type_ == "usb" ? usb_params_.width : hik_params_.width;
}

int CameraSource::height() const
{
    return camera_type_ == "usb" ? usb_params_.height : hik_params_.height;
}

int CameraSource::fps() const
{
    return camera_type_ == "usb" ? usb_params_.fps : 0;
}
