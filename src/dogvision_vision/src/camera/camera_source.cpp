#include <dogvision_vision/camera/camera_source.hpp>

#include <chrono>
#include <iostream>
#include <thread>

CameraSource::CameraSource(const Appconfig& config)
    : camera_type_(config.camera_type)
{
    // Hik/MVS is currently disabled. Keep this block for future restore.
    // hik_params_.device_id = config.hikcamera_config.device_id;
    // hik_params_.width = config.hikcamera_config.width;
    // hik_params_.height = config.hikcamera_config.height;
    // hik_params_.offset_x = config.hikcamera_config.offset_x;
    // hik_params_.offset_y = config.hikcamera_config.offset_y;
    // hik_params_.exposure = config.hikcamera_config.exposure;

    const int usb_index = config.usb_camera_index;
    usb_params_ = config.usbcamera_config[usb_index];
}

CameraSource::~CameraSource()
{
    shutdown();
}

bool CameraSource::init()
{
    if (stream_paused_)
    {
        return resume_stream();
    }
    if (initialized_)
    {
        return true;
    }

    if (camera_type_ == "hik")
    {
        std::cerr << "Hik/MVS camera support is disabled. Set camera.type to \"usb\"."
                  << std::endl;
        return false;
    }
    if (camera_type_ == "usb")
    {
        return init_usb();
    }

    std::cerr << "Unsupported camera type: " << camera_type_ << std::endl;
    return false;
}

// Hik/MVS is currently disabled. Keep this implementation for future restore.
// bool CameraSource::init_hik()
// {
//     hik_ = std::make_unique<HikGrab>(hik_params_);
//     hik_->Hik_init();
//     initialized_ = true;
//     stream_paused_ = false;
//     return true;
// }

bool CameraSource::init_usb()
{
    const bool use_device_path = !usb_params_.device_path.empty();
    const std::string usb_source = use_device_path ? usb_params_.device_path : "0";

    if (use_device_path)
    {
        usb_capture_.open(usb_params_.device_path);
    }
    else
    {
        usb_capture_.open(0);
    }

    if (!usb_capture_.isOpened())
    {
        std::cerr << "Error: USB camera failed to open source "
                  << usb_source << std::endl;
        return false;
    }

    usb_capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    usb_capture_.set(cv::CAP_PROP_FRAME_WIDTH, usb_params_.width);
    usb_capture_.set(cv::CAP_PROP_FRAME_HEIGHT, usb_params_.height);
    usb_capture_.set(cv::CAP_PROP_FPS, usb_params_.fps);

    const bool manual_exposure = usb_params_.exposure >= 0.0;
    const double auto_exposure_value = manual_exposure ? 0.25 : 0.75;
    const bool auto_exposure_set =
        usb_capture_.set(cv::CAP_PROP_AUTO_EXPOSURE, auto_exposure_value);
    if (!auto_exposure_set)
    {
        std::cerr << "Warning: USB camera driver rejected "
                  << (manual_exposure ? "manual" : "automatic")
                  << " exposure mode; continuing with driver mode." << std::endl;
    }
    if (manual_exposure)
    {
        const bool exposure_set =
            usb_capture_.set(cv::CAP_PROP_EXPOSURE, usb_params_.exposure);
        const double actual_exposure = usb_capture_.get(cv::CAP_PROP_EXPOSURE);
        if (!exposure_set)
        {
            std::cerr << "Warning: USB camera driver rejected exposure "
                      << usb_params_.exposure << " (readback "
                      << actual_exposure << "); continuing with driver value."
                      << std::endl;
        }
        else
        {
            std::cout << "USB camera exposure requested="
                      << usb_params_.exposure << ", readback="
                      << actual_exposure << std::endl;
        }
    }
    else
    {
        std::cout << "USB camera automatic exposure requested, readback="
                  << usb_capture_.get(cv::CAP_PROP_AUTO_EXPOSURE) << std::endl;
    }

    if (usb_params_.brightness >= 0.0)
    {
        const bool brightness_set =
            usb_capture_.set(cv::CAP_PROP_BRIGHTNESS, usb_params_.brightness);
        const double actual_brightness =
            usb_capture_.get(cv::CAP_PROP_BRIGHTNESS);
        if (!brightness_set)
        {
            std::cerr << "Warning: USB camera driver rejected brightness "
                      << usb_params_.brightness << " (readback "
                      << actual_brightness << "); continuing with driver value."
                      << std::endl;
        }
        else
        {
            std::cout << "USB camera brightness requested="
                      << usb_params_.brightness << ", readback="
                      << actual_brightness << std::endl;
        }
    }

    if (usb_params_.contrast >= 0.0)
    {
        const bool contrast_set =
            usb_capture_.set(cv::CAP_PROP_CONTRAST, usb_params_.contrast);
        const double actual_contrast = usb_capture_.get(cv::CAP_PROP_CONTRAST);
        if (!contrast_set)
        {
            std::cerr << "Warning: USB camera driver rejected contrast "
                      << usb_params_.contrast << " (readback "
                      << actual_contrast << "); continuing with driver value."
                      << std::endl;
        }
        else
        {
            std::cout << "USB camera contrast requested="
                      << usb_params_.contrast << ", readback="
                      << actual_contrast << std::endl;
        }
    }

    cv::Mat test;
    if (!usb_capture_.read(test) || test.empty())
    {
        std::cerr << "Error: USB camera opened but failed to read a frame from source "
                  << usb_source << std::endl;
        shutdown_usb();
        return false;
    }

    initialized_ = true;
    stream_paused_ = false;
    std::cout << "USB camera initialized successfully (source "
              << usb_source << ", " << usb_params_.width << "x"
              << usb_params_.height << " @" << usb_params_.fps << " FPS)" << std::endl;
    return true;
}

bool CameraSource::get_frame(cv::Mat& frame)
{
    frame.release();
    if (stream_paused_ && !resume_stream())
    {
        return false;
    }
    if (!initialized_ && !init())
    {
        return false;
    }

    if (camera_type_ == "hik")
    {
        std::cerr << "Hik/MVS camera support is disabled. Set camera.type to \"usb\"."
                  << std::endl;
        return false;
    }

    if (camera_type_ == "usb")
    {
        if (!usb_capture_.isOpened() || !usb_capture_.read(frame) || frame.empty())
        {
            return false;
        }
        if (usb_params_.software_brightness_offset != 0.0)
        {
            frame.convertTo(
                frame, -1, 1.0, usb_params_.software_brightness_offset);
        }
        return true;
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
        // Hik/MVS is currently disabled. Restore Hik_end() here if needed.
    }
    else if (camera_type_ == "usb")
    {
        shutdown_usb();
    }
    initialized_ = false;
    stream_paused_ = false;
}

void CameraSource::shutdown_usb()
{
    if (usb_capture_.isOpened())
    {
        usb_capture_.release();
    }
}

bool CameraSource::pause_stream()
{
    if (!initialized_ || stream_paused_)
    {
        return true;
    }

    if (camera_type_ == "hik")
    {
        std::cerr << "Hik/MVS camera support is disabled. Set camera.type to \"usb\"."
                  << std::endl;
        return false;
    }

    if (camera_type_ == "usb")
    {
        shutdown_usb();
        initialized_ = false;
        stream_paused_ = true;
        return true;
    }

    return false;
}

bool CameraSource::resume_stream()
{
    if (!stream_paused_)
    {
        return initialized_ || init();
    }

    if (camera_type_ == "hik")
    {
        std::cerr << "Hik/MVS camera support is disabled. Set camera.type to \"usb\"."
                  << std::endl;
        return false;
    }

    if (camera_type_ == "usb")
    {
        stream_paused_ = false;
        return init_usb();
    }

    return false;
}

int CameraSource::device_id() const
{
    return usb_params_.device_id;
}

int CameraSource::width() const
{
    return usb_params_.width;
}

int CameraSource::height() const
{
    return usb_params_.height;
}

int CameraSource::fps() const
{
    return camera_type_ == "usb" ? usb_params_.fps : 0;
}
