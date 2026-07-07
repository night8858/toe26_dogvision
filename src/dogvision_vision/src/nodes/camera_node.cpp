/**
 * @file camera_node.cpp
 * @brief ROS2 共享相机采集节点
 *
 * vision.launch 中只允许本节点打开物理相机，YOLO/OCR 通过图像话题共享同一帧流，
 * 避免多个进程同时占用 USB 相机导致其中一个节点启动失败。
 */

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <chrono>
#include <string>

#include <opencv2/opencv.hpp>

#include <dogvision_vision/camera/camera_source.hpp>
#include <dogvision_vision/ocr_detect.hpp>

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("camera_node");
    auto logger = node->get_logger();

    const std::string share_dir =
        ament_index_cpp::get_package_share_directory("dogvision_vision");
    node->declare_parameter<std::string>("config_path", share_dir + "/config/settings.json");
    node->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    node->declare_parameter<std::string>("frame_id", "camera");
    node->declare_parameter<double>("publish_rate", 30.0);
    node->declare_parameter<int>("recover_max_retries", 5);
    node->declare_parameter<bool>("auto_suspend_stream", true);
    node->declare_parameter<int>("idle_grace_ms", 500);

    const std::string config_path = node->get_parameter("config_path").as_string();
    const std::string image_topic = node->get_parameter("image_topic").as_string();
    const std::string frame_id = node->get_parameter("frame_id").as_string();
    double publish_rate = node->get_parameter("publish_rate").as_double();
    const int recover_max_retries =
        node->get_parameter("recover_max_retries").as_int();
    const bool auto_suspend_stream =
        node->get_parameter("auto_suspend_stream").as_bool();
    const int idle_grace_ms =
        static_cast<int>(std::max<int64_t>(
            0, node->get_parameter("idle_grace_ms").as_int()));

    Appconfig config;
    detect_det_ppocr config_loader(nullptr);
    config_loader.load_config(config, config_path);

    CameraSource camera(config);
    if (!camera.init())
    {
        RCLCPP_ERROR(logger, "Failed to initialize %s camera",
                     camera.type_name().c_str());
        rclcpp::shutdown();
        return 1;
    }

    if (publish_rate <= 0.0)
    {
        publish_rate = camera.fps() > 0 ? static_cast<double>(camera.fps()) : 30.0;
    }

    auto image_pub = node->create_publisher<sensor_msgs::msg::Image>(
        image_topic, rclcpp::SensorDataQoS());

    RCLCPP_INFO(logger, "Shared camera ready: type=%s device=%d %dx%d @ %.2f Hz",
                camera.type_name().c_str(), camera.device_id(),
                camera.width(), camera.height(), publish_rate);
    RCLCPP_INFO(logger, "Publishing raw BGR frames to %s", image_topic.c_str());
    RCLCPP_INFO(logger, "Auto suspend stream: %s (idle_grace_ms=%d)",
                auto_suspend_stream ? "enabled" : "disabled", idle_grace_ms);

    rclcpp::WallRate rate(publish_rate);
    const auto idle_grace = std::chrono::milliseconds(idle_grace_ms);
    std::chrono::steady_clock::time_point no_subscriber_since;
    bool has_no_subscriber_since = false;
    bool stream_suspended = false;
    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);

        if (auto_suspend_stream)
        {
            const size_t subscriber_count = image_pub->get_subscription_count();
            const auto now = std::chrono::steady_clock::now();
            if (subscriber_count == 0)
            {
                if (!has_no_subscriber_since)
                {
                    no_subscriber_since = now;
                    has_no_subscriber_since = true;
                }
                if (!stream_suspended && now - no_subscriber_since >= idle_grace)
                {
                    if (camera.pause_stream())
                    {
                        stream_suspended = true;
                        RCLCPP_INFO(logger,
                                    "No image subscribers; camera stream suspended.");
                    }
                    else
                    {
                        RCLCPP_WARN(logger, "Failed to suspend camera stream.");
                    }
                }
                rate.sleep();
                continue;
            }

            has_no_subscriber_since = false;
            if (stream_suspended)
            {
                RCLCPP_INFO(logger,
                            "Image subscriber detected; resuming camera stream.");
                if (!camera.resume_stream())
                {
                    RCLCPP_WARN(logger,
                                "Failed to resume camera stream, trying recovery...");
                    if (!camera.recover(std::max(1, recover_max_retries)))
                    {
                        RCLCPP_ERROR(logger,
                                     "Camera recover failed while resuming stream.");
                        rate.sleep();
                        continue;
                    }
                }
                stream_suspended = false;
            }
        }

        cv::Mat frame;
        if (!camera.get_frame(frame))
        {
            // 采集节点负责断线重连，推理节点只需要等待下一张有效图像。
            RCLCPP_WARN(logger, "Camera frame grab failed, trying to recover...");
            if (!camera.recover(std::max(1, recover_max_retries)))
            {
                RCLCPP_ERROR(logger, "Camera recover failed; retrying after one cycle.");
                rate.sleep();
                continue;
            }
            continue;
        }

        std_msgs::msg::Header header;
        header.stamp = node->now();
        header.frame_id = frame_id;
        auto msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
        image_pub->publish(*msg);

        rate.sleep();
    }

    camera.shutdown();
    rclcpp::shutdown();
    return 0;
}
