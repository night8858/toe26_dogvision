/**
 * @file video_replay_node.cpp
 * @brief Publish a recorded video as a ROS 2 image stream at its native rate.
 */

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("video_replay_node");
    auto logger = node->get_logger();

    node->declare_parameter<std::string>("input_path", "");
    node->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    node->declare_parameter<std::string>("frame_id", "video");
    node->declare_parameter<std::string>("eof_topic", "/video_replay/eof");
    node->declare_parameter<int>("required_subscribers", 1);
    node->declare_parameter<int>("subscriber_wait_timeout_ms", 120000);
    node->declare_parameter<int>("eof_wait_timeout_ms", 30000);

    const std::string input_path = node->get_parameter("input_path").as_string();
    const std::string image_topic = node->get_parameter("image_topic").as_string();
    const std::string frame_id = node->get_parameter("frame_id").as_string();
    const std::string eof_topic = node->get_parameter("eof_topic").as_string();
    const int required_subscribers = static_cast<int>(std::max<int64_t>(
        1, node->get_parameter("required_subscribers").as_int()));
    const int subscriber_wait_timeout_ms = static_cast<int>(std::max<int64_t>(
        1, node->get_parameter("subscriber_wait_timeout_ms").as_int()));
    const int eof_wait_timeout_ms = static_cast<int>(std::max<int64_t>(
        0, node->get_parameter("eof_wait_timeout_ms").as_int()));

    std::error_code ec;
    if (input_path.empty() || !fs::is_regular_file(input_path, ec))
    {
        RCLCPP_ERROR(logger, "input_path is not a readable video file: '%s'",
                     input_path.c_str());
        rclcpp::shutdown();
        return 1;
    }

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened())
    {
        RCLCPP_ERROR(logger, "Cannot open video decoder for: %s", input_path.c_str());
        rclcpp::shutdown();
        return 1;
    }

    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps) || fps <= 0.0)
    {
        RCLCPP_WARN(logger, "Video FPS is invalid (%.3f); using 30 Hz.", fps);
        fps = 30.0;
    }
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    const int total_frames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));

    auto image_pub = node->create_publisher<sensor_msgs::msg::Image>(
        image_topic, rclcpp::SensorDataQoS());
    const auto eof_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    auto eof_pub = node->create_publisher<std_msgs::msg::Bool>(eof_topic, eof_qos);

    RCLCPP_INFO(logger, "Video ready: %s (%dx%d, %.3f FPS, frames=%d)",
                input_path.c_str(), width, height, fps, total_frames);
    RCLCPP_INFO(logger, "Waiting for %d image subscriber(s) on %s...",
                required_subscribers, image_topic.c_str());

    const auto subscriber_deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(subscriber_wait_timeout_ms);
    rclcpp::WallRate wait_rate(20.0);
    while (rclcpp::ok() &&
           image_pub->get_subscription_count() <
               static_cast<std::size_t>(required_subscribers))
    {
        rclcpp::spin_some(node);
        if (std::chrono::steady_clock::now() >= subscriber_deadline)
        {
            RCLCPP_ERROR(logger,
                         "Timed out waiting for %d subscriber(s); only %zu connected.",
                         required_subscribers, image_pub->get_subscription_count());
            capture.release();
            rclcpp::shutdown();
            return 1;
        }
        wait_rate.sleep();
    }

    if (!rclcpp::ok())
    {
        capture.release();
        rclcpp::shutdown();
        return 0;
    }

    RCLCPP_INFO(logger, "Subscribers ready; starting playback from frame 0.");
    const auto frame_period = std::chrono::duration<double>(1.0 / fps);
    auto next_frame_time = std::chrono::steady_clock::now();
    std::size_t published_frames = 0;
    cv::Mat frame;
    while (rclcpp::ok() && capture.read(frame))
    {
        if (frame.empty())
        {
            RCLCPP_WARN(logger, "Decoder returned an empty frame; skipping it.");
            continue;
        }

        std_msgs::msg::Header header;
        header.stamp = node->now();
        header.frame_id = frame_id;
        image_pub->publish(*cv_bridge::CvImage(header, "bgr8", frame).toImageMsg());
        ++published_frames;
        rclcpp::spin_some(node);

        next_frame_time += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            frame_period);
        std::this_thread::sleep_until(next_frame_time);
    }
    capture.release();

    if (!rclcpp::ok())
    {
        rclcpp::shutdown();
        return 0;
    }

    std_msgs::msg::Bool eof_msg;
    eof_msg.data = true;
    eof_pub->publish(eof_msg);
    RCLCPP_INFO(logger, "Playback complete: published %zu frames; EOF sent on %s.",
                published_frames, eof_topic.c_str());

    const auto eof_deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(eof_wait_timeout_ms);
    while (rclcpp::ok() && image_pub->get_subscription_count() > 0 &&
           std::chrono::steady_clock::now() < eof_deadline)
    {
        rclcpp::spin_some(node);
        wait_rate.sleep();
    }
    if (image_pub->get_subscription_count() > 0)
    {
        RCLCPP_WARN(logger, "EOF wait timed out with %zu image subscriber(s) still connected.",
                    image_pub->get_subscription_count());
    }

    rclcpp::shutdown();
    return 0;
}
