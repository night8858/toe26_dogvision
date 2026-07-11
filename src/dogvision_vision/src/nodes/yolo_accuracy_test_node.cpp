/**
 * @file yolo_accuracy_test_node.cpp
 * @brief YOLO 视觉准确性测试节点
 *
 * 从海康相机连续取流，对每帧执行 YOLO 检测，将标注了检测框的结果图
 * 录制为视频文件，供离线评估检测准确性使用。
 *
 * 支持：
 *   - 实时视频录制（MP4 格式）
 *   - 鱼眼去畸变（可选）
 *   - 自定义 NMS 阈值（通过参数 visual_nms_thresh 覆盖配置文件）
 */

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include <dogvision_vision/camera/camera_source.hpp>
#include <dogvision_vision/yolo_utils.hpp>

namespace fs = std::filesystem;

namespace
{
constexpr char kWindowName[] = "yolo_accuracy_test"; ///< 可视化窗口名称

struct LatestTestFrame
{
    std::mutex mutex;
    cv::Mat frame;
    bool has_frame = false;
};

void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg,
                    const std::shared_ptr<LatestTestFrame>& buffer,
                    const rclcpp::Logger& logger)
{
    try
    {
        auto cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
        std::lock_guard<std::mutex> lock(buffer->mutex);
        buffer->frame = cv_ptr->image.clone();
        buffer->has_frame = true;
    }
    catch (const cv_bridge::Exception& e)
    {
        RCLCPP_WARN(logger, "Failed to convert replay frame: %s", e.what());
    }
}

bool take_latest_frame(const std::shared_ptr<LatestTestFrame>& buffer, cv::Mat& frame)
{
    std::lock_guard<std::mutex> lock(buffer->mutex);
    if (!buffer->has_frame || buffer->frame.empty())
    {
        frame.release();
        return false;
    }
    frame = buffer->frame.clone();
    buffer->frame.release();
    buffer->has_frame = false;
    return true;
}

bool run_detection_on_frame(cv::Mat input,
                            detect_oponvino& detector,
                            bool enable_undistort,
                            cv::Mat& processed_frame,
                            std::vector<Detection>& dets)
{
    if (input.empty())
    {
        processed_frame.release();
        dets.clear();
        return false;
    }
    if (enable_undistort)
        input = detector.diatorion(input);
    processed_frame = input;
    dets.clear();
    detector.yolo_run(processed_frame, dets);
    return true;
}

/**
 * @brief 将类别名称拼接成便于日志输出的字符串。
 * @param class_names 从配置文件加载的类别名称列表。
 * @retval std::string 使用空格分隔的类别名称字符串。
 */
std::string join_class_names(const std::vector<std::string>& class_names)
{
    std::ostringstream oss;
    for (const auto& name : class_names)
    {
        oss << name << " ";
    }
    return oss.str();
}

/**
 * @brief 使用当前时间构建测试视频输出路径。
 * @param output_dir 视频输出目录。
 * @retval std::string 带时间戳的 MP4 视频完整路径。
 */
std::string build_video_path(const std::string& output_dir)
{
    const auto now = std::chrono::system_clock::now();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    fs::path path(output_dir);
    path /= "yolo_accuracy_" + std::to_string(millis) + ".mp4";
    return path.string();
}

} // namespace

/**
 * @brief 运行 YOLO 视觉准确性测试节点。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @retval int 进程退出码。
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("yolo_accuracy_test_node");
    auto logger = node->get_logger();

    const std::string share_dir = ament_index_cpp::get_package_share_directory("dogvision_vision");
    node->declare_parameter<std::string>("config_path", share_dir + "/config/settings.json");
    node->declare_parameter<bool>("enable_undistort", true);
    node->declare_parameter<std::string>("output_dir", share_dir + "/data/yolotest");
    node->declare_parameter<double>("video_fps", 20.0);
    node->declare_parameter<double>("visual_nms_thresh", 0.7);
    node->declare_parameter<bool>("show_window", true);
    node->declare_parameter<std::string>("image_source", "camera");
    node->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    node->declare_parameter<std::string>("eof_topic", "/video_replay/eof");

    const std::string config_path = node->get_parameter("config_path").as_string();
    const bool enable_undistort_param = node->get_parameter("enable_undistort").as_bool();
    const std::string output_dir = node->get_parameter("output_dir").as_string();
    const bool show_window = node->get_parameter("show_window").as_bool();
    const std::string image_source = node->get_parameter("image_source").as_string();
    const std::string image_topic = node->get_parameter("image_topic").as_string();
    const std::string eof_topic = node->get_parameter("eof_topic").as_string();
    const bool use_topic_image = image_source == "topic";
    if (image_source != "camera" && !use_topic_image)
    {
        RCLCPP_ERROR(logger, "Unsupported image_source '%s'. Use 'camera' or 'topic'.",
                     image_source.c_str());
        rclcpp::shutdown();
        return 1;
    }
    double video_fps = node->get_parameter("video_fps").as_double();
    const double visual_nms_thresh = std::clamp(
        node->get_parameter("visual_nms_thresh").as_double(), 0.0, 1.0);
    if (video_fps <= 0.0)
    {
        RCLCPP_WARN(logger, "video_fps <= 0, reset to 20.0");
        video_fps = 20.0;
    }

    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec)
    {
        RCLCPP_ERROR(logger, "Cannot create output_dir '%s': %s",
                     output_dir.c_str(), ec.message().c_str());
        rclcpp::shutdown();
        return 1;
    }
    const std::string video_path = build_video_path(output_dir);

    Appconfig config;
    detect_oponvino config_loader(nullptr);
    config_loader.load_config(config, config_path);
    node->declare_parameter<bool>("save_video", config.detect_config.save_yolo_test_video);
    const bool save_video = node->get_parameter("save_video").as_bool();
    config.detect_config.nms_thresh = static_cast<float>(visual_nms_thresh);
    const bool enable_undistort =
        enable_undistort_param && config.detect_config.enable_undistort;

    const std::vector<std::string> class_names = load_class_names(config);
    RCLCPP_INFO(logger, "Loaded %d classes: %s",
                config.detect_config.classes, join_class_names(class_names).c_str());
    RCLCPP_INFO(logger, "Visual NMS threshold: %.2f", visual_nms_thresh);
    if (enable_undistort_param && !config.detect_config.enable_undistort)
    {
        RCLCPP_INFO(logger,
                    "Undistort disabled by settings.json lens_distortion.enable_undistort.");
    }

    detect_oponvino detector(&config);
    if (!detector.inference_init())
    {
        RCLCPP_ERROR(logger, "Failed to initialize YOLO detector");
        rclcpp::shutdown();
        return 1;
    }

    std::unique_ptr<CameraSource> camera;
    std::shared_ptr<LatestTestFrame> topic_buffer;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr eof_sub;
    std::atomic<bool> replay_eof{false};
    if (use_topic_image)
    {
        topic_buffer = std::make_shared<LatestTestFrame>();
        image_sub = node->create_subscription<sensor_msgs::msg::Image>(
            image_topic, rclcpp::SensorDataQoS(),
            [topic_buffer, logger](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                image_callback(msg, topic_buffer, logger);
            });
        const auto eof_qos =
            rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
        eof_sub = node->create_subscription<std_msgs::msg::Bool>(
            eof_topic, eof_qos,
            [&replay_eof](const std_msgs::msg::Bool::ConstSharedPtr msg) {
                if (msg->data)
                    replay_eof.store(true);
            });
        RCLCPP_INFO(logger, "Image input: topic %s; EOF topic: %s",
                    image_topic.c_str(), eof_topic.c_str());
    }
    else
    {
        camera = std::make_unique<CameraSource>(config);
        if (!camera->init())
        {
            RCLCPP_ERROR(logger, "Failed to initialize %s camera",
                         camera->type_name().c_str());
            rclcpp::shutdown();
            return 1;
        }
        RCLCPP_INFO(logger, "Camera: type=%s device=%d %dx%d",
                    camera->type_name().c_str(), camera->device_id(),
                    camera->width(), camera->height());
    }

    if (show_window)
        cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
    cv::VideoWriter writer;
    int frame_count = 0;
    int ret = 0;

    RCLCPP_INFO(logger, "YOLO accuracy test started. Press Q or ESC to exit.");
    if (save_video)
        RCLCPP_INFO(logger, "Video output: %s", video_path.c_str());
    else
        RCLCPP_INFO(logger, "Video saving disabled by settings.");

    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);

        cv::Mat frame;
        std::vector<Detection> dets;
        bool detection_ok = false;
        if (use_topic_image)
        {
            cv::Mat topic_frame;
            if (take_latest_frame(topic_buffer, topic_frame))
            {
                detection_ok = run_detection_on_frame(
                    topic_frame, detector, enable_undistort, frame, dets);
            }
            else if (replay_eof.load())
            {
                RCLCPP_INFO(logger, "Replay EOF received; YOLO test is complete.");
                break;
            }
        }
        else
        {
            detection_ok = run_single_detection(
                *camera, detector, enable_undistort, frame, dets);
        }

        if (!detection_ok)
        {
            if (!use_topic_image)
                RCLCPP_WARN_THROTTLE(logger, *node->get_clock(), 1000,
                                     "Failed to grab frame.");
            const int key = show_window ? cv::waitKey(1) : -1;
            if (key == 'q' || key == 'Q' || key == 27)
            {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        sort_raster(dets);
        const cv::Mat vis = render_yolo_result_image(dets, frame, class_names);
        if (vis.empty())
        {
            continue;
        }

        if (save_video && !writer.isOpened())
        {
            writer.open(video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                        video_fps, vis.size(), true);
            if (!writer.isOpened())
            {
                RCLCPP_ERROR(logger, "Cannot open video writer: %s", video_path.c_str());
                ret = 1;
                break;
            }
        }

        if (save_video)
            writer.write(vis);
        ++frame_count;

        if (frame_count % 30 == 0)
        {
            RCLCPP_INFO(logger, "Processed %d frames, latest detections: %zu",
                        frame_count, dets.size());
        }

        if (show_window)
            cv::imshow(kWindowName, vis);
        const int key = show_window ? cv::waitKey(1) : -1;
        if (key == 'q' || key == 'Q' || key == 27)
        {
            RCLCPP_INFO(logger, "Exit by user key.");
            break;
        }
    }

    if (writer.isOpened())
    {
        writer.release();
        RCLCPP_INFO(logger, "Saved video: %s", video_path.c_str());
    }
    else if (save_video)
    {
        RCLCPP_WARN(logger, "No video saved because no valid frame was written.");
    }

    if (show_window)
        cv::destroyWindow(kWindowName);
    if (camera)
        camera->shutdown();
    (void)image_sub;
    (void)eof_sub;
    rclcpp::shutdown();
    return ret;
}
