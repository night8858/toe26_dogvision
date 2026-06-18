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
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <dogvision_vision/yolo_utils.hpp>
#include <dogvision_vision/camera/hikvision.hpp>

namespace fs = std::filesystem;

namespace
{
constexpr char kWindowName[] = "yolo_accuracy_test"; ///< 可视化窗口名称

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
 * @retval std::string 带时间戳的 AVI 视频完整路径。
 */
std::string build_video_path(const std::string& output_dir)
{
    const auto now = std::chrono::system_clock::now();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    fs::path path(output_dir);
    path /= "yolo_accuracy_" + std::to_string(millis) + ".avi";
    return path.string();
}

/**
 * @brief 从应用配置构建海康相机参数。
 * @param config 已加载的应用配置。
 * @retval s_camera_params 海康相机初始化参数。
 */
s_camera_params make_camera_params(const Appconfig& config)
{
    s_camera_params cam_params{};
    cam_params.device_id = config.hikcamera_config.device_id;
    cam_params.width = config.hikcamera_config.width;
    cam_params.height = config.hikcamera_config.height;
    cam_params.offset_x = config.hikcamera_config.offset_x;
    cam_params.offset_y = config.hikcamera_config.offset_y;
    cam_params.exposure = config.hikcamera_config.exposure;
    return cam_params;
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

    const std::string config_path = node->get_parameter("config_path").as_string();
    const bool enable_undistort = node->get_parameter("enable_undistort").as_bool();
    const std::string output_dir = node->get_parameter("output_dir").as_string();
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
    config.detect_config.nms_thresh = static_cast<float>(visual_nms_thresh);

    const std::vector<std::string> class_names = load_class_names(config);
    RCLCPP_INFO(logger, "Loaded %d classes: %s",
                config.detect_config.classes, join_class_names(class_names).c_str());
    RCLCPP_INFO(logger, "Visual NMS threshold: %.2f", visual_nms_thresh);

    detect_oponvino detector(&config);
    if (!detector.inference_init())
    {
        RCLCPP_ERROR(logger, "Failed to initialize YOLO detector");
        rclcpp::shutdown();
        return 1;
    }

    s_camera_params cam_params = make_camera_params(config);
    HikGrab hik(cam_params);
    hik.Hik_init();

    cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
    cv::VideoWriter writer;
    int frame_count = 0;
    int ret = 0;

    RCLCPP_INFO(logger, "YOLO accuracy test started. Press Q or ESC to exit.");
    RCLCPP_INFO(logger, "Video output: %s", video_path.c_str());

    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);

        cv::Mat frame;
        std::vector<Detection> dets;
        if (!run_single_detection(hik, cam_params, detector, enable_undistort, frame, dets))
        {
            RCLCPP_WARN_THROTTLE(logger, *node->get_clock(), 1000, "Failed to grab frame.");
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27)
            {
                break;
            }
            continue;
        }

        sort_raster(dets);
        const cv::Mat vis = render_yolo_result_image(dets, frame, class_names);
        if (vis.empty())
        {
            continue;
        }

        if (!writer.isOpened())
        {
            writer.open(video_path, cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
                        video_fps, vis.size(), true);
            if (!writer.isOpened())
            {
                RCLCPP_ERROR(logger, "Cannot open video writer: %s", video_path.c_str());
                ret = 1;
                break;
            }
        }

        writer.write(vis);
        ++frame_count;

        if (frame_count % 30 == 0)
        {
            RCLCPP_INFO(logger, "Processed %d frames, latest detections: %zu",
                        frame_count, dets.size());
        }

        cv::imshow(kWindowName, vis);
        const int key = cv::waitKey(1);
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
    else
    {
        RCLCPP_WARN(logger, "No video saved because no valid frame was written.");
    }

    cv::destroyWindow(kWindowName);
    hik.Hik_end();
    rclcpp::shutdown();
    return ret;
}
