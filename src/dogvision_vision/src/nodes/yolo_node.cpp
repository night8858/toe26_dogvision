#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <atomic>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include <dogvision_vision/camera/hikvision.hpp>
#include <dogvision_vision/yolo_utils.hpp>

namespace
{
constexpr int kIdleLoopHz = 20;
constexpr char kTriggerMessage[] = "start_infer";
constexpr char kGridTopic[] = "/yolo/block_grid";

std::atomic<bool> g_triggered{false};
std::atomic<bool> g_running{true};
} // namespace

/**
 * @brief 将类别名称拼接成便于日志输出的字符串。
 * @param class_names 从配置文件加载的类别名称列表。
 * @retval std::string 使用空格分隔的类别名称字符串。
 */
static std::string join_class_names(const std::vector<std::string>& class_names)
{
    std::ostringstream oss;
    for (const auto& name : class_names)
    {
        oss << name << " ";
    }
    return oss.str();
}

/**
 * @brief 处理触发消息并更新节点触发标志。
 * @param msg 收到的字符串消息。
 * @retval void
 */
static void trigger_callback(const std_msgs::msg::String::SharedPtr msg)
{
    if (msg->data == kTriggerMessage)
    {
        g_triggered.store(true);
    }
}

/**
 * @brief 运行 ROS2 YOLO 节点入口。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @retval int 进程退出码。
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("yolo_node");

    const std::string share_dir = ament_index_cpp::get_package_share_directory("dogvision_vision");
    node->declare_parameter<std::string>("config_path", share_dir + "/config/settings.json");
    node->declare_parameter<std::string>("result_topic", "/yolo/result");
    node->declare_parameter<bool>       ("show_window", false);
    node->declare_parameter<bool>       ("enable_undistort", true);
    node->declare_parameter<bool>       ("save_images", true);
    node->declare_parameter<std::string>("save_dir", share_dir + "/data/yolorun");

    const std::string config_path  = node->get_parameter("config_path").as_string();
    const std::string result_topic = node->get_parameter("result_topic").as_string();
    const bool show_window         = node->get_parameter("show_window").as_bool();
    const bool enable_undistort    = node->get_parameter("enable_undistort").as_bool();
    const bool save_images         = node->get_parameter("save_images").as_bool();
    const std::string save_dir     = node->get_parameter("save_dir").as_string();

    // 加载配置文件并初始化 YOLO 检测器
    Appconfig config;
    detect_oponvino config_loader(nullptr);
    config_loader.load_config(config, config_path);

    const std::vector<std::string> class_names = load_class_names(config);
    RCLCPP_INFO(node->get_logger(), "Loaded %d classes: %s",
                config.detect_config.classes, join_class_names(class_names).c_str());

                // 初始化 YOLO 检测器
    detect_oponvino detector(&config);
    if (!detector.inference_init())
    {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize YOLO detector");
        rclcpp::shutdown();
        return 1;
    }

    // 初始化海康相机
    s_camera_params cam_params{};
    cam_params.device_id = config.hikcamera_config.device_id;
    cam_params.width = config.hikcamera_config.width;
    cam_params.height = config.hikcamera_config.height;
    cam_params.offset_x = config.hikcamera_config.offset_x;
    cam_params.offset_y = config.hikcamera_config.offset_y;
    cam_params.exposure = config.hikcamera_config.exposure;

    HikGrab hik(cam_params);
    hik.Hik_init();

    // 创建 ROS2 发布者和订阅者
    auto latched_qos  = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    auto result_pub   = node->create_publisher<std_msgs::msg::String>(result_topic, latched_qos);
    auto grid_pub     = node->create_publisher<std_msgs::msg::String>(kGridTopic, latched_qos);
    auto trigger_sub  = node->create_subscription<std_msgs::msg::String>(
        "/yolo/trigger", rclcpp::QoS(1), trigger_callback);

    // 启动键盘输入线程监听触发命令
    std::thread keyboard_thread([]() {
        std::string line;
        while (g_running.load() && std::getline(std::cin, line))
        {
            g_triggered.store(true);
        }
    });

    RCLCPP_INFO(node->get_logger(),
                "yolo_node ready. show_window=%s, enable_undistort=%s, save_images=%s",
                show_window ? "true" : "false",
                enable_undistort ? "true" : "false",
                save_images ? "true" : "false");
    RCLCPP_INFO(node->get_logger(), "YOLO image save dir: %s", save_dir.c_str());
    RCLCPP_INFO(node->get_logger(), "Publish '%s' to /yolo/trigger, or press Enter.", kTriggerMessage);

    rclcpp::WallRate idle_rate(kIdleLoopHz);
    GridBlock block;
    reset_grid(block);

    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);

        if (!g_triggered.load())
        {
            if (show_window)
            {
                cv::waitKey(1);
            }
            idle_rate.sleep();
            continue;
        }
        g_triggered.store(false);

        RCLCPP_INFO(node->get_logger(), "Triggered: running one-frame inference...");
        cv::Mat last_frame;
        std::vector<Detection> final_dets;
        if (!run_single_detection(hik, cam_params, detector, enable_undistort, last_frame, final_dets))
        {
            RCLCPP_WARN(node->get_logger(), "Failed to grab a valid frame for YOLO inference.");
        }
        RCLCPP_INFO(node->get_logger(), "Raw detections: %zu", final_dets.size());

        sort_raster(final_dets);
        RCLCPP_INFO(node->get_logger(), "Final detections: %zu", final_dets.size());

        assign_grid_kmeans(final_dets, class_names, block);
        for (const auto& line : format_grid_lines(block))
        {
            RCLCPP_INFO(node->get_logger(), "%s", line.c_str());
        }

        // 发布结果 JSON 字符串到 ROS2 话题
        std_msgs::msg::String result_msg;
        result_msg.data = build_result_json(final_dets, class_names);
        result_pub->publish(result_msg);

        std_msgs::msg::String grid_msg;
        grid_msg.data = build_grid_json(block);
        grid_pub->publish(grid_msg);

        if (save_images)
        {
            std::string saved_path;
            if (save_yolo_result_image(final_dets, last_frame, class_names, save_dir, &saved_path))
            {
                RCLCPP_INFO(node->get_logger(), "Saved YOLO image: %s", saved_path.c_str());
            }
            else
            {
                RCLCPP_WARN(node->get_logger(), "Failed to save YOLO image to: %s", save_dir.c_str());
            }
        }

        show_viz_image(final_dets, last_frame, class_names, show_window);
        RCLCPP_INFO(node->get_logger(), "Published to %s and %s. Waiting for next trigger.",
                    result_topic.c_str(), kGridTopic);
    }

    g_running.store(false);
    if (keyboard_thread.joinable())
    {
        keyboard_thread.detach();
    }
    if (show_window)
    {
        cv::destroyAllWindows();
    }
    hik.Hik_end();
    rclcpp::shutdown();
    return 0;
}
