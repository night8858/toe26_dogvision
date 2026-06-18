/**
 * @file yolo_node.cpp
 * @brief ROS2 YOLO 检测节点
 *
 * 订阅 /yolo/trigger 话题或来自 stdin 的键盘输入触发单帧 YOLO 推理，
 * 将检测结果和 2×4 网格分类结果分别发布到 /yolo/result 和 /yolo/block_grid。
 *
 * 支持：
 *   - 海康相机实时取流
 *   - 鱼眼去畸变（可选）
 *   - 检测结果可视化窗口（可选）
 *   - 结果图片自动保存（可选）
 */

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
constexpr int kIdleLoopHz = 20;         ///< 空闲状态下的循环频率（Hz）
constexpr char kTriggerMessage[] = "start_infer"; ///< 触发推理的消息内容
constexpr char kGridTopic[] = "/yolo/block_grid"; ///< 网格结果话题名

std::atomic<bool> g_triggered{false};  ///< 触发标志，由话题回调或键盘线程置位
std::atomic<bool> g_running{true};     ///< 运行标志，用于控制键盘线程退出
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
/**
 * @brief 运行 ROS2 YOLO 节点入口
 *
 * 启动流程：
 *   1. 初始化 ROS2 节点并声明参数
 *   2. 从 JSON 配置文件加载检测参数
 *   3. 初始化 YOLO OpenVINO 检测器
 *   4. 初始化海康相机
 *   5. 创建 ROS2 发布者/订阅者
 *   6. 启动键盘输入线程
 *   7. 进入主循环：等待触发 → 单帧推理 → 发布结果
 *
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return int 进程退出码
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("yolo_node");

    // ── 1. 声明并读取 ROS2 参数 ──
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

    // ── 2. 加载配置文件 ──
    Appconfig config;
    detect_oponvino config_loader(nullptr);
    config_loader.load_config(config, config_path);

    // ── 3. 加载类别名称 ──
    const std::vector<std::string> class_names = load_class_names(config);
    RCLCPP_INFO(node->get_logger(), "Loaded %d classes: %s",
                config.detect_config.classes, join_class_names(class_names).c_str());

    // ── 4. 初始化 YOLO OpenVINO 检测器 ──
    detect_oponvino detector(&config);
    if (!detector.inference_init())
    {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize YOLO detector");
        rclcpp::shutdown();
        return 1;
    }

    // ── 5. 初始化海康相机 ──
    s_camera_params cam_params{};
    cam_params.device_id = config.hikcamera_config.device_id;
    cam_params.width = config.hikcamera_config.width;
    cam_params.height = config.hikcamera_config.height;
    cam_params.offset_x = config.hikcamera_config.offset_x;
    cam_params.offset_y = config.hikcamera_config.offset_y;
    cam_params.exposure = config.hikcamera_config.exposure;

    HikGrab hik(cam_params);
    hik.Hik_init();

    // ── 6. 创建 ROS2 发布者和订阅者 ──
    // 使用 transient_local + reliable QoS 确保新订阅者能收到最后一条消息
    auto latched_qos  = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    auto result_pub   = node->create_publisher<std_msgs::msg::String>(result_topic, latched_qos);
    auto grid_pub     = node->create_publisher<std_msgs::msg::String>(kGridTopic, latched_qos);
    auto trigger_sub  = node->create_subscription<std_msgs::msg::String>(
        "/yolo/trigger", rclcpp::QoS(1), trigger_callback);

    // ── 7. 启动键盘输入线程，监听标准输入的触发命令 ──
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

    // ── 8. 主循环：等待触发 → 推理 → 发布 ──
    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);

        // 等待触发信号
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

        // ── 执行单帧 YOLO 推理 ──
        RCLCPP_INFO(node->get_logger(), "Triggered: running one-frame inference...");
        cv::Mat last_frame;
        std::vector<Detection> final_dets;
        if (!run_single_detection(hik, cam_params, detector, enable_undistort, last_frame, final_dets))
        {
            RCLCPP_WARN(node->get_logger(), "Failed to grab a valid frame for YOLO inference.");
        }
        RCLCPP_INFO(node->get_logger(), "Raw detections: %zu", final_dets.size());

        // ── 结果排序与网格分配 ──
        sort_raster(final_dets);
        RCLCPP_INFO(node->get_logger(), "Final detections: %zu", final_dets.size());

        assign_grid_kmeans(final_dets, class_names, block);
        for (const auto& line : format_grid_lines(block))
        {
            RCLCPP_INFO(node->get_logger(), "%s", line.c_str());
        }

        // ── 发布 JSON 结果到 ROS2 话题 ──
        std_msgs::msg::String result_msg;
        result_msg.data = build_result_json(final_dets, class_names);
        result_pub->publish(result_msg);

        std_msgs::msg::String grid_msg;
        grid_msg.data = build_grid_json(block);
        grid_pub->publish(grid_msg);

        // ── 保存结果图片（可选） ──
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

        // ── 可视化显示（可选） ──
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
