#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <dogvision_vision/yolo_utils.hpp>

namespace
{

    constexpr double kInferDurationSec = 1.0;
    constexpr int kIdleLoopHz = 20;
    constexpr char kTriggerMessage[] = "start_infer";
    constexpr char kGridTopic[] = "/yolo/block_grid";

    static std::atomic<bool> g_triggered{false};

} // namespace

void trigger_callback(const std_msgs::String::ConstPtr &msg)
{
    if (msg->data == kTriggerMessage)
        g_triggered.store(true);
}

// ================================================================
//  main
// ================================================================
int main(int argc, char **argv)
{
    ros::init(argc, argv, "yolo_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // ---- 参数 ----
    std::string config_path;
    std::string result_topic;
    bool show_window = true;
    bool enable_undistort = true;

    pnh.param<std::string>("config_path", config_path,
                           ros::package::getPath("dogvision_vision") + "/config/settings.json");
    pnh.param<std::string>("result_topic", result_topic, "/yolo/result");
    pnh.param<bool>("show_window", show_window, true);
    pnh.param<bool>("enable_undistort", enable_undistort, true);

    // ---- 加载配置 & 类别名称 ----
    Appconfig config;
    detect_oponvino config_loader(nullptr);
    config_loader.load_config(config, config_path);
    std::vector<std::string> class_names = load_class_names(config);
    ROS_INFO("Loaded %d classes: %s", config.detect_config.classes,
             [&]
             { std::string s; for (auto& n : class_names) s += n + " "; return s; }().c_str());

    // ---- 初始化 YOLO 检测器 ----
    detect_oponvino detector(&config);
    if (!detector.inference_init())
    {
        ROS_ERROR("Failed to initialize YOLO detector");
        return -1;
    }

    // ---- 初始化海康相机 ----
    s_camera_params cam_params = {
        config.hikcamera_config.device_id,
        config.hikcamera_config.width,
        config.hikcamera_config.height,
        config.hikcamera_config.offset_x,
        config.hikcamera_config.offset_y,
        config.hikcamera_config.exposure};
    HikGrab hik(cam_params);
    hik.Hik_init();

    // ---- ROS 话题（latched：新订阅者自动获取最新结果）----
    ros::Publisher result_pub = nh.advertise<std_msgs::String>(result_topic, 1, /*latch=*/true);
    ros::Publisher grid_pub = nh.advertise<std_msgs::String>(kGridTopic, 1, /*latch=*/true);
    ros::Subscriber trigger_sub = nh.subscribe("/yolo/trigger", 1, trigger_callback);

    // ---- 键盘监听线程（Enter 触发） ----
    std::thread([]()
                {
        std::string line;
        while (ros::ok())
            if (std::getline(std::cin, line))
                g_triggered.store(true); })
        .detach();

    ROS_INFO("yolo_node ready. show_window=%s, enable_undistort=%s",
             show_window ? "true" : "false",
             enable_undistort ? "true" : "false");
    ROS_INFO("Publish '%s' to /yolo/trigger, or press Enter.", kTriggerMessage);

    ros::Rate idle_rate(kIdleLoopHz);
    GridBlock block;
    reset_grid(block);

    // ================================================================
    //  主循环：IDLE ←→ INFER
    // ================================================================
    while (ros::ok())
    {
        ros::spinOnce();

        if (!g_triggered.load())
        {
            // IDLE 期间持续驱动 OpenCV GUI 事件循环，保持 imshow 窗口可见
            if (show_window)
                cv::waitKey(1);
            idle_rate.sleep();
            continue;
        }
        g_triggered.store(false);

        // ---- INFER：连续抓帧推理 1 秒 ----
        ROS_INFO("Triggered: collecting frames for %.1f second(s)...", kInferDurationSec);
        cv::Mat last_frame;
        std::vector<Detection> all_dets = collect_detections(
            hik,
            cam_params,
            detector,
            enable_undistort,
            last_frame,
            kInferDurationSec);
        ROS_INFO_STREAM("Raw detections: " << all_dets.size());

        // ---- AGGREGATE：跨帧 NMS + 光栅排序 ----
        std::vector<Detection> final_dets = cross_frame_nms(
            all_dets, config.detect_config.nms_thresh, config.detect_config.classes);
        sort_raster(final_dets);
        ROS_INFO_STREAM("Final detections: " << final_dets.size());

        // ---- K-Means 网格定位 ----
        assign_grid_kmeans(final_dets, class_names, block);
        log_grid(block);

        // ---- PUBLISH：结果 JSON ----
        std_msgs::String result_msg;
        result_msg.data = build_result_json(final_dets, class_names);
        result_pub.publish(result_msg);

        // ---- PUBLISH：block 网格 JSON ----
        std_msgs::String grid_msg;
        grid_msg.data = build_grid_json(block);
        grid_pub.publish(grid_msg);

        // ---- 本地可视化（受 show_window 参数控制） ----
        show_viz_image(final_dets, last_frame, class_names, show_window);

        ROS_INFO("Published to %s and %s. Waiting for next trigger.", result_topic.c_str(), kGridTopic);
    }

    if (show_window)
        cv::destroyAllWindows();

    hik.Hik_end();
    return 0;
}
