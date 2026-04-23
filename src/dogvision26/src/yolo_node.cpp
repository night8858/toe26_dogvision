#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

#include "nuc_detect.hpp"
#include "detector.hpp"
#include "hikvision.hpp"

namespace {

constexpr int kGridRows = 2;
constexpr int kGridCols = 4;
constexpr int kMaxConfigClasses = 4;
constexpr double kInferDurationSec = 1.0;
constexpr int kIdleLoopHz = 20;
constexpr char kTriggerMessage[] = "start_infer";
constexpr char kGridTopic[] = "/yolo/block_grid";

static std::atomic<bool> g_triggered{false};
using GridBlock = std::array<std::array<std::string, kGridCols>, kGridRows>;

void reset_grid(GridBlock& block)
{
    for (auto& row : block)
        for (auto& cell : row)
            cell = "null";
}

std::string class_name_of(int cls, const std::vector<std::string>& class_names)
{
    return (cls >= 0 && cls < static_cast<int>(class_names.size()))
        ? class_names[cls]
        : std::to_string(cls);
}

}  // namespace

void trigger_callback(const std_msgs::String::ConstPtr& msg)
{
    if (msg->data == kTriggerMessage)
        g_triggered.store(true);
}

// ================================================================
//  辅助函数
// ================================================================

// 跨帧 NMS：汇总多帧检测结果，按类别抑制重叠框
static std::vector<Detection> cross_frame_nms(
    const std::vector<Detection>& all_dets,
    float iou_thresh,
    int num_classes)
{
    auto iou = [](const Detection& a, const Detection& b) -> float {
        float ax2 = a.bbox[0] + a.bbox[2], ay2 = a.bbox[1] + a.bbox[3];
        float bx2 = b.bbox[0] + b.bbox[2], by2 = b.bbox[1] + b.bbox[3];
        float inter_w = std::max(0.f, std::min(ax2, bx2) - std::max(a.bbox[0], b.bbox[0]));
        float inter_h = std::max(0.f, std::min(ay2, by2) - std::max(a.bbox[1], b.bbox[1]));
        float inter   = inter_w * inter_h;
        float denom   = a.bbox[2] * a.bbox[3] + b.bbox[2] * b.bbox[3] - inter;
        return denom > 1e-6f ? inter / denom : 0.f;
    };

    std::vector<Detection> result;
    for (int cls = 0; cls < num_classes; ++cls)
    {
        std::vector<int> idx;
        for (int i = 0; i < (int)all_dets.size(); ++i)
            if ((int)all_dets[i].class_id == cls) idx.push_back(i);
        if (idx.empty()) continue;

        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return all_dets[a].conf > all_dets[b].conf;
        });

        std::vector<bool> suppressed(idx.size(), false);
        for (size_t i = 0; i < idx.size(); ++i) {
            if (suppressed[i]) continue;
            result.push_back(all_dets[idx[i]]);
            for (size_t j = i + 1; j < idx.size(); ++j)
                if (!suppressed[j] && iou(all_dets[idx[i]], all_dets[idx[j]]) > iou_thresh)
                    suppressed[j] = true;
        }
    }
    return result;
}

// 光栅扫描排序（原地）：从左上到右下，y 相近视为同行
static void sort_raster(std::vector<Detection>& dets)
{
    if (dets.empty()) return;

    float avg_h = 0.f;
    for (const auto& d : dets) avg_h += d.bbox[3];
    float row_tol = (avg_h / dets.size()) * 0.5f;

    std::sort(dets.begin(), dets.end(), [row_tol](const Detection& a, const Detection& b) {
        float cya = a.bbox[1] + a.bbox[3] * .5f;
        float cyb = b.bbox[1] + b.bbox[3] * .5f;
        if (std::fabs(cya - cyb) > row_tol) return cya < cyb;
        return (a.bbox[0] + a.bbox[2] * .5f) < (b.bbox[0] + b.bbox[2] * .5f);
    });
}

// 构建 JSON 结果字符串（pos_id 从 1 开始）
static std::string build_result_json(
    const std::vector<Detection>& dets,
    const std::vector<std::string>& class_names)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << "{\"detections\":[";
    for (size_t i = 0; i < dets.size(); ++i) {
        const Detection& d = dets[i];
        const int cls = static_cast<int>(d.class_id);
        const std::string name = class_name_of(cls, class_names);
        if (i) oss << ",";
        oss << "{\"pos_id\":"  << (i + 1)
            << ",\"class\":\"" << name << "\""
            << ",\"conf\":"    << d.conf
            << ",\"bbox\":["   << d.bbox[0] << "," << d.bbox[1]
            << ","             << d.bbox[2] << "," << d.bbox[3] << "]}";
    }
    oss << "]}";
    return oss.str();
}

// 在图像上绘制检测框 + 编号标签，本地窗口显示
// show_window 为 false 时跳过，不创建任何窗口
static void show_viz_image(
    const std::vector<Detection>& dets,
    const cv::Mat& frame,
    const std::vector<std::string>& class_names,
    bool show_window)
{
    if (!show_window || frame.empty()) return;

    static const cv::Scalar kColors[5] = 
    {
        {0,255,0}, {255,0,0}, {0,0,255}, {255,255,0}, {255,0,255}
    };

    cv::Mat vis = frame.clone();
    for (size_t i = 0; i < dets.size(); ++i) 
    {
        const Detection& d = dets[i];
        const int cls = static_cast<int>(d.class_id);
        const std::string name = class_name_of(cls, class_names);
        cv::Scalar color = kColors[cls % 5];

        int x  = std::max(0, (int)d.bbox[0]);
        int y  = std::max(0, (int)d.bbox[1]);
        int x2 = std::min(vis.cols - 1, (int)(d.bbox[0] + d.bbox[2]));
        int y2 = std::min(vis.rows - 1, (int)(d.bbox[1] + d.bbox[3]));

        cv::rectangle(vis, {x, y}, {x2, y2}, color, 2);

        std::string label = "#" + std::to_string(i + 1) + " " + name;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        int ty = std::max(y - 4, ts.height + 4);
        cv::rectangle(vis, {x, ty - ts.height - 4}, {x + ts.width, ty}, color, cv::FILLED);
        cv::putText(vis, label, {x, ty - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255,255,255}, 1);

    }

    cv::imshow("yolo_result", vis);
    // 不在此处调用 waitKey，由主循环统一驱动 GUI 事件
}

// ================================================================
//  K-Means 鲁棒定位：将检测结果分配到 2 行 × 4 列网格
// ================================================================

// 将 dets 按 Y 中心用 K-Means 聚成 2 行，行内按 X 排列填充 
static void assign_grid_kmeans(
    const std::vector<Detection>& dets,
    const std::vector<std::string>& class_names,
    GridBlock& block)
{
    reset_grid(block);

    if (dets.empty()) return;

    // 提取各目标中心
    int N = (int)dets.size();
    std::vector<float> cx(N), cy(N);
    for (int i = 0; i < N; ++i) {
        cx[i] = dets[i].bbox[0] + dets[i].bbox[2] * 0.5f;
        cy[i] = dets[i].bbox[1] + dets[i].bbox[3] * 0.5f;
    }

    std::vector<int> row_labels(N, 0);

    if (N == 1) {
        row_labels[0] = 0;
    } else {
        // 对 Y 坐标做 1D K-Means（K=2）
        cv::Mat y_data(N, 1, CV_32F);
        for (int i = 0; i < N; ++i) y_data.at<float>(i, 0) = cy[i];

        cv::Mat labels, centers;
        int attempts = 5;
        cv::kmeans(y_data, 2, labels,
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 100, 0.01f),
            attempts, cv::KMEANS_PP_CENTERS, centers);

        // label 对应 row：Y 更小的聚类为 row 0
        float c0y = centers.at<float>(0, 0);
        float c1y = centers.at<float>(1, 0);
        int top_label = (c0y <= c1y) ? 0 : 1;  // 哪个 label 是上行

        for (int i = 0; i < N; ++i)
            row_labels[i] = (labels.at<int>(i, 0) == top_label) ? 0 : 1;
    }

    // 按行分组，行内按 X 排序后填入列
    for (int row = 0; row < kGridRows; ++row) {
        std::vector<std::pair<float, int>> items;
        for (int i = 0; i < N; ++i)
            if (row_labels[i] == row)
                items.push_back({cx[i], i});
        std::sort(items.begin(), items.end());
        for (size_t col = 0; col < items.size() && col < static_cast<size_t>(kGridCols); ++col) {
            int idx = items[col].second;
            const int cls = static_cast<int>(dets[idx].class_id);
            block[row][col] = class_name_of(cls, class_names);
        }
    }
}

// 将 block[2][4] 序列化为 JSON 字符串
static std::string build_grid_json(const GridBlock& block)
{
    std::ostringstream oss;
    oss << "{\"block\":[";
    for (int r = 0; r < kGridRows; ++r) {
        if (r) oss << ",";
        oss << "[";
        for (int c = 0; c < kGridCols; ++c) {
            if (c) oss << ",";
            oss << "\"" << block[r][c] << "\"";
        }
        oss << "]";
    }
    oss << "]}";
    return oss.str();
}

static std::vector<std::string> load_class_names(const Appconfig& config)
{
    const int num_classes = config.detect_config.classes;
    const std::array<std::string, kMaxConfigClasses> cls_pool = {
        config.detect_config.class0,
        config.detect_config.class1,
        config.detect_config.class2,
        config.detect_config.class3,
    };

    std::vector<std::string> class_names;
    for (int i = 0; i < std::min(num_classes, kMaxConfigClasses); ++i)
        class_names.push_back(cls_pool[i]);
    return class_names;
}

// 连续抓帧推理指定秒数，返回所有检测结果
static std::vector<Detection> collect_detections(
    HikGrab& hik,
    const s_camera_params& cam_params,
    detect_oponvino& detector,
    bool enable_undistort,
    cv::Mat& last_frame,
    double duration_sec)
{
    std::vector<Detection> all_dets;
    const ros::Time t_start = ros::Time::now();

    while (ros::ok() && (ros::Time::now() - t_start).toSec() < duration_sec) {
        cv::Mat frame;
        if (!hik.get_one_frame(frame, cam_params.device_id) || frame.empty()) {
            continue;
        }

        if (enable_undistort) {
            frame = detector.diatorion(frame);
        }

        last_frame = frame;
        std::vector<Detection> frame_dets;
        detector.yolo_run(frame, frame_dets);
        all_dets.insert(all_dets.end(), frame_dets.begin(), frame_dets.end());
    }
    return all_dets;
}

// 打印 block 网格内容到 ROS 日志（也可选调用 build_grid_json 发布到 ROS 话题）
// 注意：block 内容仅为类别名称字符串，不包含坐标等信息
static void log_grid(const GridBlock& block)
{
    ROS_INFO("Block grid:");
    for (int r = 0; r < kGridRows; ++r) {
        ROS_INFO("  row%d: [%s, %s, %s, %s]",
            r,
            block[r][0].c_str(),
            block[r][1].c_str(),
            block[r][2].c_str(),
            block[r][3].c_str());
    }
}

// ================================================================
//  main
// ================================================================
int main(int argc, char** argv)
{
    ros::init(argc, argv, "yolo_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // ---- 参数 ----
    std::string config_path;
    std::string result_topic;
    bool show_window = true;
    bool enable_undistort = true;

    pnh.param<std::string>("config_path",  config_path,
        ros::package::getPath("dogvision26") + "/src/settings.json");

    pnh.param<std::string>("result_topic", result_topic, "/yolo/result");
    pnh.param<bool>("show_window", show_window, true);
    pnh.param<bool>("enable_undistort", enable_undistort, true);

    // ---- 加载配置 ----
    Appconfig config;
    detect_oponvino config_loader(nullptr);
    config_loader.load_config(config, config_path);

    std::vector<std::string> class_names = load_class_names(config);
    ROS_INFO("Loaded %d classes: %s", config.detect_config.classes,
        [&]{ std::string s; for (auto& n:class_names) s+=n+" "; return s; }().c_str());

    // ---- 初始化 YOLO 检测器 ----
    detect_oponvino detector(&config);
    if (!detector.inference_init()) {
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
        config.hikcamera_config.exposure
    };
    HikGrab hik(cam_params);
    hik.Hik_init();

    // ---- ROS 话题 ----
    // latched：新订阅者自动获取上次结果，无需主动重发
    // 注意：图像仅用于本地窗口显示，不通过 ROS 话题传递。
    ros::Publisher  result_pub  = nh.advertise<std_msgs::String>(result_topic, 1, /*latch=*/true);
    ros::Publisher  grid_pub    = nh.advertise<std_msgs::String>(kGridTopic, 1, /*latch=*/true);
    ros::Subscriber trigger_sub = nh.subscribe("/yolo/trigger", 1, trigger_callback);

    // ---- 键盘监听线程（Enter 触发） ----
    std::thread([]() {
        std::string line;
        while (ros::ok())
            if (std::getline(std::cin, line))
                g_triggered.store(true);
    }).detach();

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

        if (!g_triggered.load()) {
            // IDLE 期间持续驱动 OpenCV GUI 事件循环，保持 imshow 窗口可见
            if (show_window) cv::waitKey(1);
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
