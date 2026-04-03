#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

#include "nuc_detect.hpp"
#include "detector.hpp"
#include "hikvision.hpp"

// ================================================================
//  全局触发标志
// ================================================================
static std::atomic<bool> g_triggered{false};

std::string block[2][4];

void trigger_callback(const std_msgs::String::ConstPtr& msg)
{
    if (msg->data == "start_infer")
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
    int num_classes = (int)class_names.size();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << "{\"detections\":[";
    for (size_t i = 0; i < dets.size(); ++i) {
        const Detection& d = dets[i];
        int cls = (int)d.class_id;
        const std::string name = (cls >= 0 && cls < num_classes) ? class_names[cls] : std::to_string(cls);
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

    int num_classes = (int)class_names.size();
    static const cv::Scalar kColors[5] = {
        {0,255,0}, {255,0,0}, {0,0,255}, {255,255,0}, {255,0,255}
    };

    cv::Mat vis = frame.clone();
    for (size_t i = 0; i < dets.size(); ++i) {
        const Detection& d = dets[i];
        int cls = (int)d.class_id;
        const std::string name = (cls >= 0 && cls < num_classes) ? class_names[cls] : std::to_string(cls);
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

// 将 dets 按 Y 中心用 K-Means 聚成 2 行，行内按 X 排列填充 block[2][4]
// 不足 8 个目标的槽位设为 "null"
static void assign_grid_kmeans(
    const std::vector<Detection>& dets,
    const std::vector<std::string>& class_names,
    std::string block[2][4])
{
    int num_classes = (int)class_names.size();
    // 初始化
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 4; ++c)
            block[r][c] = "null";

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
    for (int row = 0; row < 2; ++row) {
        std::vector<std::pair<float, int>> items;
        for (int i = 0; i < N; ++i)
            if (row_labels[i] == row)
                items.push_back({cx[i], i});
        std::sort(items.begin(), items.end());
        for (size_t col = 0; col < items.size() && col < 4; ++col) {
            int idx = items[col].second;
            int cls = (int)dets[idx].class_id;
            block[row][col] = (cls >= 0 && cls < num_classes) ? class_names[cls] : "unknown";
        }
    }
}

// 将 block[2][4] 序列化为 JSON 字符串
static std::string build_grid_json(const std::string block[2][4])
{
    std::ostringstream oss;
    oss << "{\"block\":[";
    for (int r = 0; r < 2; ++r) {
        if (r) oss << ",";
        oss << "[";
        for (int c = 0; c < 4; ++c) {
            if (c) oss << ",";
            oss << "\"" << block[r][c] << "\"";
        }
        oss << "]";
    }
    oss << "]}";
    return oss.str();
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
    std::string config_path, result_topic , img_path;
    bool show_window;

	pnh.param<std::string>("img_path" , img_path,
        ros::package::getPath("dogvision26") + "/src/data/img/402test.jpg");

    pnh.param<std::string>("config_path",  config_path,
        ros::package::getPath("dogvision26") + "/src/settings.json");

    pnh.param<std::string>("result_topic", result_topic, "/yolo/result");
    pnh.param<bool>("show_window", show_window, true);  // 是否显示本地可视化窗口

    // ---- 加载配置 ----
    Appconfig config;
    detect_oponvino config_loader(nullptr);
    config_loader.load_config(config, config_path);

    // 从 settings 动态读取类别名称（支持 classes 字段定义的数量）
    const int num_classes = config.detect_config.classes;
    const std::string cls_pool[4] = {
        config.detect_config.class0, config.detect_config.class1,
        config.detect_config.class2, config.detect_config.class3
    };
    std::vector<std::string> class_names;
    for (int i = 0; i < std::min(num_classes, 4); ++i)
        class_names.push_back(cls_pool[i]);
    ROS_INFO("Loaded %d classes: %s", num_classes,
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
    ros::Publisher  result_pub  = nh.advertise<std_msgs::String>(result_topic, 1, /*latch=*/true);
    ros::Publisher  grid_pub    = nh.advertise<std_msgs::String>("/yolo/block_grid", 1, /*latch=*/true);
    ros::Subscriber trigger_sub = nh.subscribe("/yolo/trigger", 1, trigger_callback);

    // ---- 键盘监听线程（Enter 触发） ----
    std::thread([]() {
        std::string line;
        while (ros::ok())
            if (std::getline(std::cin, line))
                g_triggered.store(true);
    }).detach();

    ROS_INFO("yolo_node ready. show_window=%s", show_window ? "true" : "false");
    ROS_INFO("Publish 'start_infer' to /yolo/trigger, or press Enter.");

    ros::Rate idle_rate(20);

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
        ROS_INFO("Triggered: collecting frames for 1 second...");
        std::vector<Detection> all_dets;
        cv::Mat last_frame;
        ros::Time t_start = ros::Time::now();

        while (ros::ok() && (ros::Time::now() - t_start).toSec() < 1.0) {
            cv::Mat frame;
            if (hik.get_one_frame(frame, cam_params.device_id) && !frame.empty()) {
                last_frame = frame;
				last_frame = cv::imread(img_path).clone();
				//
				frame = cv::imread(img_path).clone();
                std::vector<Detection> dets;
                detector.yolo_run(frame, dets);
                all_dets.insert(all_dets.end(), dets.begin(), dets.end());
            }
        }
        ROS_INFO_STREAM("Raw detections: " << all_dets.size());

        // ---- AGGREGATE：跨帧 NMS + 光栅排序 ----
        std::vector<Detection> final_dets = cross_frame_nms(
            all_dets, config.detect_config.nms_thresh, config.detect_config.classes);
        sort_raster(final_dets);
        ROS_INFO_STREAM("Final detections: " << final_dets.size());

        // ---- K-Means 网格定位 ----
        assign_grid_kmeans(final_dets, class_names, block);
        ROS_INFO("Block grid:");
        for (int r = 0; r < 2; ++r) {
            ROS_INFO("  row%d: [%s, %s, %s, %s]",
                r, block[r][0].c_str(), block[r][1].c_str(),
                    block[r][2].c_str(), block[r][3].c_str());
        }

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

        ROS_INFO("Published to %s and /yolo/block_grid. Waiting for next trigger.", result_topic.c_str());
    }

    if (show_window)
        cv::destroyAllWindows();

    hik.Hik_end();
    return 0;
}
