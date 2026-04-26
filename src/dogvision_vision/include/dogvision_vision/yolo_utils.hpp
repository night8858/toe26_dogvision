#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <ros/ros.h>

#include <dogvision_vision/common_structs.h>
#include <dogvision_vision/nuc_detect.hpp>
#include <dogvision_camera/hikvision.hpp>

// ============================================================
//  yolo_utils.hpp
//  yolo_node 中的辅助工具函数
//  职责：跨帧NMS、排序、网格分配、JSON序列化、抓帧推理、可视化
// ============================================================

/// 网格维度
constexpr int kGridRows         = 2;
constexpr int kGridCols         = 4;
constexpr int kMaxConfigClasses = 4;

/// 2×4 类别名称网格类型
using GridBlock = std::array<std::array<std::string, kGridCols>, kGridRows>;

// ----------------------------------------------------------------

/// 将 block 所有格子置为 "null"
void reset_grid(GridBlock& block);

/// 根据类别 id 返回类别名称（越界时返回数字字符串）
std::string class_name_of(int cls, const std::vector<std::string>& class_names);

/// 从 Appconfig 中读取配置的类别名称列表
std::vector<std::string> load_class_names(const Appconfig& config);

/// 跨帧 NMS：合并多帧检测结果，按类别抑制重叠框
std::vector<Detection> cross_frame_nms(
    const std::vector<Detection>& all_dets,
    float iou_thresh,
    int num_classes);

/// 光栅扫描排序（原地）：左→右、上→下，y 相近视为同行
void sort_raster(std::vector<Detection>& dets);

/// K-Means 将检测结果按 Y 中心聚为 2 行，行内按 X 填入 kGridCols 列
void assign_grid_kmeans(
    const std::vector<Detection>& dets,
    const std::vector<std::string>& class_names,
    GridBlock& block);

/// 序列化检测结果为 JSON 字符串（pos_id 从 1 开始）
std::string build_result_json(
    const std::vector<Detection>& dets,
    const std::vector<std::string>& class_names);

/// 序列化 block 网格为 JSON 字符串
std::string build_grid_json(const GridBlock& block);

/// 将 block 网格内容逐行输出到 ROS INFO 日志
void log_grid(const GridBlock& block);

/// 在本地 OpenCV 窗口中绘制检测框和编号标签（show_window=false 时跳过）
void show_viz_image(
    const std::vector<Detection>& dets,
    const cv::Mat& frame,
    const std::vector<std::string>& class_names,
    bool show_window);

/// 连续抓帧并推理 duration_sec 秒，返回所有原始检测结果
/// @param last_frame 输出：最后一帧图像（供可视化使用）
std::vector<Detection> collect_detections(
    HikGrab& hik,
    const s_camera_params& cam_params,
    detect_oponvino& detector,
    bool enable_undistort,
    cv::Mat& last_frame,
    double duration_sec);
