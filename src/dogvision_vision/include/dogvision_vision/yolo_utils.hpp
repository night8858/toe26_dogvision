#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <dogvision_vision/camera/camera_source.hpp>
#include <dogvision_vision/common_structs.h>
#include <dogvision_vision/nuc_detect.hpp>

constexpr int kGridRows = 2;
constexpr int kGridCols = 4;
constexpr int kMaxConfigClasses = 4;

using GridBlock = std::array<std::array<std::string, kGridCols>, kGridRows>;

/**
 * @brief 将网格中的每个单元重置为 null 标记。
 * @param block 需要原地更新的网格对象。
 * @retval void
 */
void reset_grid(GridBlock& block);

/**
 * @brief 将类别编号转换为配置中的类别名称。
 * @param cls 数值类别编号。
 * @param class_names 从配置文件加载的类别名称列表。
 * @retval std::string 类别名称；越界时返回类别编号字符串。
 */
std::string class_name_of(int cls, const std::vector<std::string>& class_names);

/**
 * @brief 从应用配置中加载类别名称。
 * @param config 已解析的应用配置。
 * @retval std::vector<std::string> 按类别编号排序的类别名称列表。
 */
std::vector<std::string> load_class_names(const Appconfig& config);

/**
 * @brief 使用按类别的 NMS 合并多帧检测结果。
 * @param all_dets 所有帧的原始检测结果。
 * @param iou_thresh 用于抑制重叠框的 IoU 阈值。
 * @param num_classes 参与分组抑制的类别数量。
 * @retval std::vector<Detection> 抑制后的最终检测结果。
 */
std::vector<Detection> cross_frame_nms(
    const std::vector<Detection>& all_dets,
    float iou_thresh,
    int num_classes);

/**
 * @brief 按从左到右、从上到下的光栅顺序排序检测框。
 * @param dets 需要原地排序的检测结果。
 * @retval void
 */
void sort_raster(std::vector<Detection>& dets);

/**
 * @brief 将检测结果分配到 2 行 4 列的类别名称网格。
 * @param dets 需要分配的检测结果。
 * @param class_names 从配置文件加载的类别名称列表。
 * @param block 输出网格对象。
 * @retval void
 */
void assign_grid_kmeans(
    const std::vector<Detection>& dets,
    const std::vector<std::string>& class_names,
    GridBlock& block);

/**
 * @brief 将检测结果序列化为兼容旧接口的 JSON 数据。
 * @param dets 需要序列化的检测结果。
 * @param class_names 从配置文件加载的类别名称列表。
 * @retval std::string 用于 /yolo/result 的 JSON 字符串。
 */
std::string build_result_json(
    const std::vector<Detection>& dets,
    const std::vector<std::string>& class_names);

/**
 * @brief 将 2x4 网格序列化为兼容旧接口的 JSON 数据。
 * @param block 需要序列化的网格对象。
 * @retval std::string 用于 /yolo/block_grid 的 JSON 字符串。
 */
std::string build_grid_json(const GridBlock& block);

/**
 * @brief 将网格格式化为便于节点日志输出的可读文本。
 * @param block 需要格式化的网格对象。
 * @retval std::vector<std::string> 每行网格对应一条字符串。
 */
std::vector<std::string> format_grid_lines(const GridBlock& block);

/**
 * @brief 生成绘制了 YOLO 检测框和类别标签的结果图。
 * @param dets 需要绘制的检测结果。
 * @param frame 作为背景的源图像。
 * @param class_names 从配置文件加载的类别名称列表。
 * @retval cv::Mat 绘制后的结果图；输入为空时返回空图像。
 */
cv::Mat render_yolo_result_image(
    const std::vector<Detection>& dets,
    const cv::Mat& frame,
    const std::vector<std::string>& class_names);

/**
 * @brief 将 YOLO 结果图保存到指定目录。
 * @param dets 需要绘制并保存的检测结果。
 * @param frame 作为背景的源图像。
 * @param class_names 从配置文件加载的类别名称列表。
 * @param save_dir 保存图片的目录。
 * @param saved_path 输出实际保存的图片路径；可为 nullptr。
 * @retval bool 保存成功时返回 true。
 */
bool save_yolo_result_image(
    const std::vector<Detection>& dets,
    const cv::Mat& frame,
    const std::vector<std::string>& class_names,
    const std::string& save_dir,
    std::string* saved_path);

/**
 * @brief 在本地 OpenCV 可视化窗口中绘制检测结果。
 * @param dets 需要绘制的检测结果。
 * @param frame 作为背景的源图像。
 * @param class_names 从配置文件加载的类别名称列表。
 * @param show_window 是否显示可视化窗口。
 * @retval void
 */
void show_viz_image(
    const std::vector<Detection>& dets,
    const cv::Mat& frame,
    const std::vector<std::string>& class_names,
    bool show_window);

/**
 * @brief 从相机获取一帧图像并执行一次 YOLO 推理。
 * @param camera settings.json 选择的相机适配对象。
 * @param detector YOLO 检测器实例。
 * @param enable_undistort 是否执行鱼眼去畸变。
 * @param processed_frame 输出推理使用的图像。
 * @param dets 输出单帧检测结果。
 * @retval bool 成功获取有效图像并完成推理流程时返回 true。
 */
bool run_single_detection(
    CameraSource& camera,
    detect_oponvino& detector,
    bool enable_undistort,
    cv::Mat& processed_frame,
    std::vector<Detection>& dets);

/**
 * @brief 在指定时长内连续取帧，并对每帧执行 YOLO 推理。
 * @param camera settings.json 选择的相机适配对象。
 * @param detector YOLO 检测器实例。
 * @param enable_undistort 是否执行鱼眼去畸变。
 * @param last_frame 输出最后一帧处理过的图像。
 * @param duration_sec 取帧持续时间，单位为秒。
 * @retval std::vector<Detection> 时间窗口内收集到的所有原始检测结果。
 */
std::vector<Detection> collect_detections(
    CameraSource& camera,
    detect_oponvino& detector,
    bool enable_undistort,
    cv::Mat& last_frame,
    double duration_sec);
