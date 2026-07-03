#include <dogvision_vision/yolo_utils.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>

// ============================================================
//  reset_grid / class_name_of / load_class_names
// ============================================================
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

std::vector<std::string> load_class_names(const Appconfig& config)
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

// ============================================================
//  cross_frame_nms
// ============================================================
std::vector<Detection> cross_frame_nms(
    const std::vector<Detection>& all_dets,
    float iou_thresh,
    int /*num_classes*/)
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

    const size_t N = all_dets.size();
    if (N == 0) return {};

    // 所有候选框按置信度降序排列（跨类别）
    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return all_dets[a].conf > all_dets[b].conf;
    });

    std::vector<Detection> result;
    std::vector<bool> suppressed(N, false);
    for (size_t i = 0; i < N; ++i) {
        const size_t keep = idx[i];
        if (suppressed[keep]) continue;
        result.push_back(all_dets[keep]);
        // 跨类别抑制：与 keep 框 IoU 超过阈值的所有后续框
        for (size_t j = i + 1; j < N; ++j) {
            const size_t other = idx[j];
            if (suppressed[other]) continue;
            if (iou(all_dets[keep], all_dets[other]) > iou_thresh) {
                suppressed[other] = true;
            }
        }
    }
    return result;
}

// ============================================================
//  sort_raster
// ============================================================
void sort_raster(std::vector<Detection>& dets)
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

// ============================================================
//  assign_grid_kmeans
// ============================================================
void assign_grid_kmeans(
    const std::vector<Detection>& dets,
    const std::vector<std::string>& class_names,
    GridBlock& block)
{
    reset_grid(block);
    if (dets.empty()) return;

    int N = (int)dets.size();
    std::vector<float> cx(N), cy(N);
    for (int i = 0; i < N; ++i) {
        cx[i] = dets[i].bbox[0] + dets[i].bbox[2] * 0.5f;
        cy[i] = dets[i].bbox[1] + dets[i].bbox[3] * 0.5f;
    }

    std::vector<int> row_labels(N, 0);
    if (N > 1) {
        cv::Mat y_data(N, 1, CV_32F);
        for (int i = 0; i < N; ++i) y_data.at<float>(i, 0) = cy[i];

        cv::Mat labels, centers;
        cv::kmeans(y_data, 2, labels,
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 100, 0.01f),
            5, cv::KMEANS_PP_CENTERS, centers);

        // Y 更小的聚类为 row 0
        int top_label = (centers.at<float>(0, 0) <= centers.at<float>(1, 0)) ? 0 : 1;
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
            block[row][col] = class_name_of(static_cast<int>(dets[idx].class_id), class_names);
        }
    }
}

// ============================================================
//  build_result_json / build_grid_json / format_grid_lines
// ============================================================
std::string build_result_json(
    const std::vector<Detection>& dets,
    const std::vector<std::string>& class_names)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << "{\"detections\":[";
    for (size_t i = 0; i < dets.size(); ++i) {
        const Detection& d = dets[i];
        const std::string name = class_name_of(static_cast<int>(d.class_id), class_names);
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

std::string build_grid_json(const GridBlock& block)
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

std::vector<std::string> format_grid_lines(const GridBlock& block)
{
    std::vector<std::string> lines;
    lines.reserve(kGridRows);
    for (int r = 0; r < kGridRows; ++r)
    {
        std::ostringstream oss;
        oss << "row" << r << ": ["
            << block[r][0] << ", "
            << block[r][1] << ", "
            << block[r][2] << ", "
            << block[r][3] << "]";
        lines.push_back(oss.str());
    }
    return lines;
}

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
    const std::vector<std::string>& class_names)
{
    if (frame.empty()) return cv::Mat();

    static const cv::Scalar kColors[5] = {
        {0,255,0}, {255,0,0}, {0,0,255}, {255,255,0}, {255,0,255}
    };

    cv::Mat vis = frame.clone();
    for (size_t i = 0; i < dets.size(); ++i) {
        const Detection& d = dets[i];
        const int cls = static_cast<int>(d.class_id);
        const std::string name = class_name_of(cls, class_names);
        cv::Scalar color = kColors[((cls % 5) + 5) % 5];

        int x  = std::max(0, (int)d.bbox[0]);
        int y  = std::max(0, (int)d.bbox[1]);
        int x2 = std::min(vis.cols - 1, (int)(d.bbox[0] + d.bbox[2]));
        int y2 = std::min(vis.rows - 1, (int)(d.bbox[1] + d.bbox[3]));
        cv::rectangle(vis, {x, y}, {x2, y2}, color, 2);

        std::ostringstream label_stream;
        label_stream << "#" << (i + 1) << " " << name << " " << std::fixed
                     << std::setprecision(2) << d.conf;
        const std::string label = label_stream.str();
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        int ty = std::max(y - 4, ts.height + 4);
        const int label_right = std::min(vis.cols - 1, x + ts.width + 4);
        cv::rectangle(vis, {x, ty - ts.height - 4}, {label_right, ty}, color, cv::FILLED);
        cv::putText(vis, label, {x, ty - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255,255,255}, 1);
    }

    return vis;
}

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
    std::string* saved_path)
{
    if (frame.empty() || save_dir.empty()) return false;

    std::error_code ec;
    std::filesystem::create_directories(save_dir, ec);
    if (ec) return false;

    const auto now = std::chrono::system_clock::now();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    std::filesystem::path output_path(save_dir);
    output_path /= "yolo_" + std::to_string(millis) + ".jpg";

    const cv::Mat vis = render_yolo_result_image(dets, frame, class_names);
    if (vis.empty()) return false;

    const bool ok = cv::imwrite(output_path.string(), vis);
    if (ok && saved_path != nullptr)
    {
        *saved_path = output_path.string();
    }
    return ok;
}

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
    bool show_window)
{
    if (!show_window || frame.empty()) return;

    const cv::Mat vis = render_yolo_result_image(dets, frame, class_names);
    if (vis.empty()) return;

    cv::imshow("yolo_result", vis);
    // waitKey 由主循环统一驱动，此处不调用
}

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
    std::vector<Detection>& dets)
{
    cv::Mat frame;
    if (!camera.get_frame(frame))
    {
        processed_frame.release();
        dets.clear();
        return false;
    }

    if (enable_undistort)
    {
        frame = detector.diatorion(frame);
    }

    processed_frame = frame;
    dets.clear();
    detector.yolo_run(processed_frame, dets);
    return true;
}

// ============================================================
//  collect_detections
// ============================================================
std::vector<Detection> collect_detections(
    CameraSource& camera,
    detect_oponvino& detector,
    bool enable_undistort,
    cv::Mat& last_frame,
    double duration_sec)
{
    std::vector<Detection> all_dets;
    const auto t_start = std::chrono::steady_clock::now();

    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() < duration_sec) {
        cv::Mat frame;
        if (!camera.get_frame(frame))
            continue;

        if (enable_undistort)
            frame = detector.diatorion(frame);

        last_frame = frame;
        std::vector<Detection> frame_dets;
        detector.yolo_run(frame, frame_dets);
        all_dets.insert(all_dets.end(), frame_dets.begin(), frame_dets.end());
    }
    return all_dets;
}
