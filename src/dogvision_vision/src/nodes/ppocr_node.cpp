/**
 * @file ppocr_node.cpp
 * @brief ROS2 PP-OCR 算术题识别节点
 *
 * 支持两种运行模式：
 *   - test（测试模式）：连续 OCR + 多帧投票，稳定结果追加写入 YAML
 *   - production（生产模式）：通过 /ocr/trigger 话题触发，发布 /ocr/result
 *
 * 完整流水线：
 *   1. 从海康相机获取帧
 *   2. 鱼眼去畸变（可选）
 *   3. 按配置准备彩色或三通道灰度整帧 OCR 输入
 *   4. PPOCR 文本检测（detect_det_ppocr）
 *   5. PPOCR 文本识别（detect_rec_ppocr，含数学字符白名单）
 *   6. 单行算术候选组合与严格表达式校验
 *   7. 文字外围白色环带筛选
 *   8. 多帧滑动窗口投票（OCRMultiFrameVoter）提高稳定性
 */

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8.hpp>

#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <dogvision_vision/camera/hikvision.hpp>
#include <dogvision_vision/common_structs.h>
#include <dogvision_vision/ocr_MultiFrameVoter.hpp>
#include <dogvision_vision/ocr_detect.hpp>
#include <dogvision_vision/ocr_utils.hpp>

namespace fs = std::filesystem;

namespace
{
constexpr char kOcrWindowName[] = "Math OCR";
constexpr char kOcrRoiWindowName[] = "Math OCR Candidate";
constexpr char kOcrDebugWindowName[] = "Math OCR Debug";

std::atomic<bool> g_ocr_triggered{false};
std::atomic<bool> g_ocr_running{true};

struct OCRDebugFrame
{
    cv::Mat original_frame;
    cv::Mat annotated_frame;
    cv::Mat ocr_input;
    cv::Mat white_mask;
    cv::Rect2f white_roi;
    cv::Mat white_roi_crop;
    cv::Mat best_candidate_canvas;
    std::vector<OCRItem> ocr_items;
    std::vector<OCRMathCandidate> candidates;
    bool success = false;
    std::string expr;
    int result = 0;
    int mod4 = 0;
    double elapsed_ms = 0.0;
    std::string source_name;
    std::size_t voter_frames = 0;
    std::size_t voter_valid = 0;
    bool voter_stable = false;
    std::string stable_expr;
};

struct VisualizationKey
{
    bool should_exit = false;
    bool save_snapshot = false;
    bool pause_toggle = false;
    bool next_frame = false;
};

cv::Mat make_no_roi_canvas()
{
    cv::Mat canvas(240, 640, CV_8UC3, cv::Scalar(24, 24, 24));
    const std::string message = "No math candidate detected";
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(
        message, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
    const cv::Point origin(
        std::max(10, (canvas.cols - text_size.width) / 2),
        std::max(text_size.height + 10, (canvas.rows + text_size.height) / 2));
    cv::putText(canvas, message, origin, cv::FONT_HERSHEY_SIMPLEX,
                0.8, cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
    return canvas;
}

cv::Mat ensure_bgr(const cv::Mat& input)
{
    if (input.empty())
        return {};
    if (input.channels() == 3)
        return input.clone();
    cv::Mat bgr;
    if (input.channels() == 1)
        cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
    else if (input.channels() == 4)
        cv::cvtColor(input, bgr, cv::COLOR_BGRA2BGR);
    return bgr;
}

cv::Mat fit_canvas(const cv::Mat& input, const cv::Size& size)
{
    cv::Mat canvas(size, CV_8UC3, cv::Scalar(24, 24, 24));
    if (input.empty())
        return canvas;

    cv::Mat bgr = ensure_bgr(input);
    const double scale = std::min(
        static_cast<double>(size.width) / static_cast<double>(bgr.cols),
        static_cast<double>(size.height) / static_cast<double>(bgr.rows));
    const cv::Size resized_size(
        std::max(1, static_cast<int>(std::round(bgr.cols * scale))),
        std::max(1, static_cast<int>(std::round(bgr.rows * scale))));
    cv::Mat resized;
    cv::resize(bgr, resized, resized_size);
    const cv::Rect dst(
        (size.width - resized.cols) / 2,
        (size.height - resized.rows) / 2,
        resized.cols,
        resized.rows);
    resized.copyTo(canvas(dst));
    return canvas;
}

void put_label(cv::Mat& image,
               const std::string& text,
               const cv::Point& origin,
               double scale = 0.55,
               const cv::Scalar& color = cv::Scalar(255, 255, 255))
{
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv::LINE_AA);
}

cv::Mat make_debug_panel(const OCRDebugFrame& debug)
{
    const cv::Size tile_size(480, 270);
    cv::Mat panel(tile_size.height * 2, tile_size.width * 2, CV_8UC3,
                  cv::Scalar(18, 18, 18));

    cv::Mat original = fit_canvas(debug.original_frame, tile_size);
    cv::Mat input = fit_canvas(debug.ocr_input, tile_size);
    cv::Mat mask = fit_canvas(debug.white_mask, tile_size);
    cv::Mat roi = fit_canvas(debug.white_roi_crop.empty()
                                 ? debug.best_candidate_canvas
                                 : debug.white_roi_crop,
                             tile_size);

    original.copyTo(panel(cv::Rect(0, 0, tile_size.width, tile_size.height)));
    input.copyTo(panel(cv::Rect(tile_size.width, 0, tile_size.width, tile_size.height)));
    mask.copyTo(panel(cv::Rect(0, tile_size.height, tile_size.width, tile_size.height)));
    roi.copyTo(panel(cv::Rect(tile_size.width, tile_size.height, tile_size.width, tile_size.height)));

    put_label(panel, "Original", cv::Point(12, 24));
    put_label(panel, "OCR input", cv::Point(tile_size.width + 12, 24));
    put_label(panel, "White mask", cv::Point(12, tile_size.height + 24));
    put_label(panel, "ROI / best candidate", cv::Point(tile_size.width + 12, tile_size.height + 24));

    std::vector<std::string> lines;
    std::ostringstream oss;
    oss << "src=" << (debug.source_name.empty() ? "camera" : debug.source_name);
    lines.push_back(oss.str());
    oss.str("");
    oss << "white_roi=" << static_cast<int>(debug.white_roi.x) << ","
        << static_cast<int>(debug.white_roi.y) << " "
        << static_cast<int>(debug.white_roi.width) << "x"
        << static_cast<int>(debug.white_roi.height);
    lines.push_back(oss.str());
    oss.str("");
    oss << "ocr_items=" << debug.ocr_items.size()
        << " candidates=" << debug.candidates.size()
        << " elapsed=" << std::fixed << std::setprecision(1)
        << debug.elapsed_ms << "ms";
    lines.push_back(oss.str());
    oss.str("");
    oss << "voter=" << debug.voter_valid << "/" << debug.voter_frames
        << " stable=" << (debug.voter_stable ? debug.stable_expr : "none");
    lines.push_back(oss.str());
    if (debug.success)
    {
        oss.str("");
        oss << "result=" << debug.expr << "=" << debug.result
            << " mod4=" << debug.mod4;
        lines.push_back(oss.str());
    }
    const std::size_t max_candidates = std::min<std::size_t>(debug.candidates.size(), 4);
    for (std::size_t i = 0; i < max_candidates; ++i)
    {
        const OCRMathCandidate& candidate = debug.candidates[i];
        oss.str("");
        oss << "#" << i << " " << candidate.expression
            << " score=" << std::fixed << std::setprecision(2)
            << candidate.mean_score
            << " white=" << candidate.surround_white_ratio
            << (candidate.white_pass ? " PASS" : " REJECT");
        lines.push_back(oss.str());
    }

    int y = 50;
    for (const std::string& line : lines)
    {
        put_label(panel, line, cv::Point(12, y), 0.52,
                  cv::Scalar(90, 240, 255));
        y += 22;
    }
    put_label(panel, "keys: q/ESC exit, s save, space pause, n next",
              cv::Point(12, panel.rows - 14), 0.52,
              cv::Scalar(200, 200, 200));
    return panel;
}

VisualizationKey decode_key(int key)
{
    VisualizationKey decoded;
    if (key == 'q' || key == 'Q' || key == 27)
        decoded.should_exit = true;
    else if (key == 's' || key == 'S')
        decoded.save_snapshot = true;
    else if (key == ' ')
        decoded.pause_toggle = true;
    else if (key == 'n' || key == 'N')
        decoded.next_frame = true;
    return decoded;
}

VisualizationKey handle_visualization(const cv::Mat& frame,
                                      const cv::Mat& ocr_roi,
                                      const cv::Mat& debug_panel,
                                      bool show_visual,
                                      bool show_ocr_roi,
                                      bool show_debug_panels,
                                      int wait_ms)
{
    if (show_visual)
    {
        cv::imshow(kOcrWindowName, frame);
    }
    if (show_ocr_roi)
    {
        cv::imshow(
            kOcrRoiWindowName,
            ocr_roi.empty() ? make_no_roi_canvas() : ocr_roi);
    }
    if (show_debug_panels)
    {
        cv::imshow(
            kOcrDebugWindowName,
            debug_panel.empty() ? make_no_roi_canvas() : debug_panel);
    }
    if (!show_visual && !show_ocr_roi && !show_debug_panels)
    {
        return {};
    }

    const int key = cv::waitKey(wait_ms);
    return decode_key(key);
}

bool save_debug_snapshot(const OCRDebugFrame& debug,
                         const cv::Mat& display_frame,
                         const cv::Mat& ocr_roi,
                         const cv::Mat& debug_panel,
                         const std::string& snapshot_dir,
                         const rclcpp::Logger& logger)
{
    if (snapshot_dir.empty())
        return false;

    std::error_code ec;
    fs::create_directories(snapshot_dir, ec);
    if (ec)
    {
        RCLCPP_WARN(logger, "Cannot create OCR debug dir: %s", ec.message().c_str());
        return false;
    }

    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    const fs::path base = fs::path(snapshot_dir) / ("ocr_" + std::to_string(ms));

    bool ok = true;
    if (!debug.original_frame.empty())
        ok &= cv::imwrite(base.string() + "_original.png", debug.original_frame);
    if (!display_frame.empty())
        ok &= cv::imwrite(base.string() + "_annotated.png", display_frame);
    if (!debug.ocr_input.empty())
        ok &= cv::imwrite(base.string() + "_ocr_input.png", debug.ocr_input);
    if (!debug.white_mask.empty())
        ok &= cv::imwrite(base.string() + "_white_mask.png", debug.white_mask);
    if (!debug.white_roi_crop.empty())
        ok &= cv::imwrite(base.string() + "_white_roi.png", debug.white_roi_crop);
    if (!ocr_roi.empty())
        ok &= cv::imwrite(base.string() + "_candidate.png", ocr_roi);
    if (!debug_panel.empty())
        ok &= cv::imwrite(base.string() + "_debug.png", debug_panel);

    RCLCPP_INFO(logger, "%s OCR debug snapshot: %s_*",
                ok ? "Saved" : "Partially saved",
                base.string().c_str());
    return ok;
}

void draw_ocr_item(cv::Mat& frame, const OCRItem& item)
{
    std::vector<cv::Point> polygon;
    for (const cv::Point2f& point : item.box.pts)
    {
        polygon.emplace_back(
            static_cast<int>(std::round(point.x)),
            static_cast<int>(std::round(point.y)));
    }
    cv::polylines(
        frame, polygon, true, cv::Scalar(255, 180, 0), 2, cv::LINE_AA);

    std::ostringstream label;
    label << item.rec.text << " " << std::fixed
          << std::setprecision(2) << item.rec.score;
    const cv::Point origin(
        polygon[0].x,
        std::max(15, polygon[0].y - 5));
    cv::putText(
        frame, label.str(), origin, cv::FONT_HERSHEY_SIMPLEX,
        0.5, cv::Scalar(255, 180, 0), 1, cv::LINE_AA);
}

void draw_math_candidate(
    cv::Mat& frame,
    const OCRMathCandidate& candidate,
    bool selected)
{
    const cv::Scalar color =
        candidate.white_pass ? cv::Scalar(0, 220, 0)
                             : cv::Scalar(0, 0, 255);
    cv::rectangle(
        frame, candidate.surround_bounds,
        selected ? color : cv::Scalar(0, 165, 255),
        selected ? 3 : 1);
    cv::rectangle(frame, candidate.text_bounds, color, selected ? 3 : 2);

    std::ostringstream label;
    label << candidate.expression
          << " white=" << std::fixed << std::setprecision(2)
          << candidate.surround_white_ratio
          << (candidate.white_pass ? " PASS" : " REJECT");
    const cv::Point origin(
        candidate.surround_bounds.x,
        std::max(18, candidate.surround_bounds.y - 6));
    cv::putText(
        frame, label.str(), origin, cv::FONT_HERSHEY_SIMPLEX,
        selected ? 0.65 : 0.5, color, selected ? 2 : 1, cv::LINE_AA);
}

cv::Mat make_candidate_canvas(
    const cv::Mat& original,
    const OCRMathCandidate& candidate)
{
    if (original.empty() || candidate.surround_bounds.empty())
        return {};

    cv::Mat crop = original(candidate.surround_bounds).clone();
    const int banner_height = 42;
    cv::Mat canvas(
        crop.rows + banner_height, crop.cols, CV_8UC3,
        cv::Scalar(24, 24, 24));
    crop.copyTo(canvas(cv::Rect(0, banner_height, crop.cols, crop.rows)));

    std::ostringstream label;
    label << candidate.expression
          << "  white=" << std::fixed << std::setprecision(2)
          << candidate.surround_white_ratio
          << (candidate.white_pass ? " PASS" : " REJECT");
    cv::putText(
        canvas, label.str(), cv::Point(8, 28),
        cv::FONT_HERSHEY_SIMPLEX, 0.62,
        candidate.white_pass ? cv::Scalar(0, 220, 0)
                             : cv::Scalar(0, 0, 255),
        2, cv::LINE_AA);
    return canvas;
}
} // namespace

/**
 * @brief 在取帧失败时重连海康相机。
 * @param hik 需要重连的相机封装对象。
 * @param logger 用于输出过程信息的日志对象。
 * @param max_retries 最大重连次数。
 * @retval bool 重连后能够成功获取有效帧时返回 true。
 */
static bool ensure_camera(HikGrab& hik, const rclcpp::Logger& logger, int max_retries = 5)
{
    for (int i = 0; i < max_retries; ++i)
    {
        cv::Mat test;
        if (hik.get_one_frame(test, 0) && !test.empty())
        {
            return true;
        }

        RCLCPP_WARN(logger, "Camera lost (attempt %d/%d), reconnecting...", i + 1, max_retries);
        hik.Hik_end();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        hik.Hik_init();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    RCLCPP_ERROR(logger, "Camera reconnection failed after %d attempts.", max_retries);
    return false;
}

/**
 * @brief 处理 OCR 触发消息。
 * @param msg 收到的触发消息。
 * @retval void
 */
static void ocr_trigger_callback(const std_msgs::msg::String::SharedPtr msg)
{
    (void)msg;
    g_ocr_triggered.store(true);
}

/**
 * @brief 执行一次完整的 OCR 与算术表达式解析流程。
 * @param img 输入 BGR 图像。
 * @param det 文本检测模型封装对象。
 * @param rec 文本识别模型封装对象。
 * @param logger 用于输出流程信息的日志对象。
 * @param expr_str 输出识别到的表达式字符串。
 * @param int_result 输出四舍五入后的整数结果。
 * @param mod_result 输出非负的模 4 结果。
 * @param out_roi 可选输出最终算术题文字区域。
 * @param out_ocr_roi 可选输出当前最优算术候选可视化图。
 * @retval bool 成功识别并解析表达式时返回 true。
 */
static bool run_ocr_pipeline(cv::Mat& img,
                             detect_det_ppocr& det,
                             detect_rec_ppocr& rec,
                             const s_detector_params& ocr_config,
                             const rclcpp::Logger& logger,
                             std::string& expr_str,
                             int& int_result,
                             int& mod_result,
                             cv::Rect2f* out_roi = nullptr,
                             cv::Mat* out_ocr_roi = nullptr,
                             OCRDebugFrame* debug = nullptr)
{
    const auto start = std::chrono::steady_clock::now();
    const cv::Mat original_frame = img.clone();
    auto finish = [&](bool success)
    {
        if (debug != nullptr)
        {
            debug->annotated_frame = img.clone();
            debug->success = success;
            debug->expr = expr_str;
            debug->result = int_result;
            debug->mod4 = mod_result;
            const auto end = std::chrono::steady_clock::now();
            debug->elapsed_ms =
                std::chrono::duration<double, std::milli>(end - start).count();
        }
        return success;
    };

    if (out_ocr_roi != nullptr)
    {
        out_ocr_roi->release();
    }
    if (debug != nullptr)
    {
        *debug = OCRDebugFrame{};
        debug->original_frame = original_frame.clone();
        debug->white_roi = find_math_proble(
            original_frame,
            &debug->white_mask,
            ocr_config.ocr_math_white_s_max,
            ocr_config.ocr_math_white_v_min);
        if (debug->white_roi.area() > 0.0f)
        {
            cv::Rect roi = cv::Rect(debug->white_roi) &
                           cv::Rect(0, 0, original_frame.cols, original_frame.rows);
            if (!roi.empty())
                debug->white_roi_crop = original_frame(roi).clone();
        }
    }

    cv::Mat det_input = prepare_ocr_input(
        original_frame, ocr_config.ocr_math_use_grayscale);
    if (debug != nullptr)
    {
        debug->ocr_input = det_input.clone();
    }
    if (det_input.empty())
    {
        if (out_roi != nullptr)
            *out_roi = cv::Rect2f();
        RCLCPP_WARN(logger, "Empty full-frame OCR input, skip inference.");
        return finish(false);
    }

    RCLCPP_INFO(
        logger, "Full-frame OCR: %dx%d, grayscale=%s",
        det_input.cols, det_input.rows,
        ocr_config.ocr_math_use_grayscale ? "true" : "false");
    det.preprocess(det_input);
    det.inference();
    det.postprocess();
    const std::vector<OCRBox>& all_boxes = det.ocr_det_out_;
    RCLCPP_INFO(logger, "Detected   : %zu text region(s)", all_boxes.size());

    std::vector<OCRItem> ocr_items;
    for (size_t i = 0; i < all_boxes.size(); ++i)
    {
        cv::Mat crop = crop_text_region(det_input, all_boxes[i]);
        if (crop.empty())
        {
            continue;
        }

        rec.preprocess(crop);
        rec.inference();
        rec.postprocess();

        if (!rec.result.empty() && !rec.result[0].text.empty())
        {
            OCRItem item;
            item.box = all_boxes[i];
            item.rec = rec.result[0];
            ocr_items.push_back(item);
            RCLCPP_INFO(logger, "  [%zu] \"%s\"  score=%.3f",
                        i, rec.result[0].text.c_str(), rec.result[0].score);
        }
    }
    if (debug != nullptr)
    {
        debug->ocr_items = ocr_items;
    }

    for (const auto& item : ocr_items)
        draw_ocr_item(img, item);

    std::vector<OCRMathCandidate> candidates =
        find_math_candidates(original_frame, ocr_items, ocr_config);
    if (debug != nullptr)
    {
        debug->candidates = candidates;
    }
    RCLCPP_INFO(logger, "Math candidates: %zu", candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        const OCRMathCandidate& candidate = candidates[i];
        draw_math_candidate(img, candidate, i == 0);
        RCLCPP_INFO(
            logger,
            "  candidate[%zu] \"%s\" score=%.3f white=%.3f %s",
            i, candidate.expression.c_str(), candidate.mean_score,
            candidate.surround_white_ratio,
            candidate.white_pass ? "PASS" : "REJECT");
    }

    if (candidates.empty())
    {
        if (out_roi != nullptr)
            *out_roi = cv::Rect2f();
        RCLCPP_WARN(logger, "No valid arithmetic candidate found.");
        return finish(false);
    }

    const OCRMathCandidate& best = candidates.front();
    cv::Mat candidate_canvas = make_candidate_canvas(original_frame, best);
    if (out_ocr_roi != nullptr)
        *out_ocr_roi = candidate_canvas;
    if (debug != nullptr)
        debug->best_candidate_canvas = candidate_canvas.clone();
    if (!best.white_pass)
    {
        if (out_roi != nullptr)
            *out_roi = cv::Rect2f();
        RCLCPP_WARN(
            logger,
            "Best arithmetic candidate rejected by white surround: %.3f < %.3f",
            best.surround_white_ratio,
            ocr_config.ocr_math_min_surround_white_ratio);
        return finish(false);
    }

    expr_str = best.expression;
    if (best.result < static_cast<double>(std::numeric_limits<int>::min()) ||
        best.result > static_cast<double>(std::numeric_limits<int>::max()))
    {
        if (out_roi != nullptr)
            *out_roi = cv::Rect2f();
        RCLCPP_WARN(logger, "Arithmetic result is outside int range.");
        return finish(false);
    }
    int_result = static_cast<int>(std::round(best.result));
    mod_result = ((int_result % 4) + 4) % 4;
    if (out_roi != nullptr)
        *out_roi = cv::Rect2f(best.text_bounds);
    return finish(true);
}

/**
 * @brief 在算术题区域附近绘制识别结果。
 * @param frame 需要绘制标注的图像。
 * @param math_roi 已定位的算术题区域。
 * @param expr_str 识别到的表达式。
 * @param int_result 四舍五入后的整数结果。
 * @param mod_result 非负的模 4 结果。
 * @retval void
 */
static void draw_result_overlay(cv::Mat& frame,
                                const cv::Rect2f& math_roi,
                                const std::string& expr_str,
                                int int_result,
                                int mod_result)
{
    std::ostringstream disp;
    disp << expr_str << " = " << int_result << "  (mod4: " << mod_result << ")";
    const std::string display_text = disp.str();

    int text_x = static_cast<int>(math_roi.x);
    int text_y = static_cast<int>(math_roi.y + math_roi.height) + 40;
    text_y = std::min(text_y, frame.rows - 10);

    const double font_scale = 0.8;
    const int thickness = 2;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(display_text, cv::FONT_HERSHEY_DUPLEX, font_scale, thickness, &baseline);
    if (text_x + ts.width > frame.cols)
    {
        text_x = frame.cols - ts.width - 10;
    }

    cv::Rect bg(text_x - 4, text_y - ts.height - 4, ts.width + 8, ts.height + 8);
    bg &= cv::Rect(0, 0, frame.cols, frame.rows);
    cv::rectangle(frame, bg, cv::Scalar(0, 0, 0), -1);
    cv::rectangle(frame, bg, cv::Scalar(0, 200, 0), 2);
    cv::putText(frame, display_text, cv::Point(text_x, text_y - 4),
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

static void fill_voter_debug(OCRDebugFrame& debug, const OCRMultiFrameVoter& voter)
{
    debug.voter_frames = voter.frame_count();
    debug.voter_valid = voter.valid_result_count();
    debug.voter_stable = voter.has_stable_result();
    debug.stable_expr = voter.has_stable_result() ? voter.stable_result().expr : "";
}

/**
 * @brief 运行连续 OCR 测试模式，并将稳定结果变化追加写入 YAML。
 * @param node ROS2 节点对象。
 * @param hik 相机封装对象。
 * @param det 文本检测模型封装对象。
 * @param rec 文本识别模型封装对象。
 * @param yaml_path YAML 输出文件路径。
 * @param show_visual 是否启用 OpenCV 可视化窗口。
 * @param show_ocr_roi 是否显示当前最优算术候选。
 * @retval int 类进程退出码。
 */
static int run_test_mode(const rclcpp::Node::SharedPtr& node,
                         HikGrab& hik,
                         detect_det_ppocr& det,
                         detect_rec_ppocr& rec,
                         const s_detector_params& ocr_config,
                         const std::string& yaml_path,
                         bool show_visual,
                         bool show_ocr_roi,
                         bool show_debug_panels,
                         const std::string& debug_snapshot_dir)
{
    auto logger = node->get_logger();
    int problem_id = 0;
    bool yaml_header_written = false;
    OCRMultiFrameVoter voter;
    cv::Rect2f stable_roi;
    bool has_stable_roi = false;

    RCLCPP_INFO(logger, "MODE       : TEST (continuous OCR + YAML output)");
    RCLCPP_INFO(logger, "YAML output: %s", yaml_path.c_str());
    RCLCPP_INFO(logger, "Press Q or ESC to exit.");

    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);

        cv::Mat frame;
        if (!hik.get_one_frame(frame, 0))
        {
            if (!ensure_camera(hik, logger))
            {
                RCLCPP_ERROR(logger, "Cannot recover camera, exiting.");
                break;
            }
            continue;
        }

        cv::Mat undistorted = undistort_image(frame);
        if (!undistorted.empty())
        {
            frame = undistorted;
        }
        else
        {
            RCLCPP_WARN(logger, "Undistortion failed, using raw frame.");
        }

        std::string expr_str;
        int int_result = 0;
        int mod_result = 0;
        cv::Rect2f math_roi;
        cv::Mat ocr_roi;
        OCRDebugFrame debug;
        std::optional<OCRVoteResult> frame_result;
        if (run_ocr_pipeline(
                frame, det, rec, ocr_config, logger,
                expr_str, int_result, mod_result, &math_roi,
                &ocr_roi, &debug))
        {
            RCLCPP_INFO(logger, "Expr: %s  =>  %d  %%4 = %d",
                        expr_str.c_str(), int_result, mod_result);
            frame_result = OCRVoteResult{expr_str, int_result, mod_result};
            stable_roi = math_roi;
            has_stable_roi = math_roi.area() > 0.0f;
        }

        const OCRVoteEvent event = voter.update(frame_result);
        if (event == OCRVoteEvent::StableChanged)
        {
            const OCRVoteResult& stable = voter.stable_result();
            ++problem_id;

            std::ofstream ofs(yaml_path, std::ios::app);
            if (ofs.is_open())
            {
                if (!yaml_header_written)
                {
                    ofs << "ocr_results:" << std::endl;
                    yaml_header_written = true;
                }
                ofs << "  - id: " << problem_id << std::endl;
                ofs << "    question: \"" << stable.expr << "\"" << std::endl;
                ofs << "    answer: " << stable.result << std::endl;
                ofs << "    mod4: " << stable.mod4 << std::endl;
                RCLCPP_INFO(logger, "Stable OCR : %s => %d (mod4=%d)",
                            stable.expr.c_str(), stable.result, stable.mod4);
                RCLCPP_INFO(logger, "YAML write : id=%d", problem_id);
            }
            else
            {
                RCLCPP_WARN(logger, "Cannot open YAML: %s", yaml_path.c_str());
            }
        }
        else if (event == OCRVoteEvent::StableLost)
        {
            has_stable_roi = false;
            RCLCPP_INFO(logger, "Stable OCR lost after 10 invalid frames.");
        }

        if (show_visual && voter.has_stable_result() && has_stable_roi)
        {
            const OCRVoteResult& stable = voter.stable_result();
            draw_result_overlay(
                frame, stable_roi, stable.expr, stable.result, stable.mod4);
        }

        fill_voter_debug(debug, voter);
        const cv::Mat debug_panel = make_debug_panel(debug);
        const VisualizationKey key = handle_visualization(
            frame, ocr_roi, debug_panel,
            show_visual, show_ocr_roi, show_debug_panels, 10);
        if (key.save_snapshot)
        {
            save_debug_snapshot(
                debug, frame, ocr_roi, debug_panel,
                debug_snapshot_dir, logger);
        }
        if (key.should_exit)
        {
            RCLCPP_INFO(logger, "Exit by user.");
            break;
        }
    }

    RCLCPP_INFO(logger, "Test mode finished. Total stable changes: %d", problem_id);
    return 0;
}

/**
 * @brief 运行基于触发的话题生产模式 OCR。
 * @param node ROS2 节点对象。
 * @param hik 相机封装对象。
 * @param det 文本检测模型封装对象。
 * @param rec 文本识别模型封装对象。
 * @param show_visual 是否启用 OpenCV 可视化窗口。
 * @param show_ocr_roi 是否显示当前最优算术候选。
 * @retval int 类进程退出码。
 */
static int run_production_mode(const rclcpp::Node::SharedPtr& node,
                               HikGrab& hik,
                               detect_det_ppocr& det,
                               detect_rec_ppocr& rec,
                               const s_detector_params& ocr_config,
                               bool show_visual,
                               bool show_ocr_roi,
                               bool show_debug_panels,
                               bool enable_keyboard_trigger,
                               const std::string& debug_snapshot_dir)
{
    auto logger = node->get_logger();
    auto latched_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    auto trigger_sub = node->create_subscription<std_msgs::msg::String>(
        "/ocr/trigger", rclcpp::QoS(1), ocr_trigger_callback);
    auto result_pub = node->create_publisher<std_msgs::msg::String>("/ocr/result", latched_qos);
    auto answer_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    auto answer_pub = node->create_publisher<std_msgs::msg::UInt8>("/ocr/answer", answer_qos);

    std::thread keyboard_thread;
    if (enable_keyboard_trigger)
    {
        keyboard_thread = std::thread([]() {
            std::string line;
            while (g_ocr_running.load() && rclcpp::ok() &&
                   std::getline(std::cin, line))
            {
                g_ocr_triggered.store(true);
            }
        });
    }

    RCLCPP_INFO(logger, "MODE       : PRODUCTION (trigger-based)");
    RCLCPP_INFO(logger, "Subscribed : /ocr/trigger");
    RCLCPP_INFO(logger, "Publishing : /ocr/result (transient_local)");
    RCLCPP_INFO(logger, "Publishing : /ocr/answer (UInt8, volatile)");
    RCLCPP_INFO(logger, "Keyboard   : %s",
                enable_keyboard_trigger ? "Enter enabled" : "disabled");
    RCLCPP_INFO(logger, "Waiting for trigger%s...",
                enable_keyboard_trigger ? " or Enter" : "");

    OCRMultiFrameVoter voter;
    cv::Rect2f stable_roi;
    bool has_stable_roi = false;
    bool tracking_active = false;
    cv::Mat last_display_frame;
    cv::Mat last_ocr_roi;
    cv::Mat last_debug_panel;
    OCRDebugFrame last_debug;
    rclcpp::WallRate idle_rate(20);
    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);
        if (g_ocr_triggered.exchange(false))
        {
            voter.reset();
            stable_roi = cv::Rect2f();
            has_stable_roi = false;
            tracking_active = true;
            RCLCPP_INFO(logger, "Trigger received, OCR tracking reset and started.");
        }

        if (!tracking_active)
        {
            if (show_visual || show_ocr_roi || show_debug_panels)
            {
                const VisualizationKey key = handle_visualization(
                    last_display_frame.empty()
                        ? cv::Mat::zeros(240, 640, CV_8UC3)
                        : last_display_frame,
                    last_ocr_roi,
                    last_debug_panel,
                    show_visual, show_ocr_roi, show_debug_panels, 1);
                if (key.save_snapshot)
                {
                    save_debug_snapshot(
                        last_debug, last_display_frame, last_ocr_roi,
                        last_debug_panel, debug_snapshot_dir, logger);
                }
                if (key.should_exit)
                {
                    RCLCPP_INFO(logger, "Exit by user.");
                    break;
                }
            }
            idle_rate.sleep();
            continue;
        }

        cv::Mat frame;
        if (!hik.get_one_frame(frame, 0))
        {
            RCLCPP_ERROR(logger, "Failed to grab frame, reconnecting...");
            if (ensure_camera(hik, logger))
            {
                continue;
            }
            RCLCPP_ERROR(logger, "Cannot recover camera, exiting.");
            break;
        }

        cv::Mat undistorted = undistort_image(frame);
        if (!undistorted.empty())
        {
            frame = undistorted;
        }

        std::string expr_str;
        int int_result = 0;
        int mod_result = 0;
        cv::Rect2f math_roi;
        cv::Mat ocr_roi;
        OCRDebugFrame debug;
        std::optional<OCRVoteResult> frame_result;
        if (run_ocr_pipeline(
                frame, det, rec, ocr_config, logger,
                expr_str, int_result, mod_result, &math_roi,
                &ocr_roi, &debug))
        {
            frame_result = OCRVoteResult{expr_str, int_result, mod_result};
            stable_roi = math_roi;
            has_stable_roi = math_roi.area() > 0.0f;
        }
        if (!ocr_roi.empty())
        {
            last_ocr_roi = ocr_roi;
        }

        const OCRVoteEvent event = voter.update(frame_result);
        if (event == OCRVoteEvent::StableChanged)
        {
            const OCRVoteResult& stable = voter.stable_result();
            Json::Value result_json;
            result_json["expr"] = stable.expr;
            result_json["result"] = stable.result;
            result_json["mod4"] = stable.mod4;

            Json::FastWriter writer;
            std_msgs::msg::String msg;
            msg.data = writer.write(result_json);
            result_pub->publish(msg);
            RCLCPP_INFO(logger, "Published  : %s", msg.data.c_str());

            std_msgs::msg::UInt8 answer_msg;
            answer_msg.data = static_cast<uint8_t>(stable.mod4);
            answer_pub->publish(answer_msg);
            RCLCPP_INFO(logger, "Answer     : mod4=%u published to /ocr/answer",
                        static_cast<unsigned>(answer_msg.data));

            tracking_active = false;
            RCLCPP_INFO(logger, "OCR cycle complete. Waiting for next trigger.");
        }
        else if (event == OCRVoteEvent::StableLost)
        {
            has_stable_roi = false;
            RCLCPP_INFO(logger, "Stable OCR lost after 10 invalid frames.");
        }

        if (show_visual && voter.has_stable_result() && has_stable_roi)
        {
            const OCRVoteResult& stable = voter.stable_result();
            draw_result_overlay(
                frame, stable_roi, stable.expr, stable.result, stable.mod4);
        }
        last_display_frame = frame.clone();
        fill_voter_debug(debug, voter);
        last_debug_panel = make_debug_panel(debug);
        last_debug = debug;

        const VisualizationKey key = handle_visualization(
            frame, ocr_roi, last_debug_panel,
            show_visual, show_ocr_roi, show_debug_panels, 1);
        if (key.save_snapshot)
        {
            save_debug_snapshot(
                debug, frame, ocr_roi, last_debug_panel,
                debug_snapshot_dir, logger);
        }
        if (key.should_exit)
        {
            RCLCPP_INFO(logger, "Exit by user.");
            break;
        }
    }

    (void)trigger_sub;
    g_ocr_running.store(false);
    if (keyboard_thread.joinable())
    {
        keyboard_thread.detach();
    }
    RCLCPP_INFO(logger, "Production mode finished.");
    return 0;
}

static bool is_image_file(const fs::path& path)
{
    if (!path.has_extension())
        return false;
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
           ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

static std::vector<fs::path> collect_visual_inputs(const std::string& input_path)
{
    std::vector<fs::path> inputs;
    if (input_path.empty())
        return inputs;

    const fs::path path(input_path);
    std::error_code ec;
    if (fs::is_regular_file(path, ec) && is_image_file(path))
    {
        inputs.push_back(path);
        return inputs;
    }
    if (!fs::is_directory(path, ec))
        return inputs;

    for (const auto& entry : fs::directory_iterator(path, ec))
    {
        if (entry.is_regular_file() && is_image_file(entry.path()))
            inputs.push_back(entry.path());
    }
    std::sort(inputs.begin(), inputs.end());
    return inputs;
}

static int run_visual_test_mode(const rclcpp::Node::SharedPtr& node,
                                detect_det_ppocr& det,
                                detect_rec_ppocr& rec,
                                const s_detector_params& ocr_config,
                                const std::string& input_path,
                                bool loop,
                                int wait_ms,
                                bool show_visual,
                                bool show_ocr_roi,
                                bool show_debug_panels,
                                const std::string& debug_snapshot_dir)
{
    auto logger = node->get_logger();
    const std::vector<fs::path> inputs = collect_visual_inputs(input_path);
    if (inputs.empty())
    {
        RCLCPP_ERROR(logger, "No visual test image found at: %s", input_path.c_str());
        return 1;
    }

    RCLCPP_INFO(logger, "MODE       : VISUAL_TEST (offline image playback)");
    RCLCPP_INFO(logger, "Input      : %s (%zu image%s)",
                input_path.c_str(), inputs.size(), inputs.size() == 1 ? "" : "s");
    RCLCPP_INFO(logger, "Keys       : q/ESC exit, s save, space pause, n next");

    OCRMultiFrameVoter voter;
    bool paused = inputs.size() == 1 || wait_ms <= 0;
    std::size_t index = 0;

    while (rclcpp::ok() && index < inputs.size())
    {
        rclcpp::spin_some(node);

        cv::Mat frame = cv::imread(inputs[index].string(), cv::IMREAD_COLOR);
        if (frame.empty())
        {
            RCLCPP_WARN(logger, "Cannot read image: %s", inputs[index].string().c_str());
            ++index;
            continue;
        }

        cv::Mat undistorted = undistort_image(frame);
        if (!undistorted.empty())
            frame = undistorted;

        std::string expr_str;
        int int_result = 0;
        int mod_result = 0;
        cv::Rect2f math_roi;
        cv::Mat ocr_roi;
        OCRDebugFrame debug;
        std::optional<OCRVoteResult> frame_result;
        if (run_ocr_pipeline(
                frame, det, rec, ocr_config, logger,
                expr_str, int_result, mod_result, &math_roi,
                &ocr_roi, &debug))
        {
            frame_result = OCRVoteResult{expr_str, int_result, mod_result};
        }
        debug.source_name = inputs[index].filename().string();

        const OCRVoteEvent event = voter.update(frame_result);
        (void)event;
        fill_voter_debug(debug, voter);
        if (show_visual && voter.has_stable_result() && math_roi.area() > 0.0f)
        {
            const OCRVoteResult& stable = voter.stable_result();
            draw_result_overlay(frame, math_roi, stable.expr, stable.result, stable.mod4);
        }

        cv::putText(frame, inputs[index].filename().string(),
                    cv::Point(16, 32), cv::FONT_HERSHEY_SIMPLEX,
                    0.75, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        const cv::Mat debug_panel = make_debug_panel(debug);

        const bool any_window = show_visual || show_ocr_roi || show_debug_panels;
        bool advance = !any_window;
        while (rclcpp::ok() && !advance)
        {
            const int key_wait = paused ? 0 : std::max(1, wait_ms);
            const VisualizationKey key = handle_visualization(
                frame, ocr_roi, debug_panel,
                show_visual, show_ocr_roi, show_debug_panels, key_wait);
            if (key.save_snapshot)
            {
                save_debug_snapshot(
                    debug, frame, ocr_roi, debug_panel,
                    debug_snapshot_dir, logger);
            }
            if (key.should_exit)
            {
                RCLCPP_INFO(logger, "Exit by user.");
                return 0;
            }
            if (key.pause_toggle)
            {
                paused = !paused;
            }
            if (key.next_frame || !paused)
            {
                advance = true;
            }
        }

        ++index;
        if (index >= inputs.size() && loop)
            index = 0;
    }

    RCLCPP_INFO(logger, "Visual test finished.");
    return 0;
}

/**
 * @brief 运行 ROS2 PPOCR 节点入口。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @retval int 进程退出码。
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("ppocr_node");
    auto logger = node->get_logger();

    const std::string share_dir = ament_index_cpp::get_package_share_directory("dogvision_vision");
    node->declare_parameter<std::string>("config_path", share_dir + "/config/settings.json");
    node->declare_parameter<std::string>("mode", "production");
    node->declare_parameter<bool>("show_visual", true);
    node->declare_parameter<bool>("show_ocr_roi", false);
    node->declare_parameter<bool>("show_debug_panels", true);
    node->declare_parameter<bool>("enable_keyboard_trigger", true);
    node->declare_parameter<std::string>("yaml_path", share_dir + "/data/ocr_output/ocr_results.yaml");
    node->declare_parameter<std::string>("debug_snapshot_dir", share_dir + "/data/ocr_debug");
    node->declare_parameter<std::string>("input_path", "");
    node->declare_parameter<bool>("loop", false);
    node->declare_parameter<int>("wait_ms", 1000);

    const std::string config_path = node->get_parameter("config_path").as_string();
    const std::string mode = node->get_parameter("mode").as_string();
    const bool show_visual = node->get_parameter("show_visual").as_bool();
    const bool show_ocr_roi =
        node->get_parameter("show_ocr_roi").as_bool();
    const bool show_debug_panels =
        node->get_parameter("show_debug_panels").as_bool();
    const bool enable_keyboard_trigger =
        node->get_parameter("enable_keyboard_trigger").as_bool();
    const std::string yaml_path = node->get_parameter("yaml_path").as_string();
    const std::string debug_snapshot_dir =
        node->get_parameter("debug_snapshot_dir").as_string();
    const std::string input_path = node->get_parameter("input_path").as_string();
    const bool loop = node->get_parameter("loop").as_bool();
    const int wait_ms = node->get_parameter("wait_ms").as_int();

    Appconfig config;
    {
        detect_det_ppocr loader(nullptr);
        loader.load_config(config, config_path);
    }
    RCLCPP_INFO(logger, "Config     : %s", config_path.c_str());

    detect_det_ppocr det(&config);
    det.load_model(config.detect_config.ppocr_det_model_path, config.detect_config.det_device);

    detect_rec_ppocr rec(&config);
    rec.load_model(config.detect_config.ppocr_rec_model_path, config.detect_config.rec_device);
    rec.loda_dict(config.detect_config.rec_char_dict_path);
    rec.load_allowed_chars(config.detect_config.rec_allowed_chars_path);

    const float default_wh_ratio =
        (config.detect_config.rec_img_h > 0)
            ? static_cast<float>(config.detect_config.rec_img_w) / static_cast<float>(config.detect_config.rec_img_h)
            : 320.0f / 48.0f;
    rec.set_max_wh_ratio(default_wh_ratio);

    init_fisheye_undistort(
        config.hikcamera_config.width,
        config.hikcamera_config.height);

    std::error_code ec;
    fs::create_directories(fs::path(yaml_path).parent_path(), ec);
    if (ec)
    {
        RCLCPP_WARN(logger, "Cannot create YAML dir: %s", ec.message().c_str());
    }
    fs::create_directories(debug_snapshot_dir, ec);
    if (ec)
    {
        RCLCPP_WARN(logger, "Cannot create OCR debug dir: %s", ec.message().c_str());
    }

    int ret = 0;
    if (mode == "visual_test" && !input_path.empty())
    {
        ret = run_visual_test_mode(
            node, det, rec, config.detect_config,
            input_path, loop, wait_ms,
            show_visual, show_ocr_roi, show_debug_panels,
            debug_snapshot_dir);
    }
    else if (mode == "test" || mode == "production" || mode == "visual_test")
    {
        s_camera_params cam_params{};
        cam_params.device_id = config.hikcamera_config.device_id;
        cam_params.width = config.hikcamera_config.width;
        cam_params.height = config.hikcamera_config.height;
        cam_params.offset_x = config.hikcamera_config.offset_x;
        cam_params.offset_y = config.hikcamera_config.offset_y;
        cam_params.exposure = config.hikcamera_config.exposure;

        HikGrab hik(cam_params);
        hik.Hik_init();
        RCLCPP_INFO(logger, "Camera     : device=%d  %dx%d",
                    cam_params.device_id, cam_params.width, cam_params.height);

        if (mode == "production")
        {
            ret = run_production_mode(
                node, hik, det, rec, config.detect_config,
                show_visual, show_ocr_roi, show_debug_panels,
                enable_keyboard_trigger, debug_snapshot_dir);
        }
        else
        {
            ret = run_test_mode(
                node, hik, det, rec, config.detect_config, yaml_path,
                show_visual, show_ocr_roi, show_debug_panels,
                debug_snapshot_dir);
        }
        hik.Hik_end();
    }
    else
    {
        RCLCPP_ERROR(logger, "Unsupported mode '%s'. Use 'test', 'production', or 'visual_test'.", mode.c_str());
        ret = 1;
    }

    if (show_visual || show_ocr_roi || show_debug_panels)
    {
        cv::destroyAllWindows();
    }
    rclcpp::shutdown();
    return ret;
}
