#include <dogvision_vision/ocr_utils.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <opencv2/calib3d.hpp>

cv::Mat prepare_ocr_input(const cv::Mat& input, bool use_grayscale)
{
    if (input.empty())
        return {};

    cv::Mat bgr;
    if (input.channels() == 3)
        bgr = input.clone();
    else if (input.channels() == 1)
        cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
    else if (input.channels() == 4)
        cv::cvtColor(input, bgr, cv::COLOR_BGRA2BGR);
    else
        throw std::invalid_argument("OCR ROI must have 1, 3, or 4 channels");

    if (!use_grayscale)
        return bgr;

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

// ============================================================
//  crop_text_region
// ============================================================
cv::Mat crop_text_region(const cv::Mat& src, const OCRBox& box)
{
    const float w = std::max(
        static_cast<float>(cv::norm(box.pts[0] - box.pts[1])),
        static_cast<float>(cv::norm(box.pts[2] - box.pts[3])));
    const float h = std::max(
        static_cast<float>(cv::norm(box.pts[0] - box.pts[3])),
        static_cast<float>(cv::norm(box.pts[1] - box.pts[2])));

    if (w < 1.0f || h < 1.0f)
        return {};

    cv::Point2f src_pts[4], dst_pts[4];
    for (int i = 0; i < 4; ++i)
        src_pts[i] = box.pts[i];
    dst_pts[0] = {0.0f,     0.0f    };
    dst_pts[1] = {w - 1.0f, 0.0f    };
    dst_pts[2] = {w - 1.0f, h - 1.0f};
    dst_pts[3] = {0.0f,     h - 1.0f};

    cv::Mat transform = cv::getPerspectiveTransform(src_pts, dst_pts);
    cv::Mat crop;
    cv::warpPerspective(src, crop, transform,
                        cv::Size(static_cast<int>(w), static_cast<int>(h)));

    // 竖排文本（高 > 宽 × 1.5）旋转90°使其横排
    if (crop.rows > static_cast<int>(crop.cols * 1.5f))
        cv::rotate(crop, crop, cv::ROTATE_90_CLOCKWISE);

    return crop;
}

// ============================================================
//  draw_ocr_result
// ============================================================
void draw_ocr_result(cv::Mat& vis, const OCRBox& box, const OCRRecResult& rec)
{
    std::vector<cv::Point> poly;
    for (int i = 0; i < 4; ++i)
        poly.emplace_back(static_cast<int>(std::round(box.pts[i].x)),
                          static_cast<int>(std::round(box.pts[i].y)));
    cv::polylines(vis, poly, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    const int text_y = std::max(0, static_cast<int>(std::round(box.pts[0].y)) - 5);
    std::ostringstream label;
    label << rec.text << " " << std::fixed << std::setprecision(2) << rec.score;
    cv::putText(vis, label.str(),
                cv::Point(static_cast<int>(std::round(box.pts[0].x)), text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
}

// ============================================================
//  算术解析辅助（内部使用，不对外暴露）
// ============================================================

// 将 OCR 输出的中文/全角运算符归一化为 ASCII
static std::string normalize_expr(const std::string& src)
{
    std::string s = src;
    const std::pair<std::string, std::string> repl[] = {
        {"×", "*"}, {"÷", "/"}, {"＋", "+"}, {"－", "-"},
        {"（", "("}, {"）", ")"},
        {"X", "*"}, {"x", "*"},
        {" ", ""},
    };
    for (const auto& p : repl) {
        size_t pos = 0;
        while ((pos = s.find(p.first, pos)) != std::string::npos) {
            s.replace(pos, p.first.size(), p.second);
            pos += p.second.size();
        }
    }
    return s;
}

// 递归下降解析器（支持括号与标准四则运算优先级）
// 语法：
//   expr   = term  { ('+' | '-') term  }
//   term   = factor{ ('*' | '/') factor}
//   factor = '(' expr ')' | ['-'] number
struct Parser {
    const std::string& s;
    size_t pos = 0;
    int number_count = 0;
    int binary_operator_count = 0;
    bool valid = true;

    explicit Parser(const std::string& str) : s(str) {}

    void skip_ws() { while (pos < s.size() && s[pos] == ' ') ++pos; }
    char peek()    { skip_ws(); return pos < s.size() ? s[pos] : '\0'; }
    char consume() { skip_ws(); return pos < s.size() ? s[pos++] : '\0'; }

    bool read_number(double& val) {
        skip_ws();
        const size_t start = pos;
        bool has_digit = false;
        bool has_dot = false;
        while (pos < s.size())
        {
            const unsigned char c = static_cast<unsigned char>(s[pos]);
            if (std::isdigit(c))
            {
                has_digit = true;
                ++pos;
            }
            else if (s[pos] == '.' && !has_dot)
            {
                has_dot = true;
                ++pos;
            }
            else
            {
                break;
            }
        }
        if (!has_digit)
        {
            pos = start;
            return false;
        }
        try
        {
            val = std::stod(s.substr(start, pos - start));
        }
        catch (const std::exception&)
        {
            valid = false;
            return false;
        }
        ++number_count;
        return std::isfinite(val);
    }

    bool factor(double& value) {
        skip_ws();
        if (pos >= s.size()) return false;
        bool negative = false;
        if (s[pos] == '-')
        {
            negative = true;
            ++pos;
        }
        if (pos < s.size() && s[pos] == '(') {
            ++pos;
            if (!expr(value))
                return false;
            skip_ws();
            if (pos >= s.size() || s[pos] != ')')
                return false;
            ++pos;
            if (negative)
                value = -value;
            return true;
        }
        if (!read_number(value))
            return false;
        if (negative)
            value = -value;
        return true;
    }

    bool term(double& value) {
        if (!factor(value))
            return false;
        while (true) {
            char c = peek();
            if (c != '*' && c != '/')
                break;
            consume();
            double rhs = 0.0;
            if (!factor(rhs))
                return false;
            ++binary_operator_count;
            if (c == '/' && std::abs(rhs) <= 1e-12)
                return false;
            value = (c == '*') ? value * rhs : value / rhs;
            if (!std::isfinite(value))
                return false;
        }
        return true;
    }

    bool expr(double& value) {
        if (!term(value))
            return false;
        while (true) {
            char c = peek();
            if (c != '+' && c != '-')
                break;
            consume();
            double rhs = 0.0;
            if (!term(rhs))
                return false;
            ++binary_operator_count;
            value = (c == '+') ? value + rhs : value - rhs;
            if (!std::isfinite(value))
                return false;
        }
        return true;
    }
};

// ============================================================
//  parse_simple_expr
// ============================================================
bool parse_simple_expr(const std::string& text, double& result, std::string& expr_str)
{
    std::string norm = normalize_expr(text);
    if (!norm.empty() && norm.back() == '=')
        norm.pop_back();
    if (norm.empty() || norm.find('=') != std::string::npos)
        return false;

    for (const unsigned char c : norm)
    {
        if (!std::isdigit(c) &&
            c != '.' && c != '+' && c != '-' &&
            c != '*' && c != '/' && c != '(' && c != ')')
        {
            return false;
        }
    }

    Parser parser(norm);
    double value = 0.0;
    if (!parser.expr(value))
        return false;
    parser.skip_ws();
    if (!parser.valid || parser.pos != norm.size() ||
        parser.number_count < 2 || parser.binary_operator_count < 1 ||
        !std::isfinite(value))
        return false;

    expr_str = norm;
    result = value;
    return true;
}

cv::Rect ocr_box_bounds(const OCRBox& box, const cv::Size& image_size)
{
    std::vector<cv::Point2f> points(box.pts.begin(), box.pts.end());
    cv::Rect bounds = cv::boundingRect(points);
    bounds &= cv::Rect(0, 0, image_size.width, image_size.height);
    return bounds;
}

float calculate_surround_white_ratio(
    const cv::Mat& image,
    const cv::Rect& text_bounds,
    float average_text_height,
    const s_detector_params& config,
    cv::Rect* surround_bounds)
{
    if (image.empty() || text_bounds.empty() || average_text_height <= 0.0f)
        return 0.0f;

    const int margin = std::max(
        1, static_cast<int>(std::round(
               average_text_height * config.ocr_math_surround_margin_ratio)));
    cv::Rect outer(
        text_bounds.x - margin,
        text_bounds.y - margin,
        text_bounds.width + margin * 2,
        text_bounds.height + margin * 2);
    outer &= cv::Rect(0, 0, image.cols, image.rows);
    if (surround_bounds != nullptr)
        *surround_bounds = outer;
    if (outer.empty())
        return 0.0f;

    cv::Mat bgr = prepare_ocr_input(image, false);
    cv::Mat hsv;
    cv::cvtColor(bgr(outer), hsv, cv::COLOR_BGR2HSV);
    cv::Mat white_mask;
    cv::inRange(
        hsv,
        cv::Scalar(0, 0, config.ocr_math_white_v_min),
        cv::Scalar(180, config.ocr_math_white_s_max, 255),
        white_mask);

    cv::Mat ring_mask(outer.size(), CV_8UC1, cv::Scalar(255));
    cv::Rect inner = text_bounds & outer;
    inner.x -= outer.x;
    inner.y -= outer.y;
    inner &= cv::Rect(0, 0, outer.width, outer.height);
    if (!inner.empty())
        cv::rectangle(ring_mask, inner, cv::Scalar(0), cv::FILLED);

    const int ring_pixels = cv::countNonZero(ring_mask);
    if (ring_pixels <= 0)
        return 0.0f;
    cv::Mat white_ring;
    cv::bitwise_and(white_mask, ring_mask, white_ring);
    return static_cast<float>(cv::countNonZero(white_ring)) /
           static_cast<float>(ring_pixels);
}

namespace
{
struct ItemGeometry
{
    size_t index = 0;
    cv::Rect bounds;
    float center_y = 0.0f;
};

float vertical_overlap_ratio(const cv::Rect& a, const cv::Rect& b)
{
    const int overlap =
        std::max(0, std::min(a.y + a.height, b.y + b.height) -
                        std::max(a.y, b.y));
    return static_cast<float>(overlap) /
           static_cast<float>(std::max(1, std::min(a.height, b.height)));
}

OCRMathCandidate make_math_candidate(
    const cv::Mat& image,
    const std::vector<OCRItem>& items,
    const std::vector<size_t>& indices,
    const s_detector_params& config)
{
    OCRMathCandidate candidate;
    candidate.item_indices = indices;
    float height_sum = 0.0f;
    float score_sum = 0.0f;
    for (const size_t index : indices)
    {
        const cv::Rect bounds = ocr_box_bounds(items[index].box, image.size());
        candidate.text_bounds =
            candidate.text_bounds.empty()
                ? bounds
                : (candidate.text_bounds | bounds);
        candidate.raw_text += items[index].rec.text;
        height_sum += static_cast<float>(bounds.height);
        score_sum += items[index].rec.score;
    }
    const float count = static_cast<float>(indices.size());
    candidate.mean_score = count > 0.0f ? score_sum / count : 0.0f;
    const float average_height = count > 0.0f ? height_sum / count : 0.0f;
    candidate.surround_white_ratio = calculate_surround_white_ratio(
        image, candidate.text_bounds, average_height, config,
        &candidate.surround_bounds);
    candidate.white_pass =
        candidate.surround_white_ratio >=
        static_cast<float>(config.ocr_math_min_surround_white_ratio);
    return candidate;
}
} // namespace

std::vector<OCRMathCandidate> find_math_candidates(
    const cv::Mat& original_image,
    const std::vector<OCRItem>& items,
    const s_detector_params& config)
{
    std::vector<OCRMathCandidate> candidates;
    if (original_image.empty() || items.empty())
        return candidates;

    std::vector<ItemGeometry> geometries;
    geometries.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i)
    {
        const cv::Rect bounds = ocr_box_bounds(items[i].box, original_image.size());
        if (!bounds.empty())
        {
            geometries.push_back({
                i, bounds,
                static_cast<float>(bounds.y) + bounds.height * 0.5f});
        }
    }
    std::sort(geometries.begin(), geometries.end(),
              [](const ItemGeometry& a, const ItemGeometry& b)
              {
                  if (std::abs(a.center_y - b.center_y) < 1.0f)
                      return a.bounds.x < b.bounds.x;
                  return a.center_y < b.center_y;
              });

    std::vector<std::vector<ItemGeometry>> lines;
    for (const ItemGeometry& geometry : geometries)
    {
        bool assigned = false;
        for (auto& line : lines)
        {
            cv::Rect line_bounds;
            for (const auto& existing : line)
                line_bounds = line_bounds.empty()
                    ? existing.bounds : (line_bounds | existing.bounds);
            const float center_delta = std::abs(
                geometry.center_y -
                (static_cast<float>(line_bounds.y) + line_bounds.height * 0.5f));
            if (vertical_overlap_ratio(geometry.bounds, line_bounds) >= 0.40f ||
                center_delta <=
                    0.60f * static_cast<float>(
                                std::max(geometry.bounds.height, line_bounds.height)))
            {
                line.push_back(geometry);
                assigned = true;
                break;
            }
        }
        if (!assigned)
            lines.push_back({geometry});
    }

    for (auto& line : lines)
    {
        std::sort(line.begin(), line.end(),
                  [](const ItemGeometry& a, const ItemGeometry& b)
                  {
                      return a.bounds.x < b.bounds.x;
                  });

        size_t segment_begin = 0;
        while (segment_begin < line.size())
        {
            size_t segment_end = segment_begin + 1;
            while (segment_end < line.size())
            {
                const cv::Rect& previous = line[segment_end - 1].bounds;
                const cv::Rect& current = line[segment_end].bounds;
                const int gap = current.x - (previous.x + previous.width);
                const float max_gap =
                    3.0f * static_cast<float>(
                               std::max(previous.height, current.height));
                if (gap > max_gap)
                    break;
                ++segment_end;
            }

            for (size_t begin = segment_begin; begin < segment_end; ++begin)
            {
                std::vector<size_t> indices;
                std::string text;
                for (size_t end = begin; end < segment_end; ++end)
                {
                    indices.push_back(line[end].index);
                    text += items[line[end].index].rec.text;
                    double result = 0.0;
                    std::string expression;
                    if (!parse_simple_expr(text, result, expression))
                        continue;
                    OCRMathCandidate candidate =
                        make_math_candidate(
                            original_image, items, indices, config);
                    candidate.expression = expression;
                    candidate.result = result;
                    const float length_bonus = std::min(
                        0.10f,
                        static_cast<float>(expression.size()) / 100.0f);
                    candidate.rank_score =
                        candidate.mean_score * 0.45f +
                        candidate.surround_white_ratio * 0.45f +
                        length_bonus;
                    candidates.push_back(std::move(candidate));
                }
            }
            segment_begin = segment_end;
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const OCRMathCandidate& a, const OCRMathCandidate& b)
              {
                  if (a.white_pass != b.white_pass)
                      return a.white_pass > b.white_pass;
                  if (std::abs(a.rank_score - b.rank_score) > 1e-6f)
                      return a.rank_score > b.rank_score;
                  return a.expression.size() > b.expression.size();
              });
    return candidates;
}

// ============================================================
//  show_result_window
// ============================================================
void show_result_window(const std::string& expr_str, int mod_result)
{
    cv::Mat canvas(200, 500, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::putText(canvas, "Expr: " + expr_str,
                cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX,
                0.9, cv::Scalar(0, 230, 0), 2, cv::LINE_AA);
    cv::putText(canvas, "Result % 4 = " + std::to_string(mod_result),
                cv::Point(20, 140), cv::FONT_HERSHEY_SIMPLEX,
                1.1, cv::Scalar(0, 200, 255), 2, cv::LINE_AA);
    cv::imshow("OCR Arithmetic Result", canvas);
    cv::waitKey(0);
    cv::destroyWindow("OCR Arithmetic Result");
}

// ============================================================
//  find_math_proble
//  在画面中定位白底黑字算术题区域，返回其边界矩形。
//  算法：
//    1. BGR → HSV，提取低饱和度 + 高明度的白色掩码
//    2. 形态学闭运算（填充文字孔洞）+ 开运算（去除零散噪点）
//    3. 查找最外层轮廓，按面积 / 宽高比 / 白色覆盖率筛选候选
//    4. 返回"面积 × 白色比例"得分最高的候选矩形
// ============================================================
cv::Rect2f find_math_proble(const cv::Mat& input,
                            cv::Mat* mask_out,
                            int white_s_max,
                            int white_v_min)
{
    if (input.empty()) return {};

    // ── 1. HSV 白色掩码 ──────────────────────────────────────────────────────
    // 阈值可通过参数调整：
    //   white_s_max = S 上限，越大越容忍近似的白色（如米白、浅灰）
    //   white_v_min = V 下限，越小越容忍暗处的白色（如阴影中的白纸）
    cv::Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

    cv::Mat white_mask;
    cv::inRange(hsv,
            cv::Scalar(0, 0, white_v_min),
            cv::Scalar(180, white_s_max, 255),
                white_mask);

    // ── 2. 多级形态学处理 ────────────────────────────────────────────────────
    // 使用一大一小两级核，分别处理近距离（大 ROI）和远距离（小 ROI）：
    //
    //   大核 ks_large = 常规尺寸（≈ 图像宽/60），用于连通近距离大目标的文字间隙
    //   小核 ks_small = 小尺寸（固定 5px），保留远距离小目标的白色区域不被抹除
    //
    // 先闭后开是大核（去除文字孔洞+噪点），再对小核候选做保护性闭运算

    int ks_large = std::max(11, input.cols / 60);
    if (ks_large % 2 == 0) ks_large++;
    const int ks_small = 5;  // 远距离小目标专用核

    // ── 2a. 大核处理（保留常规/近距离目标）─────────────────────────────────
    cv::Mat morph_large;
    cv::Mat kernel_large = cv::getStructuringElement(cv::MORPH_RECT, {ks_large, ks_large});
    cv::morphologyEx(white_mask, morph_large, cv::MORPH_CLOSE, kernel_large);
    cv::morphologyEx(morph_large, morph_large, cv::MORPH_OPEN, kernel_large);

    // ── 2b. 小核处理（保留远距离小目标）─────────────────────────────────────
    cv::Mat morph_small;
    cv::Mat kernel_small = cv::getStructuringElement(cv::MORPH_RECT, {ks_small, ks_small});
    cv::morphologyEx(white_mask, morph_small, cv::MORPH_CLOSE, kernel_small);
    // 小目标不再做开运算，避免被腐蚀掉

    // ── 2c. 合并两侧：远距离小目标优先（避免被大核抹除）───────────────────
    // 策略：大核结果保证大区域连续性，小核结果保留小区域；两者取并集
    cv::Mat combined_mask = morph_large | morph_small;

    // ── 3. 轮廓检测 ──────────────────────────────────────────────────────────
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(combined_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return {};

    // ── 4. 候选筛选 ──────────────────────────────────────────────────────────
    const float img_area = static_cast<float>(input.cols * input.rows);

    // 放宽面积下限到 0.5%，让远距离小目标能通过
    const float min_area = img_area * 0.005f;  // 0.5%（原为 3%）
    const float max_area = img_area * 0.92f;   // 不超过 92%（排除满幅白背景）

    cv::Rect2f best;
    float best_score = -1.f;
    int best_contour_idx = -1;

    for (int i = 0; i < static_cast<int>(contours.size()); ++i)
    {
        const auto& c = contours[i];
        cv::Rect br = cv::boundingRect(c);
        float area = static_cast<float>(br.area());

        // 面积过滤
        if (area < min_area || area > max_area) continue;

        // 宽高比过滤：算术题区域通常接近横向矩形
        float ratio = static_cast<float>(br.width) / std::max(1, br.height);
        if (ratio < 0.3f || ratio > 8.0f) continue;

        // 确保矩形在图像边界内
        br &= cv::Rect(0, 0, input.cols, input.rows);
        if (br.area() == 0) continue;

        // 白色覆盖率（该矩形内白色像素 / 矩形总面积）
        cv::Mat roi_mask = white_mask(br);
        float white_ratio = static_cast<float>(cv::countNonZero(roi_mask))
                            / static_cast<float>(br.area());
        if (white_ratio < 0.45f) continue;   // 白色至少占 45%（原 55%，放宽）

        // 综合评分：优先面积大且白色纯度高的区域
        float score = static_cast<float>(br.area()) * white_ratio;
        if (score > best_score)
        {
            best_score = score;
            best = cv::Rect2f(static_cast<float>(br.x),
                              static_cast<float>(br.y),
                              static_cast<float>(br.width),
                              static_cast<float>(br.height));
            best_contour_idx = i;
        }
    }

    // ── 5. 生成精确白色区域掩码（仅最佳轮廓）───────────────────────────────
    if (mask_out != nullptr && best_contour_idx >= 0)
    {
        *mask_out = cv::Mat::zeros(input.size(), CV_8UC1);
        cv::drawContours(*mask_out, contours, best_contour_idx,
                         cv::Scalar(255), cv::FILLED);
    }

    return best;
}

// ============================================================
//  鱼眼去畸变（与 detector.cpp 共享同一份标定参数）
// ============================================================
namespace {

constexpr double kFisheyeBalance = 0.0;
constexpr double kFisheyeFovScale = 1.0;

// 从 fisheye_params.yaml 读取的 K 矩阵 (3×3)
const cv::Mat K = (cv::Mat_<double>(3, 3) <<
    8.2631010840557929e+02, 0., 7.3508237365876721e+02,
    0., 8.3234495506807673e+02, 5.6784864582942498e+02,
    0., 0., 1.);

// 从 fisheye_params.yaml 读取的 D 矩阵 (4×1)
const cv::Mat D = (cv::Mat_<double>(4, 1) <<
    1.9474519085664992e-02,
    2.2096711413330011e-02,
   -4.1006640770500716e-02,
    2.6220651979250005e-02);

cv::Mat s_map1, s_map2;
bool s_map_ready = false;

} // anonymous namespace

bool init_fisheye_undistort(int image_width, int image_height)
{
    if (image_width <= 0 || image_height <= 0)
    {
        std::cerr << "init_fisheye_undistort: invalid size "
                  << image_width << "x" << image_height << std::endl;
        s_map_ready = false;
        return false;
    }

    const cv::Size image_size(image_width, image_height);
    cv::Mat new_camera_matrix;

    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        K, D, image_size, cv::Matx33d::eye(),
        new_camera_matrix, kFisheyeBalance,
        image_size, kFisheyeFovScale);

    cv::fisheye::initUndistortRectifyMap(
        K, D, cv::Matx33d::eye(), new_camera_matrix,
        image_size, CV_16SC2, s_map1, s_map2);

    s_map_ready = !s_map1.empty() && !s_map2.empty();

    if (s_map_ready)
        std::cout << "Fisheye undistort initialized for "
                  << image_width << "x" << image_height << std::endl;
    else
        std::cerr << "Fisheye undistort init FAILED for "
                  << image_width << "x" << image_height << std::endl;

    return s_map_ready;
}

cv::Mat undistort_image(const cv::Mat& input)
{
    if (input.empty())
        return {};

    if (!s_map_ready)
        return input.clone();

    cv::Mat result;
    cv::remap(input, result, s_map1, s_map2,
              cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return result;
}
