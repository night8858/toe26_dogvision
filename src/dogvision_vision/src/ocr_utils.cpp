#include <dogvision_vision/ocr_utils.hpp>

#include <ros/ros.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <utility>

#include <opencv2/calib3d.hpp>

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
        {" ", ""}, {"=", ""}, {"?", ""}, {"？", ""},
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
    size_t pos;

    explicit Parser(const std::string& str) : s(str), pos(0) {}

    void skip_ws() { while (pos < s.size() && s[pos] == ' ') ++pos; }
    char peek()    { skip_ws(); return pos < s.size() ? s[pos] : '\0'; }
    char consume() { skip_ws(); return pos < s.size() ? s[pos++] : '\0'; }

    bool read_number(double& val) {
        skip_ws();
        size_t j = pos;
        while (j < s.size() && (std::isdigit((unsigned char)s[j]) || s[j] == '.')) ++j;
        if (j == pos) return false;
        val = std::stod(s.substr(pos, j - pos));
        pos = j;
        return true;
    }

    double factor() {
        skip_ws();
        if (pos >= s.size()) return 0.0;
        if (s[pos] == '(') {
            ++pos;
            double val = expr();
            skip_ws();
            if (pos < s.size() && s[pos] == ')') ++pos;
            return val;
        }
        bool neg = false;
        if (s[pos] == '-') { neg = true; ++pos; }
        double val = 0.0;
        read_number(val);
        return neg ? -val : val;
    }

    double term() {
        double val = factor();
        while (true) {
            char c = peek();
            if (c == '*' || c == '/') { consume(); val = (c == '*') ? val * factor() : val / factor(); }
            else break;
        }
        return val;
    }

    double expr() {
        double val = term();
        while (true) {
            char c = peek();
            if (c == '+' || c == '-') { consume(); val = (c == '+') ? val + term() : val - term(); }
            else break;
        }
        return val;
    }
};

// ============================================================
//  parse_simple_expr
// ============================================================
bool parse_simple_expr(const std::string& text, double& result, std::string& expr_str)
{
    std::string norm = normalize_expr(text);

    // 找到第一个以数字或左括号开头的子串
    std::regex start_pat(R"([(\-]?\d|[(])");
    std::smatch sm;
    if (!std::regex_search(norm, sm, start_pat))
        return false;

    std::string candidate = norm.substr(sm.position());

    // 至少含一个运算符才是表达式
    if (candidate.find_first_of("+-*/") == std::string::npos)
        return false;

    Parser parser(candidate);
    double val = parser.expr();
    expr_str = candidate.substr(0, parser.pos);

    if (expr_str.find_first_of("+-*/") == std::string::npos)
        return false;

    result = val;
    return true;
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
cv::Rect2f find_math_proble(const cv::Mat& input)
{
    if (input.empty()) return {};

    // ── 1. HSV 白色掩码 ──────────────────────────────────────────────────────
    // 白色定义：低饱和度（S < 50）+ 高明度（V > 180）
    cv::Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

    cv::Mat white_mask;
    cv::inRange(hsv,
                cv::Scalar(0,   0, 180),
                cv::Scalar(180, 50, 255),
                white_mask);

    // ── 2. 形态学处理 ────────────────────────────────────────────────────────
    // 核大小随图像分辨率自适应（最小 11px，约 1/60 图宽）
    int ks = std::max(11, input.cols / 60);
    if (ks % 2 == 0) ks++;   // 保持奇数
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {ks, ks});
    // 闭运算：连通字符之间的白色小孔，使白色区域完整
    cv::morphologyEx(white_mask, white_mask, cv::MORPH_CLOSE, kernel);
    // 开运算：消除孤立噪点
    cv::morphologyEx(white_mask, white_mask, cv::MORPH_OPEN,  kernel);

    // ── 3. 轮廓检测 ──────────────────────────────────────────────────────────
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(white_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return {};

    // ── 4. 候选筛选 ──────────────────────────────────────────────────────────
    const float img_area = static_cast<float>(input.cols * input.rows);
    const float min_area = img_area * 0.03f;   // 至少占画面 3%（排除零散噪点）
    const float max_area = img_area * 0.92f;   // 不超过 92%（排除满幅白背景）

    cv::Rect2f best;
    float best_score = -1.f;

    for (const auto& c : contours)
    {
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
        if (white_ratio < 0.55f) continue;   // 白色至少占 55%

        // 综合评分：优先面积大且白色纯度高的区域
        float score = static_cast<float>(br.area()) * white_ratio;
        if (score > best_score)
        {
            best_score = score;
            best = cv::Rect2f(static_cast<float>(br.x),
                              static_cast<float>(br.y),
                              static_cast<float>(br.width),
                              static_cast<float>(br.height));
        }
    }

    if (best_score < 0.f)
        ROS_WARN("find_math_proble: no white math region found in image");
    else
        ROS_DEBUG("find_math_proble: roi=[%.0f, %.0f, %.0f x %.0f]",
                  best.x, best.y, best.width, best.height);

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
        ROS_ERROR("init_fisheye_undistort: invalid size %dx%d",
                  image_width, image_height);
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
        ROS_INFO("Fisheye undistort initialized for %dx%d",
                 image_width, image_height);
    else
        ROS_ERROR("Fisheye undistort init FAILED for %dx%d",
                  image_width, image_height);

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
