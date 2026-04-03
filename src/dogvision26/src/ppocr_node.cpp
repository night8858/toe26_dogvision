#include <ros/ros.h>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common_structs.h"
#include "detector.hpp"
#include "ocr_detect.hpp"

namespace fs = std::filesystem;

// 透视变换裁剪文本区域（4点框 → 矩形，竖排文本自动旋转90°使其横排）
static cv::Mat crop_text_region(const cv::Mat &src, const OCRBox &box)
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

    // 竖排文本（高 > 宽 * 1.5）旋转90°
    if (crop.rows > static_cast<int>(crop.cols * 1.5f))
        cv::rotate(crop, crop, cv::ROTATE_90_CLOCKWISE);

    return crop;
}

// ── 算术解析与求值 ─────────────────────────────────────────────────────────
// 将 OCR 识别的所有文本拼接，提取第一个完整算术表达式并求值
// 支持符号：+ - * / × ÷ 以及中文全角字符
// 返回 true 表示解析成功，result 为整数结果（截断），expr_str 填充找到的原始表达式

static std::string normalize_expr(const std::string &src)
{
    // 替换常见 OCR 误识别和中文运算符
    std::string s = src;
    const std::pair<std::string, std::string> repl[] = {
        {"×", "*"}, {"÷", "/"}, {"＋", "+"}, {"－", "-"},
        // 全角括号 → 半角
        {"（", "("}, {"）", ")"},
        // 常见 OCR 把乘号识别成字母
        {"X", "*"}, {"x", "*"},
        // 去掉无意义字符
        {" ", ""}, {"=", ""}, {"?", ""}, {"？", ""},
    };
    for (const auto &p : repl) {
        size_t pos = 0;
        while ((pos = s.find(p.first, pos)) != std::string::npos) {
            s.replace(pos, p.first.size(), p.second);
            pos += p.second.size();
        }
    }
    return s;
}

// ── 递归下降解析器（支持括号、标准运算优先级）─────────────────────────────
// 语法：
//   expr   = term  { ('+' | '-') term  }
//   term   = factor{ ('*' | '/') factor}
//   factor = '(' expr ')' | ['-'] number

struct Parser {
    const std::string &s;
    size_t pos;

    explicit Parser(const std::string &str) : s(str), pos(0) {}

    void skip_ws() { while (pos < s.size() && s[pos] == ' ') ++pos; }

    bool at_end() { skip_ws(); return pos >= s.size(); }

    char peek() { skip_ws(); return pos < s.size() ? s[pos] : '\0'; }

    char consume() { skip_ws(); return pos < s.size() ? s[pos++] : '\0'; }

    // 读取数字（整数或小数，不含前缀负号，由 factor 处理）
    bool read_number(double &val) {
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

        // 括号子表达式
        if (s[pos] == '(') {
            ++pos;
            double val = expr();
            skip_ws();
            if (pos < s.size() && s[pos] == ')') ++pos;
            return val;
        }

        // 一元负号
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
            if (c == '*' || c == '/') {
                consume();
                double rhs = factor();
                val = (c == '*') ? val * rhs : val / rhs;
            } else {
                break;
            }
        }
        return val;
    }

    double expr() {
        double val = term();
        while (true) {
            char c = peek();
            if (c == '+' || c == '-') {
                consume();
                double rhs = term();
                val = (c == '+') ? val + rhs : val - rhs;
            } else {
                break;
            }
        }
        return val;
    }
};

static bool parse_simple_expr(const std::string &text, double &result, std::string &expr_str)
{
    std::string norm = normalize_expr(text);

    // 从规范化字符串中找第一个包含运算符的子串（允许括号）
    // 扫描：找到第一个数字/负号/左括号开始，到最后一个合法字符结束
    // 简单策略：找到第一个运算符，向前/后扩展到完整表达式
    // 使用正则定位表达式的起始位置（首个数字或左括号）
    std::regex start_pat(R"([(\-]?\d|[(])");
    std::smatch sm;
    if (!std::regex_search(norm, sm, start_pat))
        return false;

    // 取从匹配位置到字符串末尾，解析器会自动在遇到非法字符时停止
    std::string candidate = norm.substr(sm.position());

    // 检查是否含有运算符（不只是一个数）
    if (candidate.find_first_of("+-*/") == std::string::npos)
        return false;

    Parser parser(candidate);
    double val = parser.expr();

    // expr_str 为实际消费的部分
    expr_str = candidate.substr(0, parser.pos);

    // 必须至少消费了一个运算符
    if (expr_str.find_first_of("+-*/") == std::string::npos)
        return false;

    result = val;
    return true;
}

// 创建结果显示图像
static void show_result_window(const std::string &expr_str, int mod_result)
{
    cv::Mat canvas(200, 500, CV_8UC3, cv::Scalar(30, 30, 30));

    std::string line1 = "Expr: " + expr_str;
    std::string line2 = "Result % 4 = " + std::to_string(mod_result);

    cv::putText(canvas, line1,
                cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX,
                0.9, cv::Scalar(0, 230, 0), 2, cv::LINE_AA);
    cv::putText(canvas, line2,
                cv::Point(20, 140), cv::FONT_HERSHEY_SIMPLEX,
                1.1, cv::Scalar(0, 200, 255), 2, cv::LINE_AA);

    cv::imshow("OCR Arithmetic Result", canvas);
    cv::waitKey(0);
    cv::destroyWindow("OCR Arithmetic Result");
}

// 在图像上绘制4点检测框和识别文本
static void draw_ocr_result(cv::Mat &vis, const OCRBox &box, const OCRRecResult &rec)
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

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ppocr_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string image_path;
    std::string output_dir;
    std::string config_path;

    pnh.param<std::string>("image_path",  image_path,  "/home/toe/toe26_dogvision/src/dogvision26/src/data/img/image_143643394669487.png");
    pnh.param<std::string>("output_dir",  output_dir,  "/home/toe/toe26_dogvision/src/dogvision26/src/data/ocr_output");
    pnh.param<std::string>("config_path", config_path,
        "/home/toe/toe26_dogvision/src/dogvision26/src/settings.json");

    if (image_path.empty())
    {
        ROS_ERROR("Missing required param ~image_path");
        return -1;
    }
    if (output_dir.empty())
    {
        ROS_ERROR("Missing required param ~output_dir");
        return -1;
    }

    // ── 1. 加载配置 ──────────────────────────────────────────────────────────
    Appconfig config;
    {
        detect_det_ppocr loader(nullptr);
        loader.load_config(config, config_path);
    }
    ROS_INFO_STREAM("Config      : " << config_path);
    ROS_INFO_STREAM("Det model   : " << config.detect_config.ppocr_det_model_path);
    ROS_INFO_STREAM("Rec model   : " << config.detect_config.ppocr_rec_model_path);
    ROS_INFO_STREAM("Dict        : " << config.detect_config.rec_char_dict_path);

    // ── 2. 初始化 det 检测模型 ───────────────────────────────────────────────
    detect_det_ppocr det(&config);
    det.load_model(config.detect_config.ppocr_det_model_path,
                   config.detect_config.det_device);

    // ── 3. 初始化 rec 识别模型 ───────────────────────────────────────────────
    detect_rec_ppocr rec(&config);
    rec.load_model(config.detect_config.ppocr_rec_model_path,
                   config.detect_config.rec_device);
    rec.loda_dict(config.detect_config.rec_char_dict_path);

    // max_wh_ratio 固定为配置中的 rec_img_w / rec_img_h（默认 320/48≈6.67）
    // 超出此比例的裁剪图会被缩放到最大宽度，无需动态 reshape 模型
    const float default_wh_ratio =
        (config.detect_config.rec_img_h > 0)
        ? static_cast<float>(config.detect_config.rec_img_w) /
          static_cast<float>(config.detect_config.rec_img_h)
        : 320.0f / 48.0f;
    rec.set_max_wh_ratio(default_wh_ratio);

    // ── 4. 读取输入图像 ──────────────────────────────────────────────────────
    cv::Mat img = cv::imread(image_path);
    if (img.empty())
    {
        ROS_ERROR_STREAM("Cannot read image: " << image_path);
        return -1;
    }
    ROS_INFO_STREAM("Image       : " << image_path
                    << " [" << img.cols << "x" << img.rows << "]");

    // ── 5. 文本检测（det）───────────────────────────────────────────────────
    det.preprocess(img);
    det.inference();
    det.postprocess();
    const std::vector<OCRBox> &boxes = det.ocr_det_out_;
    ROS_INFO_STREAM("Detected " << boxes.size() << " text region(s)");

    // ── 6. 裁剪 + 文本识别（rec）────────────────────────────────────────────
    std::vector<OCRItem> ocr_items;
    cv::Mat vis = img.clone();

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        cv::Mat crop = crop_text_region(img, boxes[i]);
        if (crop.empty())
            continue;

        rec.preprocess(crop);
        rec.inference();
        rec.postprocess();

        if (!rec.result.empty() && !rec.result[0].text.empty())
        {
            OCRItem item;
            item.box = boxes[i];
            item.rec = rec.result[0];
            ocr_items.push_back(item);
            draw_ocr_result(vis, boxes[i], rec.result[0]);
            ROS_INFO_STREAM("  [" << i << "] \""
                            << rec.result[0].text
                            << "\"  score=" << rec.result[0].score);
        }
    }

    // ── 6b. 算术识别：汇总所有 OCR 文本，解析并计算 ─────────────────────────
    {
        std::string all_text;
        for (const auto &item : ocr_items)
            all_text += item.rec.text + " ";

        ROS_INFO_STREAM("All OCR text: \"" << all_text << "\"");

        double calc_result = 0.0;
        std::string expr_str;
        if (parse_simple_expr(all_text, calc_result, expr_str)) {
            int int_result = static_cast<int>(std::round(calc_result));
            int mod_result = ((int_result % 4) + 4) % 4;  // 保证非负
            ROS_INFO("Expr: %s  =>  %d  %%4 = %d",
                     expr_str.c_str(), int_result, mod_result);
            show_result_window(expr_str, mod_result);
        } else {
            ROS_WARN("No arithmetic expression found in OCR output.");
        }
    }

    // ── 7. 保存结果到指定目录 ────────────────────────────────────────────────
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec)
    {
        ROS_ERROR_STREAM("Cannot create output dir: "
                         << output_dir << " (" << ec.message() << ")");
        return -1;
    }

    const std::string out_img_path = output_dir + "/result.jpg";
    const std::string out_txt_path = output_dir + "/result.txt";

    if (!cv::imwrite(out_img_path, vis))
        ROS_WARN_STREAM("Failed to write image: " << out_img_path);
    else
        ROS_INFO_STREAM("Result image: " << out_img_path);

    {
        std::ofstream ofs(out_txt_path);
        if (!ofs.is_open())
        {
            ROS_WARN_STREAM("Failed to write text: " << out_txt_path);
        }
        else
        {
            for (size_t i = 0; i < ocr_items.size(); ++i)
            {
                ofs << "[" << i << "]\t"
                    << ocr_items[i].rec.text << "\t"
                    << std::fixed << std::setprecision(4)
                    << ocr_items[i].rec.score << "\n";
            }
            ROS_INFO_STREAM("Result text : " << out_txt_path
                            << " (" << ocr_items.size() << " items)");
        }
    }

    return 0;
}

