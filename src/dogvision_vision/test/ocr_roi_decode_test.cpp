/**
 * @file ocr_roi_decode_test.cpp
 * @brief OCR ROI 和受限 CTC 解码单元测试
 *
 * 测试覆盖：
 *   1. OCR ROI 扩张与可选灰度化
 *   2. detect_rec_ppocr::Decode：数学字符白名单过滤的 CTC 解码
 *   3. parse_simple_expr：算术表达式解析正确性
 */

#include <dogvision_vision/ocr_detect.hpp>
#include <dogvision_vision/ocr_utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
/**
 * @brief 断言条件为真，否则打印失败信息并退出
 */
void expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << std::endl;
        std::exit(1);
    }
}

/**
 * @brief 断言函数抛出 std::exception，否则退出
 */
template <typename Function>
void expect_throw(Function&& function, const char* message)
{
    try
    {
        function();
    }
    catch (const std::exception&)
    {
        return;
    }
    expect(false, message);
}

/**
 * @brief 创建测试用临时文件
 */
void write_file(const std::string& path, const std::string& content)
{
    std::ofstream out(path);
    expect(out.is_open(), "failed to create temporary test file");
    out << content;
}

void test_roi_expansion_and_grayscale()
{
    const cv::Rect expanded =
        expand_ocr_roi(cv::Rect2f(100.f, 50.f, 200.f, 100.f),
                       cv::Size(640, 480), 0.05);
    expect(expanded == cv::Rect(90, 45, 220, 110),
           "5 percent ROI expansion is incorrect");

    const cv::Rect clipped =
        expand_ocr_roi(cv::Rect2f(2.f, 3.f, 100.f, 80.f),
                       cv::Size(120, 90), 0.10);
    expect(clipped == cv::Rect(0, 0, 112, 90),
           "expanded ROI was not clipped to image bounds");

    const cv::Rect unchanged =
        expand_ocr_roi(cv::Rect2f(10.f, 20.f, 30.f, 40.f),
                       cv::Size(100, 100), 0.0);
    expect(unchanged == cv::Rect(10, 20, 30, 40),
           "zero expansion must preserve the ROI");
    expect_throw(
        [&] {
            (void)expand_ocr_roi(
                cv::Rect2f(10.f, 20.f, 30.f, 40.f),
                cv::Size(100, 100), -0.01);
        },
        "negative expansion ratio must be rejected");

    cv::Mat color(3, 4, CV_8UC3);
    for (int y = 0; y < color.rows; ++y)
    {
        for (int x = 0; x < color.cols; ++x)
        {
            color.at<cv::Vec3b>(y, x) =
                cv::Vec3b(10 + x, 40 + y, 90 + x + y);
        }
    }

    const cv::Mat passthrough = prepare_ocr_roi(color, false);
    expect(passthrough.size() == color.size(), "color ROI dimensions changed");
    expect(passthrough.type() == CV_8UC3, "color ROI must remain BGR");
    expect(cv::norm(passthrough, color, cv::NORM_INF) == 0.0,
           "grayscale disabled must preserve color pixels");

    const cv::Mat grayscale = prepare_ocr_roi(color, true);
    expect(grayscale.size() == color.size(), "grayscale ROI dimensions changed");
    expect(grayscale.type() == CV_8UC3, "grayscale ROI must be three-channel BGR");
    std::vector<cv::Mat> channels;
    cv::split(grayscale, channels);
    expect(cv::countNonZero(channels[0] != channels[1]) == 0,
           "grayscale BGR channels differ");
    expect(cv::countNonZero(channels[0] != channels[2]) == 0,
           "grayscale BGR channels differ");
}

/**
 * @brief 测试受限 CTC 解码
 *
 * 创建一个模拟的 logits 张量，其中包含白名单内外的字符，
 * 验证 Decode 方法能正确：
 *   - 只输出白名单内的字符
 *   - 合并连续重复的时序帧
 *   - 去除 CTC blank 帧
 *   - 计算正确的平均置信度
 *   - 拒绝字典大小不匹配的张量
 *   - 拒绝白名单中包含字典中没有的字符
 */
void test_restricted_ctc_decode()
{
    const std::string dict_path = "/tmp/dogvision_test_ppocr_dict.txt";
    const std::string whitelist_path = "/tmp/dogvision_test_ppocr_allowed.txt";
    const std::string missing_path = "/tmp/dogvision_test_ppocr_missing.txt";
    write_file(dict_path, "0\n1\n+\nA\n×\n");
    write_file(whitelist_path, "0\n1\n+\n×\n");
    write_file(missing_path, "÷\n");

    detect_rec_ppocr recognizer(nullptr);
    recognizer.loda_dict(dict_path);
    recognizer.load_allowed_chars(whitelist_path);

    // Classes: blank, 0, 1, +, A, ×, appended space.
    ov::Tensor logits(ov::element::f32, {1, 8, 7});
    float* values = logits.data<float>();
    std::fill(values, values + logits.get_size(), -10.0f);
    auto set_score = [&](size_t time, size_t cls, float score)
    {
        values[time * 7 + cls] = score;
    };

    set_score(0, 4, 0.99f); // Disallowed A is globally highest.
    set_score(0, 2, 0.80f); // Restricted decode must choose 1.
    set_score(1, 2, 0.90f); // Repeated 1 must collapse.
    set_score(2, 0, 0.95f);
    set_score(3, 3, 0.85f);
    set_score(4, 4, 0.98f); // Disallowed A loses to allowed 0.
    set_score(4, 1, 0.75f);
    set_score(5, 5, 0.88f);
    set_score(6, 0, 0.92f);
    set_score(7, 2, 0.82f);

    const auto result = recognizer.Decode(logits);
    expect(result.size() == 1, "unexpected CTC batch size");
    expect(result[0].text == "1+0×1", "restricted CTC decode returned wrong text");
    expect(result[0].text.find('A') == std::string::npos,
           "restricted CTC decode emitted a disallowed character");
    expect(result[0].score > 0.0f && result[0].score <= 1.0f,
           "restricted CTC score is outside expected range");

    ov::Tensor wrong_shape(ov::element::f32, {1, 1, 6});
    expect_throw([&] { recognizer.Decode(wrong_shape); },
                 "dictionary/logit class mismatch must be rejected");

    detect_rec_ppocr invalid_recognizer(nullptr);
    invalid_recognizer.loda_dict(dict_path);
    expect_throw([&] { invalid_recognizer.load_allowed_chars(missing_path); },
                 "whitelist characters absent from dictionary must be rejected");

    detect_rec_ppocr repository_recognizer(nullptr);
    const std::string source_dir = DOGVISION_VISION_SOURCE_DIR;
    repository_recognizer.loda_dict(
        source_dir + "/models/ppocr/Dict/ppocr_keys_v1.txt");
    repository_recognizer.load_allowed_chars(
        source_dir + "/models/ppocr/Dict/math_chars.txt");

    std::remove(dict_path.c_str());
    std::remove(whitelist_path.c_str());
    std::remove(missing_path.c_str());
}

/**
 * @brief 测试算术表达式解析的回归正确性
 *
 * 验证 parse_simple_expr 能正确解析含括号和混合运算符的表达式，
 * 并返回正确的计算结果。
 * 用例：(12+36)×5÷(18-9) = 240/9
 */
void test_expression_regression()
{
    double result = 0.0;
    std::string expression;
    expect(parse_simple_expr("(12+36)×5÷(18-9)=", result, expression),
           "failed to parse sample arithmetic expression");
    expect(expression == "(12+36)*5/(18-9)",
           "sample expression normalization changed unexpectedly");
    expect(std::abs(result - (240.0 / 9.0)) < 1e-6,
           "sample expression result is incorrect");
}
} // namespace

/**
 * @brief 测试入口：依次执行所有测试用例
 */
int main()
{
    test_roi_expansion_and_grayscale();
    test_restricted_ctc_decode();
    test_expression_regression();
    std::cout << "OCR ROI and restricted decode tests passed" << std::endl;
    return 0;
}
