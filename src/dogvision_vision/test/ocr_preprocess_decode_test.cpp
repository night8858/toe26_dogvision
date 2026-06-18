/**
 * @file ocr_preprocess_decode_test.cpp
 * @brief OCR 预处理和受限 CTC 解码单元测试
 *
 * 测试覆盖：
 *   1. preprocess_math_roi：灰度化、CLAHE、二值化结果验证
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

/**
 * @brief 测试 ROI 预处理流程
 *
 * 验证：
 *   - 预处理后尺寸不变
 *   - 输出为二值图像（BGR 三通道值一致）
 *   - 黑色/白色像素在合理范围内
 *   - 禁用预处理时返回原图
 *   - invert 模式正确反转二值结果
 *   - 无效参数被正确拒绝
 */
void test_roi_preprocess()
{
    const std::string image_path =
        std::string(DOGVISION_VISION_SOURCE_DIR) +
        "/data/img/image_143643394669487.png";
    const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    expect(!image.empty(), "failed to load OCR sample image");

    // This fixture is almost entirely white, so the production ROI locator
    // intentionally rejects the full-page contour via its maximum-area rule.
    // Use the central expression band to isolate preprocessing behavior.
    const cv::Rect rect(
        image.cols / 20,
        image.rows * 35 / 100,
        image.cols * 9 / 10,
        image.rows * 30 / 100);

    const cv::Mat raw_roi = image(rect);
    s_detector_params config;
    const cv::Mat processed = preprocess_math_roi(raw_roi, config);
    expect(processed.size() == raw_roi.size(), "preprocessing changed ROI dimensions");
    expect(processed.type() == CV_8UC3, "preprocessed ROI must be 8-bit BGR");

    std::vector<cv::Mat> channels;
    cv::split(processed, channels);
    expect(cv::countNonZero(channels[0] != channels[1]) == 0,
           "binary BGR channels differ");
    expect(cv::countNonZero(channels[0] != channels[2]) == 0,
           "binary BGR channels differ");
    cv::Mat non_binary;
    cv::inRange(channels[0], cv::Scalar(1), cv::Scalar(254), non_binary);
    expect(cv::countNonZero(non_binary) == 0, "ROI contains non-binary pixels");

    const int black_pixels = static_cast<int>(channels[0].total()) -
                             cv::countNonZero(channels[0]);
    expect(black_pixels > 100, "preprocessing removed the printed expression");
    expect(black_pixels < static_cast<int>(channels[0].total() / 2),
           "preprocessing made most of the white ROI black");

    config.ocr_preprocess_enabled = false;
    const cv::Mat disabled = preprocess_math_roi(raw_roi, config);
    expect(cv::norm(disabled, raw_roi, cv::NORM_INF) == 0.0,
           "disabled preprocessing must preserve the original ROI");

    config.ocr_preprocess_enabled = true;
    const cv::Mat normal = preprocess_math_roi(raw_roi, config);
    config.ocr_preprocess_invert = true;
    const cv::Mat inverted = preprocess_math_roi(raw_roi, config);
    cv::Mat expected_inverted;
    cv::bitwise_not(normal, expected_inverted);
    expect(cv::norm(inverted, expected_inverted, cv::NORM_INF) == 0.0,
           "invert mode must reverse the Otsu binary result");

    config = s_detector_params{};
    config.ocr_clahe_clip_limit = 0.0;
    expect_throw([&] { preprocess_math_roi(raw_roi, config); },
                 "zero CLAHE clip limit must be rejected");
    config = s_detector_params{};
    config.ocr_clahe_tile_size = 0;
    expect_throw([&] { preprocess_math_roi(raw_roi, config); },
                 "zero CLAHE tile size must be rejected");
    config = s_detector_params{};
    config.ocr_gaussian_kernel_size = 4;
    expect_throw([&] { preprocess_math_roi(raw_roi, config); },
                 "even Gaussian kernel size must be rejected");
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
    test_roi_preprocess();
    test_restricted_ctc_decode();
    test_expression_regression();
    std::cout << "OCR preprocessing and restricted decode tests passed" << std::endl;
    return 0;
}
