/**
 * @file ocr_math_filter_decode_test.cpp
 * @brief 全帧 OCR 数学题后置筛选和受限 CTC 解码测试
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
void expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << std::endl;
        std::exit(1);
    }
}

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

void write_file(const std::string& path, const std::string& content)
{
    std::ofstream out(path);
    expect(out.is_open(), "failed to create temporary test file");
    out << content;
}

OCRItem make_item(
    const std::string& text,
    const cv::Rect& bounds,
    float score = 0.9f)
{
    OCRItem item;
    item.box.pts = {
        cv::Point2f(static_cast<float>(bounds.x),
                    static_cast<float>(bounds.y)),
        cv::Point2f(static_cast<float>(bounds.x + bounds.width),
                    static_cast<float>(bounds.y)),
        cv::Point2f(static_cast<float>(bounds.x + bounds.width),
                    static_cast<float>(bounds.y + bounds.height)),
        cv::Point2f(static_cast<float>(bounds.x),
                    static_cast<float>(bounds.y + bounds.height))};
    item.rec.text = text;
    item.rec.score = score;
    return item;
}

s_detector_params make_filter_config()
{
    s_detector_params config;
    config.ocr_math_min_surround_white_ratio = 0.50;
    config.ocr_math_surround_margin_ratio = 0.50;
    config.ocr_math_white_s_max = 110;
    config.ocr_math_white_v_min = 50;
    return config;
}

void test_full_frame_grayscale()
{
    cv::Mat color(3, 4, CV_8UC3);
    for (int y = 0; y < color.rows; ++y)
    {
        for (int x = 0; x < color.cols; ++x)
        {
            color.at<cv::Vec3b>(y, x) =
                cv::Vec3b(10 + x, 40 + y, 90 + x + y);
        }
    }

    const cv::Mat passthrough = prepare_ocr_input(color, false);
    expect(passthrough.size() == color.size(), "full-frame color size changed");
    expect(passthrough.type() == CV_8UC3, "full-frame color must remain BGR");
    expect(cv::norm(passthrough, color, cv::NORM_INF) == 0.0,
           "disabled grayscale must preserve full-frame pixels");

    const cv::Mat grayscale = prepare_ocr_input(color, true);
    expect(grayscale.size() == color.size(), "full-frame grayscale size changed");
    expect(grayscale.type() == CV_8UC3, "grayscale input must be three-channel");
    std::vector<cv::Mat> channels;
    cv::split(grayscale, channels);
    expect(cv::countNonZero(channels[0] != channels[1]) == 0,
           "grayscale channels differ");
    expect(cv::countNonZero(channels[0] != channels[2]) == 0,
           "grayscale channels differ");
}

void test_strict_expression_parser()
{
    double result = 0.0;
    std::string expression;
    expect(parse_simple_expr("(12+36)×5÷(18-9)=", result, expression),
           "valid expression was rejected");
    expect(expression == "(12+36)*5/(18-9)",
           "expression normalization is incorrect");
    expect(std::abs(result - (240.0 / 9.0)) < 1e-6,
           "expression result is incorrect");

    expect(!parse_simple_expr("123", result, expression),
           "number without binary operator was accepted");
    expect(!parse_simple_expr("(1+2", result, expression),
           "unclosed parenthesis was accepted");
    expect(!parse_simple_expr("1+", result, expression),
           "incomplete expression was accepted");
    expect(!parse_simple_expr("1/0", result, expression),
           "division by zero was accepted");
    expect(!parse_simple_expr("1..2+3", result, expression),
           "malformed decimal was accepted");
    expect(!parse_simple_expr("A1+2", result, expression),
           "unrelated character was accepted");
}

void test_surround_white_ratio_and_edge_clipping()
{
    const s_detector_params config = make_filter_config();
    cv::Mat white(120, 180, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Rect surround;
    const float white_ratio = calculate_surround_white_ratio(
        white, cv::Rect(0, 20, 70, 30), 30.0f, config, &surround);
    expect(white_ratio > 0.99f, "white surround ratio should be near one");
    expect(surround.x == 0, "edge surround was not clipped");

    cv::Mat dark(120, 180, CV_8UC3, cv::Scalar(20, 20, 20));
    const float dark_ratio = calculate_surround_white_ratio(
        dark, cv::Rect(30, 20, 70, 30), 30.0f, config);
    expect(dark_ratio < 0.01f, "dark surround ratio should be near zero");
}

void test_candidate_grouping_and_selection()
{
    const s_detector_params config = make_filter_config();
    cv::Mat image(220, 360, CV_8UC3, cv::Scalar(255, 255, 255));

    std::vector<OCRItem> items{
        make_item("12", cv::Rect(230, 40, 42, 30), 0.90f),
        make_item("+", cv::Rect(276, 40, 20, 30), 0.88f),
        make_item("3", cv::Rect(300, 40, 20, 30), 0.92f),
        make_item("999", cv::Rect(20, 150, 45, 28), 0.99f)};

    const std::vector<OCRMathCandidate> candidates =
        find_math_candidates(image, items, config);
    expect(!candidates.empty(), "edge arithmetic candidate was not found");
    expect(candidates.front().expression == "12+3",
           "split OCR boxes were not combined in reading order");
    expect(candidates.front().white_pass,
           "white-surrounded edge candidate was rejected");
    expect(candidates.front().text_bounds.x >= 200,
           "candidate unexpectedly depends on image center");

    cv::Mat mixed(220, 360, CV_8UC3, cv::Scalar(20, 20, 20));
    cv::rectangle(mixed, cv::Rect(180, 0, 180, 110),
                  cv::Scalar(255, 255, 255), cv::FILLED);
    std::vector<OCRItem> multiple{
        make_item("1+2", cv::Rect(30, 40, 70, 30), 0.99f),
        make_item("8*9", cv::Rect(230, 40, 70, 30), 0.80f)};
    const std::vector<OCRMathCandidate> ranked =
        find_math_candidates(mixed, multiple, config);
    expect(ranked.size() >= 2, "multiple arithmetic candidates were not built");
    expect(ranked.front().expression == "8*9" && ranked.front().white_pass,
           "accepted white candidate was not ranked before rejected candidate");
}

void test_restricted_ctc_decode()
{
    const std::string dict_path = "/tmp/dogvision_test_ppocr_dict.txt";
    const std::string yaml_dict_path = "/tmp/dogvision_test_ppocr_dict.yml";
    const std::string whitelist_path = "/tmp/dogvision_test_ppocr_allowed.txt";
    const std::string yaml_whitelist_path = "/tmp/dogvision_test_ppocr_yaml_allowed.txt";
    const std::string missing_path = "/tmp/dogvision_test_ppocr_missing.txt";
    write_file(dict_path, "0\n1\n+\nA\n×\n");
    write_file(yaml_dict_path,
               "PostProcess:\n"
               "  name: CTCLabelDecode\n"
               "  character_dict:\n"
               "  - '0'\n"
               "  - '1'\n"
               "  - +\n"
               "  - '*'\n"
               "  - ×\n");
    write_file(whitelist_path, "0\n1\n+\n×\n");
    write_file(yaml_whitelist_path, "0\n*\n×\n");
    write_file(missing_path, "÷\n");

    detect_rec_ppocr recognizer(nullptr);
    recognizer.loda_dict(dict_path);
    recognizer.load_allowed_chars(whitelist_path);

    ov::Tensor logits(ov::element::f32, {1, 8, 7});
    float* values = logits.data<float>();
    std::fill(values, values + logits.get_size(), -10.0f);
    auto set_score = [&](size_t time, size_t cls, float score)
    {
        values[time * 7 + cls] = score;
    };

    set_score(0, 4, 0.99f);
    set_score(0, 2, 0.80f);
    set_score(1, 2, 0.90f);
    set_score(2, 0, 0.95f);
    set_score(3, 3, 0.85f);
    set_score(4, 4, 0.98f);
    set_score(4, 1, 0.75f);
    set_score(5, 5, 0.88f);
    set_score(6, 0, 0.92f);
    set_score(7, 2, 0.82f);

    const auto result = recognizer.Decode(logits);
    expect(result.size() == 1, "unexpected CTC batch size");
    expect(result[0].text == "1+0×1", "restricted CTC decode returned wrong text");
    expect(result[0].text.find('A') == std::string::npos,
           "restricted CTC decode emitted a disallowed character");

    ov::Tensor wrong_shape(ov::element::f32, {1, 1, 6});
    expect_throw([&] { recognizer.Decode(wrong_shape); },
                 "dictionary/logit class mismatch must be rejected");

    detect_rec_ppocr invalid_recognizer(nullptr);
    invalid_recognizer.loda_dict(dict_path);
    expect_throw([&] { invalid_recognizer.load_allowed_chars(missing_path); },
                 "missing whitelist character must be rejected");

    detect_rec_ppocr yaml_recognizer(nullptr);
    yaml_recognizer.loda_dict(yaml_dict_path);
    yaml_recognizer.load_allowed_chars(yaml_whitelist_path);
    ov::Tensor yaml_logits(ov::element::f32, {1, 4, 7});
    float* yaml_values = yaml_logits.data<float>();
    std::fill(yaml_values, yaml_values + yaml_logits.get_size(), -10.0f);
    yaml_values[1] = 0.95f;
    yaml_values[7 + 4] = 0.96f;
    yaml_values[14] = 0.97f;
    yaml_values[21 + 5] = 0.98f;
    const auto yaml_result = yaml_recognizer.Decode(yaml_logits);
    expect(yaml_result[0].text == "0*×",
           "PaddleOCR inference.yml dictionary was not decoded correctly");

    detect_rec_ppocr repository_recognizer(nullptr);
    const std::string source_dir = DOGVISION_VISION_SOURCE_DIR;
    repository_recognizer.loda_dict(
        source_dir + "/models/ppocr/Dict/ppocr_keys_v1.txt");
    repository_recognizer.load_allowed_chars(
        source_dir + "/models/ppocr/Dict/math_chars.txt");

    std::remove(dict_path.c_str());
    std::remove(yaml_dict_path.c_str());
    std::remove(whitelist_path.c_str());
    std::remove(yaml_whitelist_path.c_str());
    std::remove(missing_path.c_str());
}
} // namespace

int main()
{
    test_full_frame_grayscale();
    test_strict_expression_parser();
    test_surround_white_ratio_and_edge_clipping();
    test_candidate_grouping_and_selection();
    test_restricted_ctc_decode();
    std::cout << "OCR math filter and restricted decode tests passed"
              << std::endl;
    return 0;
}
