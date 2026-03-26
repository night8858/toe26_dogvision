#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp> // OpenVINO 2025 API
#include <vector>
#include <algorithm>
#include <fstream>

#include "common_structs.h"
#include "detector.hpp"
#include "ocr_detect.hpp"

void detect_det_ppocr::load_model(const std::string& model_path, const std::string& device)
{
    model_ = core_.compile_model(model_path, device);
    infer_request_ = model_.create_infer_request();
    // input_tensor_ = infer_request_.get_input_tensor();
    // output_tensor_ = infer_request_.get_output_tensor();
    // 这里实现PP-OCR的模型加载逻辑，使用OpenVINO加载模型并创建推理请求
    // 根据配置选择设备（CPU/GPU/VPU等）加载模型
}


void detect_det_ppocr::preprocess(cv::Mat &input_img)
{
    Mate.src_h = input_img.rows;
    Mate.src_w = input_img.cols;

    float ratio = 1.0f;
    // 根据配置选择缩放策略，计算缩放比例
    if (detect_config_.det_limit_type == "max")
    {
        const int max_side = std::max(Mate.src_h, Mate.src_w);
        if (max_side > detect_config_.det_limit_side_len)
        {
            ratio = static_cast<float>(detect_config_.det_limit_side_len) / static_cast<float>(max_side);
        }
    }
    else
    {
        const int min_side = std::min(Mate.src_h, Mate.src_w);
        if (min_side < detect_config_.det_limit_side_len)
        {
            ratio = static_cast<float>(detect_config_.det_limit_side_len) / static_cast<float>(min_side);
        }
    }

    int resize_h = std::max(32, static_cast<int>(std::round(Mate.src_h * ratio / 32.0f) * 32.0f));
    int resize_w = std::max(32, static_cast<int>(std::round(Mate.src_w * ratio / 32.0f) * 32.0f));

    Mate.resize_h = resize_h;
    Mate.resize_w = resize_w;
    Mate.ratio_h = static_cast<float>(resize_h) / static_cast<float>(Mate.src_h);
    Mate.ratio_w = static_cast<float>(resize_w) / static_cast<float>(Mate.src_w);

    cv::Mat resized;
    cv::resize(input_img, resized, cv::Size(resize_w, resize_h));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    // 图像归一化标准化
    const cv::Scalar mean(0.485, 0.456, 0.406);
    const cv::Scalar std(0.229, 0.224, 0.225);
    // 把每个通道的值减去对应通道的平均值
    cv::subtract(resized, mean, resized);
    // 把每个通道的值除以对应通道的标准差
    cv::divide(resized, std, resized);

    ov::Tensor input(ov::element::f32, {1, 3, static_cast<size_t>(resize_h), static_cast<size_t>(resize_w)});
    float *data = input.data<float>();

    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c)
    {
        channels[c] = cv::Mat(resize_h, resize_w, CV_32FC1, data + c * resize_h * resize_w);
    }

    cv::split(resized, channels);

    input_tensor_ = input;
    // 这里实现PP-OCR的预处理逻辑，将输入图像转换为模型输入张量
    // 包括文本区域的裁剪、缩放、归一化等操作
    // 返回预处理后的张量供后续推理使用
}

void detect_det_ppocr::inference()
{
    infer_request_.set_input_tensor(input_tensor_);
    infer_request_.infer();
    output_tensor_ = infer_request_.get_output_tensor();
    // 这里实现PP-OCR的推理逻辑，使用OpenVINO进行模型推理
    // 将预处理后的输入张量传递给模型，并获取输出张量
}

std::array<cv::Point2f, 4> detect_det_ppocr::OrderPointsClockwise(const std::vector<cv::Point2f> &pts) const
{
    std::array<cv::Point2f, 4> rect;
    std::vector<float> s(4), d(4);
    for (int i = 0; i < 4; ++i)
    {
        s[i] = pts[i].x + pts[i].y;
        d[i] = pts[i].y - pts[i].x;
    }
    rect[0] = pts[static_cast<size_t>(std::distance(s.begin(), std::min_element(s.begin(), s.end())))];
    rect[2] = pts[static_cast<size_t>(std::distance(s.begin(), std::max_element(s.begin(), s.end())))];
    rect[1] = pts[static_cast<size_t>(std::distance(d.begin(), std::min_element(d.begin(), d.end())))];
    rect[3] = pts[static_cast<size_t>(std::distance(d.begin(), std::max_element(d.begin(), d.end())))];
    return rect;
}

void detect_det_ppocr::postprocess()
{

    const auto shape = output_tensor_.get_shape();
    if (shape.size() != 4)
    {
        return;
    }

    const size_t h = shape[2];
    const size_t w = shape[3];
    const float *p = output_tensor_.data<const float>();

    cv::Mat prob_map(static_cast<int>(h), static_cast<int>(w), CV_32FC1);
    std::memcpy(prob_map.data, p, sizeof(float) * h * w);

    cv::Mat bin;

    cv::threshold(prob_map, bin, detect_config_.det_db_thresh, 255, cv::THRESH_BINARY);

    bin.convertTo(bin, CV_8UC1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    //清理cortours，留下符合条件的文本区域轮廓，并将文本区域坐标转换为原图坐标，存储在ocr_det_out_中
    ocr_det_out_.clear();

    for (const auto &contour : contours)
    {
        if (contour.size() < 4)
        {
            continue;
        }

        cv::RotatedRect r = cv::minAreaRect(contour);
        if (std::min(r.size.width, r.size.height) < 3.0f)
        {
            continue;
        }

        cv::Rect bbox = cv::boundingRect(contour);
        bbox &= cv::Rect(0, 0, static_cast<int>(w), static_cast<int>(h));
        if (bbox.empty())
        {
            continue;
        }

        const cv::Scalar mean_score = cv::mean(prob_map(bbox));
        if (mean_score[0] < detect_config_.det_db_box_thresh)
        {
            continue;
        }

        cv::Point2f pts_arr[4];
        r.points(pts_arr);
        std::vector<cv::Point2f> pts{pts_arr, pts_arr + 4};
        auto ordered = OrderPointsClockwise(pts);

        OCRBox box;
        for (int i = 0; i < 4; ++i)
        {
            const float x = std::clamp(ordered[i].x / Mate.ratio_w, 0.0f, static_cast<float>(Mate.src_w - 1));
            const float y = std::clamp(ordered[i].y / Mate.ratio_h, 0.0f, static_cast<float>(Mate.src_h - 1));
            box.pts[static_cast<size_t>(i)] = cv::Point2f(x, y);
        }
        ocr_det_out_.push_back(box);
    }

    std::sort(ocr_det_out_.begin(), ocr_det_out_.end(), [](const OCRBox &a, const OCRBox &b)
              {
        if (std::abs(a.pts[0].y - b.pts[0].y) < 10.0f) {
            return a.pts[0].x < b.pts[0].x;
        }
        return a.pts[0].y < b.pts[0].y; });

    // 这里实现PP-OCR的后处理逻辑，将模型输出张量转换为可用的检测结果
    // 包括文本区域的坐标、置信度等信息的提取和处理
    // 返回最终的检测结果供后续使用
}


void detect_rec_ppocr::load_model(const std::string& model_path, const std::string& device)
{
    model_ = core_.compile_model(model_path, device);
    infer_request_ = model_.create_infer_request();

}

//读字典
void detect_rec_ppocr::loda_dict(const std::string& dict_path)
{
    dict_.clear();
    dict_.push_back("blank");
    std::ifstream ifs(dict_path);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        dict_.push_back(line);
    }
    dict_.push_back(" ");
}

void detect_rec_ppocr::preprocess(cv::Mat &input_img)
{
    const int img_c = detect_config_.rec_img_c;
    const int img_h = detect_config_.rec_img_h;
    const int img_w = static_cast<int>(img_h * max_wh_ratio);

    const float ratio = static_cast<float>(input_img.cols) / static_cast<float>(input_img.rows);
    int resized_w = static_cast<int>(std::ceil(img_h * ratio));
    resized_w = std::min(resized_w, img_w);

    cv::Mat resized;
    cv::resize(input_img, resized, cv::Size(resized_w, img_h));
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    cv::Mat chw(img_c, img_h * img_w, CV_32F, cv::Scalar(0));
    std::vector<cv::Mat> channels;
    cv::split(resized, channels);
    for (int c = 0; c < img_c; ++c) {
        cv::Mat dst = chw.row(c).reshape(1, img_h);
        channels[c].copyTo(dst.colRange(0, resized_w));
        dst = (dst - 0.5f) / 0.5f;
    }

    chw_img = chw;

}


void detect_rec_ppocr::inference()
{
    infer_request_.set_input_tensor(input_tensor_);
    infer_request_.infer();
    output_tensor_ = infer_request_.get_output_tensor();
}

void detect_rec_ppocr::postprocess()
{

    result = Decode(output_tensor_);
    // 这里实现PP-OCR的后处理逻辑，将模型输出张量转换为可用的识别结果
    // 包括文本内容的解码、置信度计算等信息的提取和处理
    // 返回最终的识别结果供后续使用
    
}


std::vector<OCRRecResult> detect_rec_ppocr::Decode(const ov::Tensor& logits)
{
    const auto s = output_tensor_.get_shape();
    const size_t batch = s[0];
    const size_t time_step = s[1];
    const size_t cls_num = s[2];
    const float* p = output_tensor_.data<const float>();

    std::vector<OCRRecResult> out(batch);
    for (size_t b = 0; b < batch; ++b) {
        std::string text;
        float score_sum = 0.0f;
        int count = 0;
        int prev_idx = -1;

        for (size_t t = 0; t < time_step; ++t) {
            size_t best = 0;
            float best_score = p[b * time_step * cls_num + t * cls_num];
            for (size_t c = 1; c < cls_num; ++c) {
                const float v = p[b * time_step * cls_num + t * cls_num + c];
                if (v > best_score) {
                    best_score = v;
                    best = c;
                }
            }

            if (static_cast<int>(best) == prev_idx) {
                continue;
            }
            prev_idx = static_cast<int>(best);
            if (best == 0) {
                continue;
            }
            if (best < dict_.size()) {
                text += dict_[best];
            }
            score_sum += best_score;
            ++count;
        }

        out[b].text = text;
        out[b].score = count > 0 ? score_sum / static_cast<float>(count) : 0.0f;
    }

    return out;

}



// std::vector<OCRRecResult> TextRecognizer::Run(const std::vector<cv::Mat>& crops) {
//     std::vector<OCRRecResult> out(crops.size());
//     if (crops.empty()) {
//         return out;
//     }

//     std::vector<int> idx(crops.size());
//     std::iota(idx.begin(), idx.end(), 0);
//     std::sort(idx.begin(), idx.end(), [&crops](int a, int b) {
//         const float ra = static_cast<float>(crops[a].cols) / static_cast<float>(crops[a].rows);
//         const float rb = static_cast<float>(crops[b].cols) / static_cast<float>(crops[b].rows);
//         return ra < rb;
//     });

//     for (size_t beg = 0; beg < crops.size(); beg += static_cast<size_t>(cfg_.rec_batch_num)) {
//         const size_t end = std::min(crops.size(), beg + static_cast<size_t>(cfg_.rec_batch_num));
//         const size_t bs = end - beg;

//         float max_wh_ratio = static_cast<float>(cfg_.rec_img_w) / static_cast<float>(cfg_.rec_img_h);
//         for (size_t i = beg; i < end; ++i) {
//             const cv::Mat& m = crops[static_cast<size_t>(idx[i])];
//             max_wh_ratio = std::max(max_wh_ratio, static_cast<float>(m.cols) / static_cast<float>(m.rows));
//         }

//         const int dyn_w = static_cast<int>(cfg_.rec_img_h * max_wh_ratio);
//         ov::Tensor input(ov::element::f32, {bs, static_cast<size_t>(cfg_.rec_img_c), static_cast<size_t>(cfg_.rec_img_h), static_cast<size_t>(dyn_w)});
//         float* data = input.data<float>();

//         const size_t step = static_cast<size_t>(cfg_.rec_img_c * cfg_.rec_img_h * dyn_w);
//         for (size_t i = 0; i < bs; ++i) {
//             cv::Mat chw = ResizeNorm(crops[static_cast<size_t>(idx[beg + i])], max_wh_ratio);
//             std::memcpy(data + i * step, chw.ptr<float>(), sizeof(float) * step);
//         }

//         infer_request_.set_input_tensor(input);
//         infer_request_.infer();
//         ov::Tensor output = infer_request_.get_output_tensor(0);

//         std::vector<OCRRecResult> batch_res = DecodeCTC(output);
//         for (size_t i = 0; i < bs; ++i) {
//             out[static_cast<size_t>(idx[beg + i])] = batch_res[i];
//         }
//     } 

//     return out;
// }
