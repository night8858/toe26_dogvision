#include <iostream>
#include <opencv2/opencv.hpp>
#include "nuc_detect.hpp"

namespace {

float calc_iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    const float inter_x1 = std::max(a.x, b.x);
    const float inter_y1 = std::max(a.y, b.y);
    const float inter_x2 = std::min(a.x + a.width, b.x + b.width);
    const float inter_y2 = std::min(a.y + a.height, b.y + b.height);

    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    const float area_a = std::max(0.0f, a.width) * std::max(0.0f, a.height);
    const float area_b = std::max(0.0f, b.width) * std::max(0.0f, b.height);
    const float denom = area_a + area_b - inter_area;

    if (denom <= 1e-6f) {
        return 0.0f;
    }
    return inter_area / denom;
}

} // namespace


bool detect_oponvino::inference_init(void)
{
    try {
        // 从基类配置中获取参数
        std::string xml_path = detect_config_.xml_file_path;
        std::string bin_path = detect_config_.bin_file_path;
        
        // 使用配置中的输入尺寸
        input_width_ = detect_config_.w;
        input_height_ = detect_config_.h;
        int batch_size = detect_config_.batch_size;
        
        // 读取模型 (OpenVINO 2025 API)
        // 如果 bin 文件路径为空，OpenVINO 会自动查找同名 .bin 文件
        std::shared_ptr<ov::Model> model;
        if (bin_path.empty()) {
            model = core_.read_model(xml_path);
        } else {
            model = core_.read_model(xml_path, bin_path);
        }
        
        auto input = model->input();
        std::string input_name = input.get_any_name();
        
        // 3. 设置输入形状 (NCHW格式)
        ov::Shape input_shape = {static_cast<size_t>(batch_size), 
                                  static_cast<size_t>(detect_config_.c), 
                                  static_cast<size_t>(input_height_), 
                                  static_cast<size_t>(input_width_)};
        model->reshape({{input_name, input_shape}});

        //测试先用着cpu，实际部署时根据需要选择设备用GPU
        model_ = core_.compile_model(model, "CPU");

        infer_request_ = model_.create_infer_request();
        
        // 获取模型输入的精度类型 (FP32/FP16/INT8)
        ov::element::Type input_type = model_.input().get_element_type();
        input_tensor_ = ov::Tensor(input_type, input_shape);
        infer_request_.set_input_tensor(input_tensor_);
        
        // 记录输入精度用于预处理
        input_precision_ = input_type;

        // 预读输出张量形状，缓存到成员变量，避免每帧重复解析
        // YOLOv8 输出形状: [1, 4+num_classes, num_anchors]
        ov::Shape out_shape = model_.output().get_shape();
        if (out_shape.size() >= 3) {
            out_num_channels_ = static_cast<int>(out_shape[1]);  // 4 + classes
            out_num_anchors_  = static_cast<int>(out_shape[2]);  // 如 8400
            out_num_classes_  = out_num_channels_ - 4;
        }
        
        
        std::string precision_str;
        if (input_type == ov::element::f32) precision_str = "FP32";
        else if (input_type == ov::element::f16) precision_str = "FP16";
        else if (input_type == ov::element::i8) precision_str = "INT8";
        else if (input_type == ov::element::u8) precision_str = "UINT8";
        else precision_str = input_type.get_type_name();
        
        //非调试状态需要注释掉这些打印，避免频繁输出影响性能
        std::cout << "OpenVINO model loaded successfully!" << std::endl;
        std::cout << "  Model: " << xml_path << std::endl;
        std::cout << "  Input precision: " << precision_str << std::endl;
        std::cout << "  Input shape: [" << batch_size << ", " << detect_config_.c 
                  << ", " << input_height_ << ", " << input_width_ << "]" << std::endl;
        std::cout << "  Classes: " << detect_config_.classes << std::endl;
        std::cout << "  Output shape: [1, " << out_num_channels_ << ", " << out_num_anchors_ << "]" << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "OpenVINO initialization failed: " << e.what() << std::endl;
        return false;
    }
}

// letterbox预处理：保持宽高比缩放并填充到目标尺寸
cv::Mat detect_oponvino::letterbox(const cv::Mat& src, int target_w, int target_h)
{
    int src_w = src.cols;
    int src_h = src.rows;
    
    // 计算缩放比例（取较小值保持宽高比）
    float scale_w = static_cast<float>(target_w) / src_w;
    float scale_h = static_cast<float>(target_h) / src_h;
    scale_ = std::min(scale_w, scale_h);
    
    // 计算缩放后的尺寸
    int new_w = static_cast<int>(src_w * scale_);
    int new_h = static_cast<int>(src_h * scale_);
    
    // 计算填充量
    pad_w_ = (target_w - new_w) / 2;
    pad_h_ = (target_h - new_h) / 2;
    
    // 缩放图像
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    
    // 创建目标尺寸的灰色画布 (114, 114, 114)
    cv::Mat dst(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    
    // 将缩放后的图像放置在中心
    resized.copyTo(dst(cv::Rect(pad_w_, pad_h_, new_w, new_h)));
    
    return dst;
}


void detect_oponvino::preprocess(cv::Mat &input_img)
{
    // 获取输入图像 (从基类获取)
    //cv::Mat& src = input_img_hik_;  // 或根据需要选择 input_img_usb_[cam_id]
    
    if (input_img.empty()) {
        return;
    }
    cv::Mat letterboxed = letterbox(input_img, input_width_, input_height_);
    
    //  BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    
    int channel_size = input_width_ * input_height_;
    
    // 根据模型输入精度选择不同的数据写入方式
    if (input_precision_ == ov::element::f32) {
        // FP32 精度
        float* input_data = input_tensor_.data<float>();
        float* ch_r = input_data;
        float* ch_g = input_data + channel_size;
        float* ch_b = input_data + channel_size * 2;
        
        const uchar* pixel = rgb.data;
        for (int i = 0; i < channel_size; ++i) {
            ch_r[i] = pixel[0] / 255.0f;
            ch_g[i] = pixel[1] / 255.0f;
            ch_b[i] = pixel[2] / 255.0f;
            pixel += 3;
        }
    } else if (input_precision_ == ov::element::f16) {
        // FP16 精度
        ov::float16* input_data = input_tensor_.data<ov::float16>();
        ov::float16* ch_r = input_data;
        ov::float16* ch_g = input_data + channel_size;
        ov::float16* ch_b = input_data + channel_size * 2;
        
        const uchar* pixel = rgb.data;
        for (int i = 0; i < channel_size; ++i) {
            ch_r[i] = ov::float16(pixel[0] / 255.0f);
            ch_g[i] = ov::float16(pixel[1] / 255.0f);
            ch_b[i] = ov::float16(pixel[2] / 255.0f);
            pixel += 3;
        }
    } else if (input_precision_ == ov::element::u8) {
        // UINT8 精度 (通常用于量化模型，不做归一化)
        uint8_t* input_data = input_tensor_.data<uint8_t>();
        uint8_t* ch_r = input_data;
        uint8_t* ch_g = input_data + channel_size;
        uint8_t* ch_b = input_data + channel_size * 2;
        
        const uchar* pixel = rgb.data;
        for (int i = 0; i < channel_size; ++i) {
            ch_r[i] = pixel[0];
            ch_g[i] = pixel[1];
            ch_b[i] = pixel[2];
            pixel += 3;
        }
    } else if (input_precision_ == ov::element::i8) {
        // INT8 精度
        int8_t* input_data = input_tensor_.data<int8_t>();
        int8_t* ch_r = input_data;
        int8_t* ch_g = input_data + channel_size;
        int8_t* ch_b = input_data + channel_size * 2;
        
        const uchar* pixel = rgb.data;
        for (int i = 0; i < channel_size; ++i) {
            ch_r[i] = static_cast<int8_t>(pixel[0] - 128);
            ch_g[i] = static_cast<int8_t>(pixel[1] - 128);
            ch_b[i] = static_cast<int8_t>(pixel[2] - 128);
            pixel += 3;
        }
    }
    // input_tensor_ 已在 inference_init 中绑定到 infer_request_
}


void detect_oponvino::inference()
{
    // 执行推理
    infer_request_.infer();
    
    // 获取输出张量
    output_tensor_ = infer_request_.get_output_tensor();
}

void detect_oponvino::postprocess()
{
    nms_results_.clear();
    decode_output();
    nms();
}

const std::vector<Detection>& detect_oponvino::get_nms_results() const
{
    return nms_results_;
}


void detect_oponvino::decode_output(void)
{
    boxes_raw_.clear();
    scores_raw_.clear();
    class_ids_raw_.clear();

    const ov::Shape out_shape = output_tensor_.get_shape();
    if (out_shape.size() != 3) {
        return;
    }

    const int num_channels = static_cast<int>(out_shape[1]);
    const int num_anchors = static_cast<int>(out_shape[2]);
    constexpr int kNumClasses = 4;
    constexpr int kBoxChannels = 4;

    // 按用户需求：输出固定为 [1, 8, 8400]（4 bbox + 4 classes）
    if (num_channels < (kBoxChannels + kNumClasses) || num_anchors <= 0) {
        return;
    }

    const float* data = output_tensor_.data<const float>();
    const float conf_thresh = detect_config_.bbox_conf_thresh;

    // 通过 letterbox 参数反推原图尺寸
    const int src_w = std::max(1, static_cast<int>((input_width_ - 2 * pad_w_) / scale_));
    const int src_h = std::max(1, static_cast<int>((input_height_ - 2 * pad_h_) / scale_));

    for (int a = 0; a < num_anchors; ++a) {
        // 找最高类别得分
        float best_score = 0.0f;
        int   best_cls   = 0;
        for (int c = 0; c < kNumClasses; ++c) {
            const float s = data[(kBoxChannels + c) * num_anchors + a];
            if (s > best_score) {
                best_score = s;
                best_cls   = c;
            }
        }
        if (best_score < conf_thresh) continue;

        // cx, cy, w, h (相对于 letterbox 输入尺寸)
        const float cx = data[0 * num_anchors + a];
        const float cy = data[1 * num_anchors + a];
        const float w  = data[2 * num_anchors + a];
        const float h  = data[3 * num_anchors + a];

        // 还原到原图坐标: 减去 padding 再除以 scale
        float x1 = (cx - w * 0.5f - pad_w_) / scale_;
        float y1 = (cy - h * 0.5f - pad_h_) / scale_;
        float x2 = (cx + w * 0.5f - pad_w_) / scale_;
        float y2 = (cy + h * 0.5f - pad_h_) / scale_;

        // 边界裁剪
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(src_w - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(src_h - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(src_w - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(src_h - 1)));

        const float bw = x2 - x1;
        const float bh = y2 - y1;
        if (bw <= 1.0f || bh <= 1.0f) {
            continue;
        }

        boxes_raw_.emplace_back(x1, y1, bw, bh);
        scores_raw_.push_back(best_score);
        class_ids_raw_.push_back(best_cls);
    }
}


//非极大值抑制：同类别框之间计算IoU，去除重叠过大的框
void detect_oponvino::nms(void)
{
    nms_results_.clear();
    if (boxes_raw_.empty()) {
        return;
    }

    constexpr int kNumClasses = 4;
    const float iou_thresh = detect_config_.nms_thresh;

    for (int cls = 0; cls < kNumClasses; ++cls) {
        std::vector<int> indices;
        indices.reserve(class_ids_raw_.size());
        for (size_t i = 0; i < class_ids_raw_.size(); ++i) {
            if (class_ids_raw_[i] == cls) {
                indices.push_back(static_cast<int>(i));
            }
        }

        std::sort(indices.begin(), indices.end(), [this](int a, int b) {
            return scores_raw_[a] > scores_raw_[b];
        });

        std::vector<bool> removed(indices.size(), false);
        for (size_t i = 0; i < indices.size(); ++i) {
            if (removed[i]) {
                continue;
            }
            const int keep_idx = indices[i];

            Detection det;
            const cv::Rect2f& r = boxes_raw_[static_cast<size_t>(keep_idx)];
            det.bbox[0] = r.x;
            det.bbox[1] = r.y;
            det.bbox[2] = r.width;
            det.bbox[3] = r.height;
            det.conf = scores_raw_[static_cast<size_t>(keep_idx)];
            det.class_id = static_cast<float>(class_ids_raw_[static_cast<size_t>(keep_idx)]);
            nms_results_.push_back(det);

            for (size_t j = i + 1; j < indices.size(); ++j) {
                if (removed[j]) {
                    continue;
                }
                const int cand_idx = indices[j];
                const float iou = calc_iou(
                    boxes_raw_[static_cast<size_t>(keep_idx)],
                    boxes_raw_[static_cast<size_t>(cand_idx)]
                );
                if (iou > iou_thresh) {
                    removed[j] = true;
                }
            }
        }
    }
}

