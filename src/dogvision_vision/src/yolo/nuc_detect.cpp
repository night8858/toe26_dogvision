#include <iostream>
#include <opencv2/opencv.hpp>
#include <dogvision_vision/nuc_detect.hpp>

/**
 * @file nuc_detect.cpp
 * @brief YOLO OpenVINO 推理流水线实现
 *
 * 包含以下模块：
 *   - letterbox 预处理（保持宽高比缩放 + 填充）
 *   - 多精度（FP32/FP16/U8/I8）的 HWC→CHW 转换与归一化
 *   - OpenVINO 同步推理
 *   - YOLOv8 输出解码（cxcywh → xywh + 逆 letterbox）
 *   - 按类别的贪心 NMS
 */

namespace {

/**
 * @brief 计算两个矩形框的交并比（IoU）
 *
 * @param a 矩形框 a（xywh 格式）
 * @param b 矩形框 b（xywh 格式）
 * @return float IoU 值，范围 [0.0, 1.0]；分母 ≤ 1e-6 时返回 0.0
 */
float calc_iou(const cv::Rect2f& a, const cv::Rect2f& b)
{
    // 计算相交矩形
    const float inter_x1 = std::max(a.x, b.x);
    const float inter_y1 = std::max(a.y, b.y);
    const float inter_x2 = std::min(a.x + a.width, b.x + b.width);
    const float inter_y2 = std::min(a.y + a.height, b.y + b.height);

    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    // 并集面积 = area_a + area_b - inter_area
    const float area_a = std::max(0.0f, a.width) * std::max(0.0f, a.height);
    const float area_b = std::max(0.0f, b.width) * std::max(0.0f, b.height);
    const float denom = area_a + area_b - inter_area;

    if (denom <= 1e-6f) {
        return 0.0f;
    }
    return inter_area / denom;
}

} // namespace

/**
 * @brief 初始化 YOLO OpenVINO 推理引擎
 *
 * 流程：
 *   1. 从基类配置中读取模型路径和输入尺寸
 *   2. 读取 OpenVINO IR 模型（.xml + .bin）
 *   3. 设置输入形状（NCHW 格式）
 *   4. 编译模型到 CPU 设备
 *   5. 创建推理请求并分配输入张量
 *   6. 预读输出张量形状，缓存 anchor/class/channel 数量以供推理时使用
 *
 * @param 无
 * @return true  初始化成功
 * @return false 初始化失败（模型路径/形状/编译错误）
 */
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

/**
 * @brief Letterbox 预处理：保持宽高比缩放 + 灰色填充至目标尺寸
 *
 * 计算原理：
 *   1. 分别计算宽/高方向的缩放比，取较小的值以确保图像完整放入目标框
 *   2. 缩放后，将图像放置在目标画布的中心，剩余区域用灰色 (114,114,114) 填充
 *   3. 记录 scale_、pad_w_、pad_h_ 供后处理坐标还原使用
 *
 * @param src      源 BGR 图像
 * @param target_w 目标宽度
 * @param target_h 目标高度
 * @return cv::Mat letterbox 处理后的图像
 */
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
    
    // 计算对称填充量
    pad_w_ = (target_w - new_w) / 2;
    pad_h_ = (target_h - new_h) / 2;
    
    // 缩放图像
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    
    // 创建目标尺寸的灰色画布（OpenVINO YOLO 训练常用值）
    cv::Mat dst(target_h, target_w, CV_8UC3, cv::Scalar(114, 114, 114));
    
    // 将缩放后的图像放置在中心
    resized.copyTo(dst(cv::Rect(pad_w_, pad_h_, new_w, new_h)));
    
    return dst;
}

/**
 * @brief YOLO 预处理：letterbox + BGR→RGB + HWC→CHW + 归一化
 *
 * 根据模型输入精度（FP32/FP16/UINT8/INT8）执行对应的归一化策略：
 *   - FP32 / FP16: 像素值除以 255.0，映射到 [0.0, 1.0]
 *   - UINT8:      保持原始 [0, 255] 范围（量化模型）
 *   - INT8:       减去 128，映射到 [-128, 127] 范围
 *
 * 数据排布从 HWC 转为 CHW 平面格式，直接写入已绑定到推理请求的 input_tensor_。
 *
 * @param input_img 输入 BGR 图像
 */
void detect_oponvino::preprocess(cv::Mat &input_img)
{
    if (input_img.empty()) {
        return;
    }

    // 1. Letterbox 缩放+填充
    cv::Mat letterboxed = letterbox(input_img, input_width_, input_height_);
    
    // 2. BGR → RGB（与 YOLO 训练时的颜色顺序对齐）
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    
    int channel_size = input_width_ * input_height_;
    
    // 3. 根据模型输入精度 HWC→CHW 并归一化
    if (input_precision_ == ov::element::f32) {
        // FP32 精度：除以 255.0 映射到 [0, 1]
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
        // FP16 精度：除以 255.0 后转为 half-float
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
        // UINT8 精度（量化模型）：保持原始 [0, 255] 范围
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
        // INT8 精度：减去 128，映射到 [-128, 127]
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
    // input_tensor_ 已在 inference_init 中绑定到 infer_request_，无需重复设置
}


/**
 * @brief 执行一次同步 OpenVINO 推理
 *
 * 调用 infer_request_.infer() 阻塞直到推理完成，
 * 然后从推理请求中获取输出张量。
 */
void detect_oponvino::inference()
{
    infer_request_.infer();
    output_tensor_ = infer_request_.get_output_tensor();
}

/**
 * @brief 后处理主入口：清空上一帧结果 → 解码原始输出 → NMS
 */
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

/**
 * @brief 解码 YOLOv8 输出张量为原始候选框
 *
 * YOLOv8 输出形状：[1, 4+num_classes, num_anchors]，channel-major 排列。
 * 每个 anchor 的格式：[cx, cy, w, h, cls0_score, cls1_score, ...]。
 *
 * 解码步骤：
 *   1. 遍历所有 anchor，找每个 anchor 的最高置信度类别
 *   2. 若最高置信度 < bbox_conf_thresh 则跳过
 *   3. 从 channel 0-3 读取 cx, cy, w, h（letterbox 空间坐标）
 *   4. 逆 letterbox 变换：减去对称 padding 再除以缩放比，还原到原图坐标
 *   5. 非极大值抑制（NMS）裁剪到原图边界内
 */
void detect_oponvino::decode_output(void)
{
    boxes_raw_.clear();
    scores_raw_.clear();
    class_ids_raw_.clear();

    // 使用 inference_init 阶段预读的输出形状，避免每帧重复解析
    // YOLOv8 输出格式: [1, 4+num_classes, num_anchors]，channel-major 排列
    if (out_num_anchors_ <= 0 || out_num_classes_ <= 0) {
        return;
    }

    const float* data = output_tensor_.data<const float>();
    const float conf_thresh = detect_config_.bbox_conf_thresh;

    // 通过 letterbox 参数反推原图尺寸（浮点，避免整数截断误差）
    const float src_w = static_cast<float>(input_width_  - 2 * pad_w_) / scale_;
    const float src_h = static_cast<float>(input_height_ - 2 * pad_h_) / scale_;

    for (int a = 0; a < out_num_anchors_; ++a) {
        // 遍历所有类别，找最高置信度及对应类别
        // 数据布局: data[channel * out_num_anchors_ + anchor]
        float best_score = 0.0f;
        int   best_cls   = 0;
        for (int c = 0; c < out_num_classes_; ++c) {
            const float s = data[(4 + c) * out_num_anchors_ + a];
            if (s > best_score) {
                best_score = s;
                best_cls   = c;
            }
        }
        if (best_score < conf_thresh) continue;

        // 从 channel 0-3 读取 cx, cy, w, h（letterbox 空间坐标）
        const float cx = data[0 * out_num_anchors_ + a];
        const float cy = data[1 * out_num_anchors_ + a];
        const float bw = data[2 * out_num_anchors_ + a];
        const float bh = data[3 * out_num_anchors_ + a];

        // 逆 letterbox：减去对称 padding 再除以缩放比，还原到原图坐标
        float x1 = (cx - bw * 0.5f - static_cast<float>(pad_w_)) / scale_;
        float y1 = (cy - bh * 0.5f - static_cast<float>(pad_h_)) / scale_;
        float x2 = (cx + bw * 0.5f - static_cast<float>(pad_w_)) / scale_;
        float y2 = (cy + bh * 0.5f - static_cast<float>(pad_h_)) / scale_;

        // 边界裁剪到原图范围
        x1 = std::max(0.0f, std::min(x1, src_w - 1.0f));
        y1 = std::max(0.0f, std::min(y1, src_h - 1.0f));
        x2 = std::max(0.0f, std::min(x2, src_w - 1.0f));
        y2 = std::max(0.0f, std::min(y2, src_h - 1.0f));

        const float rw = x2 - x1;
        const float rh = y2 - y1;
        if (rw <= 1.0f || rh <= 1.0f) continue;

        boxes_raw_.emplace_back(x1, y1, rw, rh);
        scores_raw_.push_back(best_score);
        class_ids_raw_.push_back(best_cls);
    }
}


/**
 * @brief 按类别的贪心非极大值抑制（NMS）
 *
 * 算法：
 *   1. 按类别分组，同一类别的候选框集合内执行 NMS
 *   2. 对每个类别，按置信度降序排列
 *   3. 贪心保留置信度最高的框，抑制与其 IoU 超过 nms_thresh 的其他框
 *   4. 将保留的框存入 nms_results_
 *
 * @param 无
 */
void detect_oponvino::nms(void)
{
    nms_results_.clear();
    if (boxes_raw_.empty() || out_num_classes_ <= 0) {
        return;
    }

    const float iou_thresh = detect_config_.nms_thresh;

    for (int cls = 0; cls < out_num_classes_; ++cls) {
        // 收集属于该类别的候选框索引
        std::vector<int> indices;
        indices.reserve(class_ids_raw_.size());
        for (size_t i = 0; i < class_ids_raw_.size(); ++i) {
            if (class_ids_raw_[i] == cls) {
                indices.push_back(static_cast<int>(i));
            }
        }
        if (indices.empty()) continue;

        // 按置信度降序排列
        std::sort(indices.begin(), indices.end(), [this](int a, int b) {
            return scores_raw_[a] > scores_raw_[b];
        });

        std::vector<bool> suppressed(indices.size(), false);
        for (size_t i = 0; i < indices.size(); ++i) {
            if (suppressed[i]) continue;

            const int keep = indices[i];
            Detection det;
            const cv::Rect2f& r = boxes_raw_[static_cast<size_t>(keep)];
            det.bbox[0]  = r.x;
            det.bbox[1]  = r.y;
            det.bbox[2]  = r.width;
            det.bbox[3]  = r.height;
            det.conf     = scores_raw_[static_cast<size_t>(keep)];
            det.class_id = static_cast<float>(cls);
            nms_results_.push_back(det);

            // 抑制与 keep 框 IoU 超过阈值的后续候选框
            for (size_t j = i + 1; j < indices.size(); ++j) {
                if (suppressed[j]) continue;
                if (calc_iou(boxes_raw_[static_cast<size_t>(keep)],
                             boxes_raw_[static_cast<size_t>(indices[j])]) > iou_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }
}

