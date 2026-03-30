#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp> // OpenVINO 2025 API
#include <vector>
#include <algorithm>
#include <fstream>
#include <cstring>

#include "common_structs.h"
#include "detector.hpp"
#include "ocr_detect.hpp"

// =============================================================================
// ocr_detect.cpp
// PPOCRv4 推理流水线实现，分为三个阶段：
//   1. detect_det_ppocr  ── 文本检测（DB 模型）
//      输入整张图像，输出各文字区域的四点包围框（OCRBox）。
//   2. detect_rec_ppocr  ── 文本识别（CRNN/SVTR 模型）
//      输入裁剪后的单行文字图像，输出对应的字符序列和置信度。
//   3. detect_cls_ppocr  ── 文字方向分类（可选）
//      判断文字是否倒置，结果可用于旋转 rec 输入以提升识别率。
//
// 依赖：OpenVINO 2025 Runtime，OpenCV 4.x
// 模型格式：PaddlePaddle (.pdmodel / .pdiparams) 可由 OpenVINO 直接读取。
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
// detect_det_ppocr ── 文本检测阶段
// ─────────────────────────────────────────────────────────────────────────────

// 加载 det 检测模型并在指定推理设备（CPU/GPU 等）上完成编译
void detect_det_ppocr::load_model(const std::string& model_path, const std::string& device)
{
    // 读取 PaddlePaddle 格式模型（OpenVINO 可直接解析 .pdmodel 文件）
    std::shared_ptr<ov::Model> model = core_.read_model(model_path);

    // 将模型编译到目标推理设备，生成可执行的 CompiledModel
    model_ = core_.compile_model(model, device);

    // 创建推理请求对象，后续每次推理均通过该对象完成
    infer_request_ = model_.create_infer_request();
}


// det 阶段预处理：缩放图像 → 归一化 → 转换为 NCHW float32 张量
//
// DB 模型要求输入宽高均为 32 的倍数（卷积下采样对齐约束）。
// 预处理步骤：
//   1. 按 det_limit_type（"max" 或 "min"）确定缩放比例，使图像最长/最短边
//      不超过 / 不小于 det_limit_side_len，避免显存溢出或分辨率过低。
//   2. 将缩放后的宽高对齐到 32 的倍数（向上取整后 ×32，最小为 32）。
//   3. 将 BGR uint8 图像缩放后转为 float32 并归一化到 [0,1]。
//   4. 使用 ImageNet 均值/标准差做标准化（与 PPOCRv4 训练设置一致）：
//        mean = [0.485, 0.456, 0.406]（RGB 顺序）
//        std  = [0.229, 0.224, 0.225]
//   5. 将 HWC float32 矩阵重排为 CHW（NCHW, batch=1），直接写入
//      ov::Tensor 的内存区域（零拷贝方式）。
//   6. 记录缩放元信息（ratio_h / ratio_w）用于后处理坐标还原。
void detect_det_ppocr::preprocess(cv::Mat &input_img)
{
    // 记录原图尺寸，供 postprocess 阶段将检测框坐标映射回原图
    Mate.src_h = input_img.rows;
    Mate.src_w = input_img.cols;

    float ratio = 1.0f;
    // ── 缩放比例计算 ──────────────────────────────────────────────────────────
    // "max" 策略：若最长边超过限制值，则按最长边缩放（防止大图占用过多显存）
    // 其他策略：若最短边小于限制值，则按最短边放大（确保小图文字不过模糊）
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

    // ── 对齐到 32 的倍数 ──────────────────────────────────────────────────────
    // DB 的多层下采样（步长 4/8/16/32）要求输入宽高是 32 的倍数，否则特征图
    // 尺寸不整除会导致输出概率图与输入图坐标对应关系出现偏移。
    // std::round(.../ 32) * 32 为最近邻对齐，最小值保证为 32。
    int resize_h = std::max(32, static_cast<int>(std::round(Mate.src_h * ratio / 32.0f) * 32.0f));
    int resize_w = std::max(32, static_cast<int>(std::round(Mate.src_w * ratio / 32.0f) * 32.0f));

    // 存储实际缩放比，用于 postprocess 将概率图坐标映射回原图
    Mate.resize_h = resize_h;
    Mate.resize_w = resize_w;
    Mate.ratio_h = static_cast<float>(resize_h) / static_cast<float>(Mate.src_h);
    Mate.ratio_w = static_cast<float>(resize_w) / static_cast<float>(Mate.src_w);

    // ── 缩放 + 归一化 ─────────────────────────────────────────────────────────
    cv::Mat resized;
    cv::resize(input_img, resized, cv::Size(resize_w, resize_h));
    // uint8 [0,255] → float32 [0.0, 1.0]
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    // ImageNet 标准化：(pixel - mean) / std，令模型输入分布接近训练集
    const cv::Scalar mean(0.485, 0.456, 0.406); // RGB 各通道均值
    const cv::Scalar std(0.229, 0.224, 0.225);  // RGB 各通道标准差
    cv::subtract(resized, mean, resized);
    cv::divide(resized, std, resized);

    // ── 转为 NCHW ov::Tensor（零拷贝写入）────────────────────────────────────
    // 创建形状为 [1, 3, H, W] 的 float32 张量，直接获取底层内存指针
    ov::Tensor input(ov::element::f32, {1, 3, static_cast<size_t>(resize_h), static_cast<size_t>(resize_w)});
    float *data = input.data<float>();

    // 将 HWC 的三通道矩阵拆分为三个独立平面，分别映射到张量内存的三段
    // channels[c] 直接指向 data + c * H * W，cv::split 完成 HWC → CHW 转换
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c)
    {
        channels[c] = cv::Mat(resize_h, resize_w, CV_32FC1, data + c * resize_h * resize_w);
    }
    cv::split(resized, channels); // 将 resized（HWC）各通道数据写入 channels[c]

    input_tensor_ = input; // 保存张量供 inference() 调用
}

// det 阶段推理：将预处理后的张量送入模型，获取概率图输出张量
// 输出张量形状为 [1, 1, H', W']，其中 H' = H/4，W' = W/4（DB 模型 4× 下采样）
// 每个像素值表示对应原图区域属于文字前景的概率（0~1）
void detect_det_ppocr::inference()
{
    // 绑定输入张量（preprocess 阶段已填充数据）
    infer_request_.set_input_tensor(input_tensor_);
    // 执行同步推理，阻塞直到推理完成
    infer_request_.infer();
    // 取出第一个输出张量（DB 模型只有一个输出：概率图）
    output_tensor_ = infer_request_.get_output_tensor();
}

// 将四个无序顶点按「左上→右上→右下→左下」的顺时针顺序排列
//
// 利用以下几何性质快速定位各角点（适用于接近轴对齐的矩形框）：
//   - 左上角(TL)：x+y 最小（坐标和最小，距原点最近）
//   - 右下角(BR)：x+y 最大（坐标和最大）
//   - 右上角(TR)：y-x 最小（纵坐标明显小于横坐标）
//   - 左下角(BL)：y-x 最大（纵坐标明显大于横坐标）
//
// 排列后的点序（rect[0..3]）= TL, TR, BR, BL，供后续
// Unclip 和 crop_text_region 使用。
std::array<cv::Point2f, 4> detect_det_ppocr::OrderPointsClockwise(const std::vector<cv::Point2f> &pts) const
{
    std::array<cv::Point2f, 4> rect;
    std::vector<float> s(4), d(4); // s[i] = x+y；d[i] = y-x
    for (int i = 0; i < 4; ++i)
    {
        s[i] = pts[i].x + pts[i].y;
        d[i] = pts[i].y - pts[i].x;
    }
    rect[0] = pts[static_cast<size_t>(std::distance(s.begin(), std::min_element(s.begin(), s.end())))]; // TL
    rect[2] = pts[static_cast<size_t>(std::distance(s.begin(), std::max_element(s.begin(), s.end())))]; // BR
    rect[1] = pts[static_cast<size_t>(std::distance(d.begin(), std::min_element(d.begin(), d.end())))]; // TR
    rect[3] = pts[static_cast<size_t>(std::distance(d.begin(), std::max_element(d.begin(), d.end())))]; // BL
    return rect;
}

// det 阶段后处理：概率图 → 二值化 → 轮廓提取 → Unclip → 坐标映射
//
// 完整流程：
//   1. 从输出张量（形状 [1,1,H',W']）中取出概率图。
//   2. 以 det_db_thresh 做全局二值化，将高概率像素标记为文字前景。
//   3. 用 findContours 提取所有连通域轮廓。
//   4. 对每个轮廓：
//      a. 过滤点数 < 4 或最小包围矩形的短边 < 3px 的噪声轮廓。
//      b. 计算轮廓的 axis-aligned bounding box，用概率图均值
//         过滤置信度低于 det_db_box_thresh 的候选框。
//      c. 用 minAreaRect 拟合最小旋转包围矩形，取 4 个顶点。
//      d. 执行 Unclip（DB 论文提出的多边形外扩），补偿训练时收缩的误差。
//      e. 执行局部 W/H 轴附加扩张，进一步保证文字不被截断。
//      f. 将检测图坐标除以 ratio_w / ratio_h 映射回原图，clamp 在图像范围内。
//   5. 将所有有效框按从上到下、从左到右的阅读顺序排序后存入 ocr_det_out_。
void detect_det_ppocr::postprocess()
{
    // 检查输出张量形状是否为 [N, C, H, W]（标准 4D）
    const auto shape = output_tensor_.get_shape();
    if (shape.size() != 4)
    {
        return;
    }

    // 概率图尺寸（通常为原图的 1/4）
    const size_t h = shape[2];
    const size_t w = shape[3];
    const float *p = output_tensor_.data<const float>();

    // 将输出张量数据复制到 OpenCV Mat，便于后续阈值化和轮廓操作
    cv::Mat prob_map(static_cast<int>(h), static_cast<int>(w), CV_32FC1);
    std::memcpy(prob_map.data, p, sizeof(float) * h * w);

    // ── 二值化 ────────────────────────────────────────────────────────────────
    // 以 det_db_thresh 为阈值，将文字前景概率高的像素置为 255（前景），
    // 其余置为 0（背景），生成二值掩码图
    cv::Mat bin;
    cv::threshold(prob_map, bin, detect_config_.det_db_thresh, 255, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8UC1); // findContours 要求 CV_8UC1 输入

    // ── 轮廓提取 ──────────────────────────────────────────────────────────────
    // RETR_LIST：提取所有轮廓，不建立层级关系（效率更高）
    // CHAIN_APPROX_SIMPLE：只保存端点，压缩轮廓数据
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // 清空上一帧的检测结果
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

        // ── Step 1: Unclip ────────────────────────────────────────────────────
        // DB（Differentiable Binarization）模型在训练时会将标注框向内收缩
        //（shrink），使输出概率图中的文字区域比真实边界小。
        // Unclip 通过将检测多边形各顶点沿"重心→顶点"方向向外推移来补偿
        // 这一偏差，推移距离由以下公式决定（来自原论文）：
        //
        //   distance = polygon_area × unclip_ratio / perimeter
        //
        // unclip_ratio 越大，外扩幅度越大；官方 PPOCRv4 默认值为 1.5~2.0。
        {
            // 用 Shoelace（向量叉积累加）公式计算四边形面积
            // 叉积累加结果的绝对值 / 2 = 多边形面积（系数必须为 0.5f）
            float poly_area = 0.0f;
            for (int i = 0; i < 4; ++i)
            {
                const int j = (i + 1) % 4;
                poly_area += ordered[i].x * ordered[j].y
                           - ordered[j].x * ordered[i].y;
            }
            poly_area = std::abs(poly_area) * 0.5f; // Shoelace: area = |Σ叉积| / 2

            // 计算多边形周长
            float perimeter = 0.0f;
            for (int i = 0; i < 4; ++i)
            {
                const int j = (i + 1) % 4;
                perimeter += static_cast<float>(cv::norm(ordered[j] - ordered[i]));
            }

            if (poly_area > 0.0f && perimeter > 1e-4f)
            {
                // 外扩距离（像素，检测图分辨率空间）
                const float distance =
                    poly_area * detect_config_.det_db_unclip_ratio / perimeter;

                // 计算四点重心
                cv::Point2f centroid(0.0f, 0.0f);
                for (int i = 0; i < 4; ++i) centroid += ordered[i];
                centroid *= 0.25f;

                // 每个顶点沿"重心→顶点"方向向外移动 distance 像素
                for (int i = 0; i < 4; ++i)
                {
                    cv::Point2f dir = ordered[i] - centroid;
                    const float d = static_cast<float>(cv::norm(dir));
                    if (d > 1e-4f)
                        ordered[i] += dir * (distance / d);
                }
            }
        }

        // ── Step 2: 沿局部 W / H 轴方向附加扩张 ─────────────────────────────
        // Unclip 对各顶点做各向同性（径向）扩张，对宽度和高度的控制力相同。
        // 当文字行截断仍发生时，可在矩形的局部坐标系下对 W 轴和 H 轴方向
        // 单独追加固定像素的扩张，以精确控制左右/上下各侧的扩边量。
        //
        // 点序约定（OrderPointsClockwise 输出）：
        //   ordered[0] = 左上(TL)   ordered[1] = 右上(TR)
        //   ordered[2] = 右下(BR)   ordered[3] = 左下(BL)
        //
        // W 轴单位向量 = (TR - TL) / |TR - TL|    ← 沿文本行方向（水平）
        // H 轴单位向量 = (BL - TL) / |BL - TL|    ← 垂直文本行方向（垂直）
        //
        // 调整 w_expand / h_expand 即可改变最终框的宽度/高度：
        //   增大 w_expand → 框左右各扩 w_expand 像素（补文字左右截断）
        //   增大 h_expand → 框上下各扩 h_expand 像素（补文字上下截断）
        // 注意：此处单位是检测模型输入图分辨率下的像素；
        //       映射回原图时会由 ratio_w / ratio_h 自动缩放。
        {
            // 每侧沿宽度方向（W 轴）扩展的像素数（检测图空间）
            constexpr float w_expand = 10.0f;
            // 每侧沿高度方向（H 轴）扩展的像素数（检测图空间）
            constexpr float h_expand = 10.0f;

            // 计算局部坐标系的方向向量
            const cv::Point2f w_vec = ordered[1] - ordered[0]; // TL → TR
            const cv::Point2f h_vec = ordered[3] - ordered[0]; // TL → BL
            const float w_len = static_cast<float>(cv::norm(w_vec));
            const float h_len = static_cast<float>(cv::norm(h_vec));

            if (w_len > 1e-4f && h_len > 1e-4f)
            {
                const cv::Point2f w_unit = w_vec / w_len; // W 轴单位向量
                const cv::Point2f h_unit = h_vec / h_len; // H 轴单位向量

                // TL：向左(−W) w_expand 像素，向上(−H) h_expand 像素
                ordered[0] -= w_unit * w_expand + h_unit * h_expand;
                // TR：向右(+W) w_expand 像素，向上(−H) h_expand 像素
                ordered[1] += w_unit * w_expand - h_unit * h_expand;
                // BR：向右(+W) w_expand 像素，向下(+H) h_expand 像素
                ordered[2] += w_unit * w_expand + h_unit * h_expand;
                // BL：向左(−W) w_expand 像素，向下(+H) h_expand 像素
                ordered[3] -= w_unit * w_expand - h_unit * h_expand;
            }
        }

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
        // 按阅读顺序排序：若两个框的顶部 y 坐标差距 < 10px（视为同一行）
        // 则按 x 升序（左 → 右）；否则按 y 升序（上 → 下）
        if (std::abs(a.pts[0].y - b.pts[0].y) < 10.0f) {
            return a.pts[0].x < b.pts[0].x;
        }
        return a.pts[0].y < b.pts[0].y; });
}



// =============================================================================
// detect_rec_ppocr ── 文本识别阶段
// =============================================================================

// 加载 rec 识别模型（CRNN/SVTR 结构，PaddlePaddle 格式）并在指定设备上编译
void detect_rec_ppocr::load_model(const std::string& model_path, const std::string& device)
{
    // 读取 PaddlePaddle 模型（OpenVINO 可直接解析 .pdmodel + .pdiparams）
    std::shared_ptr<ov::Model> model = core_.read_model(model_path);
    // 编译到目标设备
    model_ = core_.compile_model(model, device);
    // 创建推理请求对象
    infer_request_ = model_.create_infer_request();
}

// 加载字符字典文件，构建索引 → 字符的映射表（dict_）
//
// 字典格式：每行一个字符（中文汉字/英文字母/符号等）。
// 索引约定：
//   0     = CTC blank（空白标签，用于 CTC 解码时删除重复）
//   1…N   = 字典文件中的字符（同次序）
//   N+1   = 空格字符（字典未包含空格时的备选）
void detect_rec_ppocr::loda_dict(const std::string& dict_path)
{
    dict_.clear();
    dict_.push_back("blank"); // 索引 0：CTC blank 标签

    std::ifstream ifs(dict_path);
    std::string line;
    while (std::getline(ifs, line))
    {
        // 先除去 Windows 换行符 \r，避免 Windows 格式字典导致的识别符号包含多余字符
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        dict_.push_back(line);
    }
    dict_.push_back(" "); // 最后一个索引为空格
}

// rec 阶段预处理：裁剪单行文字图 2→ 缩放到固定高度 → 归一化 → 转 NCHW 张量
//
// rec 模型要求固定高度（rec_img_h，默认 48px）和最大宽度（rec_img_w）。
// 预处理步骤：
//   1. 根据当前裁剪图的宽高比计算实际缩放后宽度 resized_w，
//      上限为 max_wh_ratio 对应的最大宽度 img_w。
//   2. 将图像缩放到 (resized_w, rec_img_h)。
//   3. uint8 [0,255] → float32 [0.0, 1.0]。
//   4. 将 CHW buffer 初始化为 0.0（有效宽度右侧的填充区保持中性不衅深识别）。
//   5. 对有效宽度区域做归一化：(pixel - 0.5) / 0.5，将 [0,1] 映射到 [-1, 1]。
//   6. 将 CHW float32 数据拷贝入 ov::Tensor 并存入 input_tensor_。
void detect_rec_ppocr::preprocess(cv::Mat &input_img)
{
    const int img_c = detect_config_.rec_img_c;                    // 通道数（默认 3）
    const int img_h = detect_config_.rec_img_h;                    // 固定输入高度（默认 48px）
    const int img_w = static_cast<int>(img_h * max_wh_ratio);     // 最大宽度＝高度 × 最大宽高比

    // 保持高度不变缩放宽度，不超过最大宽度（敦宽图不裁断）
    const float ratio = static_cast<float>(input_img.cols) / static_cast<float>(input_img.rows);
    int resized_w = static_cast<int>(std::ceil(img_h * ratio));
    resized_w = std::min(resized_w, img_w);

    cv::Mat resized;
    cv::resize(input_img, resized, cv::Size(resized_w, img_h));
    // uint8 [0,255] → float32 [0.0, 1.0]
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    // 初始化 CHW buffer 为 0.0（归一化后的填充默认值）
    cv::Mat chw(img_c, img_h * img_w, CV_32F, cv::Scalar(0.0f));
    std::vector<cv::Mat> channels;
    cv::split(resized, channels);
    for (int c = 0; c < img_c; ++c) {
        // 只对有效宽度区域做归一化，右侧填充区保持 0.0
        cv::Mat dst = chw.row(c).reshape(1, img_h).colRange(0, resized_w);
        channels[c].copyTo(dst);  // [0.0, 1.0]
        dst -= 0.5f;              // 平移至 [-0.5, 0.5]
        dst /= 0.5f;              // 拉伸至 [-1.0, 1.0]
    }
    chw_img = chw;

    // 将 CHW 数据封装为 ov::Tensor（形状 [1, C, H, W]）并拷贝数据
    ov::Tensor input(ov::element::f32,
                     {1,
                      static_cast<size_t>(img_c),
                      static_cast<size_t>(img_h),
                      static_cast<size_t>(img_w)});
    std::memcpy(input.data<float>(), chw_img.ptr<float>(),
                sizeof(float)
                    * static_cast<size_t>(img_c)
                    * static_cast<size_t>(img_h)
                    * static_cast<size_t>(img_w));
    input_tensor_ = input; // 保存张量供 inference() 调用
}


// rec 阶段推理：将预处理张量送入 rec 模型，获取 CTC logits 输出
// 输出张量形状为 [batch, time_step, vocab_size]
//   - time_step：水平时序步数（由输入宽度决定）
//   - vocab_size：字典大小 + 1（包含 CTC blank）
void detect_rec_ppocr::inference()
{
    infer_request_.set_input_tensor(input_tensor_);
    infer_request_.infer();
    output_tensor_ = infer_request_.get_output_tensor();
}

// rec 阶段后处理：将 CTC logits 张量转换为可读文本序列
void detect_rec_ppocr::postprocess()
{
    // 调用 Decode 完成 CTC 贪心解码，结果存入成员变量 result
    result = Decode(output_tensor_);
}


// CTC 贪心解码：将每个时间步的 logits 转换为字符序列
//
// CTC（Connectionist Temporal Classification）解码规则：
//   1. 尚规：在每个时序步 t 选取 logits 最大索引 best。
//   2. 去重：连续相同索引的时序步合并为一个（best == prev_idx 则跳过）。
//   3. 去 blank：删除索引为 0（CTC blank）的帧。
// 置信度计算：每个有效字符位置的最大概率均值（排除 blank 帧）。
std::vector<OCRRecResult> detect_rec_ppocr::Decode(const ov::Tensor& logits)
{
    // 使用成员 output_tensor_（logits 形参仅保持接口一致性，未单独使用）
    const auto s = output_tensor_.get_shape(); // [batch, time_step, vocab_size]
    const size_t batch     = s[0];
    const size_t time_step = s[1]; // 时序步数（水平时序）
    const size_t cls_num   = s[2]; // 字典大小（包含 blank）
    const float* p = output_tensor_.data<const float>();

    std::vector<OCRRecResult> out(batch);
    for (size_t b = 0; b < batch; ++b) {
        std::string text;
        float score_sum = 0.0f;
        int count    = 0;  // 有效字符数量，用于计算均均置信度
        int prev_idx = -1; // 上一个时序步的索引，用于去重按照

        for (size_t t = 0; t < time_step; ++t) {
            // 在第 b*time_step*cls_num + t*cls_num 起始的 cls_num 个 logit 中找最大分
            size_t best      = 0;
            float best_score = p[b * time_step * cls_num + t * cls_num];
            for (size_t c = 1; c < cls_num; ++c) {
                const float v = p[b * time_step * cls_num + t * cls_num + c];
                if (v > best_score) {
                    best_score = v;
                    best = c;
                }
            }

            // 去重：连续类相同的帧合并
            if (static_cast<int>(best) == prev_idx) {
                continue;
            }
            prev_idx = static_cast<int>(best);

            // 去 blank：跳过 CTC blank（索引 0）
            if (best == 0) {
                continue;
            }

            // 索引在字典范围内则追加对应字符
            if (best < dict_.size()) {
                text += dict_[best];
            }
            score_sum += best_score;
            ++count;
        }

        out[b].text  = text;
        // 匹均均置信度：无有效字符时返回 0.0
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
