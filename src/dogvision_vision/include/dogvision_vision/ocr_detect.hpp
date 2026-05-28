#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp> // OpenVINO 2025 接口

#include <dogvision_vision/common_structs.h>
#include <dogvision_vision/detector.hpp>


//负责寻找检测结果中的文本区域，并将文本区域坐标传递给 rec 模块进行识别
class detect_det_ppocr : public detector
{
public:
    /**
     * @brief 构造 PPOCR 文本检测器。
     * @param config 应用配置；为 nullptr 时仅用于加载配置。
     * @retval 无
     */
    explicit detect_det_ppocr(Appconfig* config) : detector(config) {}
    ~detect_det_ppocr() override = default;

    /**
     * @brief 加载并编译文本检测模型。
     * @param model_path Paddle/OpenVINO 模型文件路径。
     * @param device OpenVINO 设备名称。
     * @retval void
     */
    void load_model(const std::string& model_path, const std::string& device);

    /**
     * @brief 为 PPOCR 文本检测预处理图像。
     * @param input_img 输入 BGR 图像。
     * @retval void
     */
    void preprocess(cv::Mat &input_img)  override;

    /**
     * @brief 执行文本检测推理请求。
     * @param 无
     * @retval void
     */
    void inference() override;

    /**
     * @brief 将检测概率图转换为 OCR 文本框。
     * @param 无
     * @retval void
     */
    void postprocess() override;

    /**
     * @brief 返回最近一次文本检测框。
     * @param 无
     * @retval const std::vector<OCRBox>& 缓存的 OCR 文本框。
     */
    const std::vector<OCRBox>& get_det_boxes() const { return ocr_det_out_; }
    
    std::vector<OCRBox> ocr_det_out_;
private:
    /**
     * @brief 将四个点排序为左上、右上、右下、左下。
     * @param pts 未排序的四点多边形。
     * @retval std::array<cv::Point2f, 4> 排序后的多边形点。
     */
    std::array<cv::Point2f, 4> OrderPointsClockwise(const std::vector<cv::Point2f>& pts) const;

    ov::Core core_;
    ov::CompiledModel model_;
    ov::InferRequest infer_request_;
    ov::Tensor input_tensor_;
    ov::Tensor output_tensor_;

    
    DetResizeMeta Mate;

    
};


//负责对 det 模块传递过来的文本区域进行识别，输出文本内容
class detect_rec_ppocr : public detector
{
public:
    /**
     * @brief 构造 PPOCR 文本识别器。
     * @param config 应用配置；为 nullptr 时仅用于加载配置。
     * @retval 无
     */
    explicit detect_rec_ppocr(Appconfig* config) : detector(config) {}
    ~detect_rec_ppocr() override = default;

    /**
     * @brief 加载识别字典。
     * @param dict_path 字典文本文件路径。
     * @retval void
     */
    void loda_dict(const std::string& dict_path);

    /**
     * @brief 使用 CTC 风格折叠解码识别 logits。
     * @param logits OpenVINO 输出张量。
     * @retval std::vector<OCRRecResult> 识别结果列表。
     */
    std::vector<OCRRecResult> Decode(const ov::Tensor& logits);

    /**
     * @brief 加载并编译文本识别模型。
     * @param model_path Paddle/OpenVINO 模型文件路径。
     * @param device OpenVINO 设备名称。
     * @retval void
     */
    void load_model(const std::string& model_path, const std::string& device);

    /**
     * @brief 为识别预处理裁剪后的文本图像。
     * @param input_img 裁剪后的文本图像。
     * @retval void
     */
    void preprocess(cv::Mat &input_img)  override;

    /**
     * @brief 执行文本识别推理请求。
     * @param 无
     * @retval void
     */
    void inference() override;

    /**
     * @brief 解码并保存最近一次识别输出。
     * @param 无
     * @retval void
     */
    void postprocess() override;

    std::vector<OCRRecResult> result;

    /**
     * @brief 设置识别输入的最大宽高比。
     * @param r 最大宽高比。
     * @retval void
     */
    void set_max_wh_ratio(float r) { max_wh_ratio = r; }

private:
    // 文本行最大宽高比（rec 输入图像 W/H），超过则压缩至最大宽度而非截断
    // 默认值与 s_detector_params 中 rec_img_w/rec_img_h 一致（320/48）
    float max_wh_ratio = 320.0f / 48.0f;
    ov::Core core_;
    ov::CompiledModel model_;
    ov::InferRequest infer_request_;
    ov::Tensor input_tensor_;
    ov::Tensor output_tensor_;

    
    std::vector<std::string> dict_;// OCR识别字典
    cv::Mat chw_img; // 预处理后CHW格式的输入图像

};

//目前没写
//负责对文本区域进行方向分类，输出文本方向（可以不用，就是用来检查文本是否被倒置的）
class detect_cls_ppocr : public detector
{
public:
    /**
     * @brief 构造 PPOCR 文本方向分类器。
     * @param config 应用配置；为 nullptr 时仅用于加载配置。
     * @retval 无
     */
    explicit detect_cls_ppocr(Appconfig* config) : detector(config) {}
    ~detect_cls_ppocr() override = default;

    /**
     * @brief 加载并编译文本方向分类模型。
     * @param model_path Paddle/OpenVINO 模型文件路径。
     * @param device OpenVINO 设备名称。
     * @retval void
     */
    void load_model(const std::string& model_path, const std::string& device);

    /**
     * @brief 为方向分类预处理文本图像。
     * @param input_img 输入文本图像。
     * @retval void
     */
    void preprocess(cv::Mat &input_img) override;

    /**
     * @brief 执行方向分类推理请求。
     * @param 无
     * @retval void
     */
    void inference() override;

    /**
     * @brief 解码方向分类结果。
     * @param 无
     * @retval void
     */
    void postprocess() override;

private:
    ov::Core core_;
    ov::CompiledModel model_;
    ov::InferRequest infer_request_;
    ov::Tensor input_tensor_;
    ov::Tensor output_tensor_;
};
