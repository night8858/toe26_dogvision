#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp> // OpenVINO 2025 API

#include <dogvision_vision/common_structs.h>
#include <dogvision_vision/detector.hpp>


//负责寻找检测结果中的文本区域，并将文本区域坐标传递给 rec 模块进行识别
class detect_det_ppocr : public detector
{
public:
    explicit detect_det_ppocr(Appconfig* config) : detector(config) {}
    ~detect_det_ppocr() override = default;

    void load_model(const std::string& model_path, const std::string& device);

    void preprocess(cv::Mat &input_img)  override;
    void inference() override;
    void postprocess() override;


    const std::vector<OCRBox>& get_det_boxes() const { return ocr_det_out_; }
    
    std::vector<OCRBox> ocr_det_out_;
private:
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
    explicit detect_rec_ppocr(Appconfig* config) : detector(config) {}
    ~detect_rec_ppocr() override = default;

    void loda_dict(const std::string& dict_path);
    std::vector<OCRRecResult> Decode(const ov::Tensor& logits);

    void load_model(const std::string& model_path, const std::string& device);

    void preprocess(cv::Mat &input_img)  override;
    void inference() override;
    void postprocess() override;

    std::vector<OCRRecResult> result;

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
    explicit detect_cls_ppocr(Appconfig* config) : detector(config) {}
    ~detect_cls_ppocr() override = default;

    void load_model(const std::string& model_path, const std::string& device);
    void preprocess(cv::Mat &input_img) override;
    void inference() override;
    void postprocess() override;

private:
    ov::Core core_;
    ov::CompiledModel model_;
    ov::InferRequest infer_request_;
    ov::Tensor input_tensor_;
    ov::Tensor output_tensor_;
};

