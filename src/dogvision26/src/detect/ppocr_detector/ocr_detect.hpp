#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp> // OpenVINO 2025 API

#include "common_structs.h"
#include "detector.hpp"


//负责寻找检测结果中的文本区域，并将文本区域坐标传递给 rec 模块进行识别
class detect_det_ppocr : public detector
{
    public:

    ov::Core core_;
    ov::CompiledModel model_;
    ov::InferRequest infer_request_;
    ov::Tensor input_tensor_;
    ov::Tensor output_tensor_;

    void preprocess(cv::Mat &input_img)  override;
    void inference() override;
    void postprocess() override;
    
    std::array<cv::Point2f, 4> OrderPointsClockwise(const std::vector<cv::Point2f>& pts) const;

    std::vector<OCRBox> ocr_det_out;

    private:

    DetResizeMeta Mate;

    
};


//负责对 det 模块传递过来的文本区域进行识别，输出文本内容
class detect_rec_ppocr : public detector
{
    public:
    ov::Core core_;
    ov::CompiledModel model_;
    ov::InferRequest infer_request_;
    ov::Tensor input_tensor_;
    ov::Tensor output_tensor_;

    void preprocess(cv::Mat &input_img)  override;
    void inference() override;
    void postprocess() override;
    


    private:


};

//负责对文本区域进行方向分类，输出文本方向（可以不用，就是用来检查文本是否被倒置的）
class detect_cls_ppocr : public detector
{
    public:



    private:


};
