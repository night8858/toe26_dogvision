#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>
#include <openvino/openvino.hpp>  // OpenVINO 2025 API
#include <string>

#include "common_structs.h"
#include "detector.hpp"


class detect_oponvino : public detector
 {

    public:
            ov::Core core_;
            ov::CompiledModel model_;
            ov::InferRequest infer_request_;
            ov::Tensor input_tensor_;
            ov::Tensor output_tensor_;

            bool inference_init(void);

            void preprocess() override;
            void inference() override;
            void postprocess() override;

    private:
            // letterbox 预处理相关
            int input_width_  ;   // YOLO输入宽度
            int input_height_ ;  // YOLO输入高度
            
            // 存储缩放信息用于后处理坐标还原
            float scale_ = 1.0f;      // 缩放比例
            int pad_w_ = 0;           // 水平填充
            int pad_h_ = 0;           // 垂直填充
                        // 模型输入精度
            ov::element::Type input_precision_ = ov::element::f32;
                        // letterbox 预处理函数
            cv::Mat letterbox(const cv::Mat& src, int target_w, int target_h);

            void decode_output(void);
            void nms(void);

            // 存储 decode 后的原始候选框（NMS 前）
            std::vector<cv::Rect2f>  boxes_raw_;    // [x1,y1,w,h] 还原到原图
            std::vector<float>       scores_raw_;   // 最高类别置信度
            std::vector<int>         class_ids_raw_; // 类别ID

            // 最终 NMS 结果（供外部读取）
            std::vector<Detection>   nms_results_;

            // 模型初始化时预读的输出张量信息（避免每帧重复解析）
            int   out_num_anchors_  = 0;  // anchor数量，如8400
            int   out_num_classes_  = 0;  // 类别数
            int   out_num_channels_ = 0;  // 4 + num_classes
        };

