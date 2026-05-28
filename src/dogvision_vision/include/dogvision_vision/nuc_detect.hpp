#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>
#include <openvino/openvino.hpp> // OpenVINO 2025 接口
#include <string>

#include <dogvision_vision/common_structs.h>
#include <dogvision_vision/detector.hpp>

class detect_oponvino : public detector
{

public:
        /**
         * @brief 构造 YOLO OpenVINO 检测器。
         * @param config 应用配置；为 nullptr 时仅用于加载配置。
         * @retval 无
         */
        detect_oponvino(Appconfig* config) : detector(config) {}
        //~detect_oponvino() override = default;

        ov::Core core_;
        ov::CompiledModel model_;
        ov::InferRequest infer_request_;
        ov::Tensor input_tensor_;
        ov::Tensor output_tensor_;

        /**
         * @brief 加载并编译 YOLO OpenVINO 模型。
         * @param 无
         * @retval bool 模型初始化成功时返回 true。
         */
        bool inference_init(void);

        /**
         * @brief 对单张图像执行 YOLO 检测。
         * @param input_img 输入 BGR 图像。
         * @param res 输出检测结果。
         * @retval bool 推理返回至少一个检测结果时返回 true。
         */
        bool yolo_deect_run(cv::Mat &input_img, std::vector<Detection> &res);

        /**
         * @brief 返回最近一次 NMS 结果。
         * @param 无
         * @retval const std::vector<Detection>& 缓存检测结果的引用。
         */
        const std::vector<Detection> &get_nms_results() const;
        
        

private:
        /**
         * @brief 向检测器基类流程返回 YOLO 结果。
         * @param 无
         * @retval const std::vector<Detection>* 指向缓存 NMS 结果的指针。
         */
        const std::vector<Detection>* yolo_results_ptr() const override { return &nms_results_; }

        // letterbox 预处理相关
        int input_width_;  // YOLO输入宽度
        int input_height_; // YOLO输入高度

        // 存储缩放信息用于后处理坐标还原
        float scale_ = 1.0f; // 缩放比例
        int pad_w_ = 0;      // 水平填充
        int pad_h_ = 0;      // 垂直填充
                            

        ov::element::Type input_precision_ = ov::element::f32; // 模型输入精度


        /**
         * @brief 保持宽高比缩放并使用 letterbox 填充。
         * @param src 源图像。
         * @param target_w 目标宽度。
         * @param target_h 目标高度。
         * @retval cv::Mat letterbox 处理后的图像。
         */
        cv::Mat letterbox(const cv::Mat &src, int target_w, int target_h);

        /**
         * @brief 将单张图像转换为模型输入张量。
         * @param input_img 输入 BGR 图像。
         * @retval void
         */
        void preprocess(cv::Mat &input_img) override;

        /**
         * @brief 执行一次同步 OpenVINO 推理请求。
         * @param 无
         * @retval void
         */
        void inference() override;

        /**
         * @brief 解码并抑制原始模型输出。
         * @param 无
         * @retval void
         */
        void postprocess() override;

        /**
         * @brief 将 YOLO 输出张量解码为原始候选框。
         * @param 无
         * @retval void
         */
        void decode_output(void);

        /**
         * @brief 执行按类别的非极大值抑制。
         * @param 无
         * @retval void
         */
        void nms(void);

        // 存储 decode 后的原始候选框（NMS 前）
        std::vector<cv::Rect2f> boxes_raw_; // [x1,y1,w,h] 还原到原图
        std::vector<float> scores_raw_;     // 最高类别置信度
        std::vector<int> class_ids_raw_;    // 类别ID

        // 最终 NMS 结果（供外部读取）
        std::vector<Detection> nms_results_;
        // 模型初始化时预读的输出张量信息（避免每帧重复解析）
        int out_num_anchors_ = 0;  // anchor数量，如8400
        int out_num_classes_ = 0;  // 类别数
        int out_num_channels_ = 0; // 4 + num_classes

};
