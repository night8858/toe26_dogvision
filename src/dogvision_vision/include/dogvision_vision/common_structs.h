/*
* common_structs.h
* Created on: 20230605
* Author: sumang
* Description: some common structs
*/
#ifndef COMMON_STRUCTS_H_
#define COMMON_STRUCTS_H_

#include <string>
#include <opencv2/opencv.hpp>
/**
 * @brief 检测器全局参数配置结构体
 *
 * 包含 YOLO 模型路径、输入尺寸、NMS 阈值以及 PPOCR 检测/识别/分类参数。
 * 通过 JSON 配置文件加载，部分字段也可在运行时动态修改。
 */
typedef struct 
{
    // ── 模型文件路径 ──
        std::string bin_file_path;       ///< OpenVINO IR .bin 权重文件路径
        std::string xml_file_path;       ///< OpenVINO IR .xml 模型结构文件路径

        std::string yolo_device = "CPU"; ///< YOLO 推理设备（"CPU" / "GPU" 等）

    // ── 网络输入尺寸（NCHW 格式） ──
        int batch_size;  ///< 批次大小
        int h;           ///< 输入高度（像素）
        int w;           ///< 输入宽度（像素）
        int c;           ///< 输入通道数

    // ── 图像属性 ──
        int type;    ///< 色彩格式：0=rgb, 1=bgr, 2=yuv, 3=yv12, 4=bayerrg8 …
        int width;   ///< 图像宽度（像素）
        int height;  ///< 图像高度（像素）

    // ── 阈值参数 ──
        float nms_thresh;        ///< NMS 去重的 IoU 阈值
        float bbox_conf_thresh;  ///< 边界框置信度阈值（低于此值则丢弃）
        float merge_thresh;      ///< 多帧融合时使用的阈值

    // ── 类别数量 ──
        int classes; ///< 检测目标类别总数

    
    // ── YOLO anchor 参数 ──
        std::vector<float> a1; ///< 第 1 组 anchor 尺寸
        std::vector<float> a2; ///< 第 2 组 anchor 尺寸
        std::vector<float> a3; ///< 第 3 组 anchor 尺寸
        std::vector<float> a4; ///< 第 4 组 anchor 尺寸
    
    float z_scale;       ///< 深度方向缩放系数（左/主相机）
    float z_scale_right; ///< 深度方向缩放系数（右相机）

///////////// PPOCR 参数 ////////////////

    // ── 模型路径 ──
    std::string ppocr_det_model_path; ///< 文本检测模型路径（旧 .pdmodel/ONNX 单文件格式）
    std::string ppocr_rec_model_path; ///< 文本识别模型路径（旧 .pdmodel/ONNX 单文件格式）
    std::string ppocr_cls_model_path; ///< 文本方向分类模型路径（.pdmodel 格式，可选）
    std::string ppocr_det_model_xml_path; ///< 文本检测 OpenVINO IR .xml 路径
    std::string ppocr_det_model_bin_path; ///< 文本检测 OpenVINO IR .bin 路径
    std::string ppocr_rec_model_xml_path; ///< 文本识别 OpenVINO IR .xml 路径
    std::string ppocr_rec_model_bin_path; ///< 文本识别 OpenVINO IR .bin 路径

    // ── 推理设备 ──
    std::string det_device = "CPU"; ///< 文本检测推理设备
    std::string rec_device = "CPU"; ///< 文本识别推理设备
    std::string cls_device = "CPU"; ///< 方向分类推理设备

    bool use_angle_cls = false; ///< 是否启用文本方向分类

    // ── 检测（det）参数 ──
    int det_limit_side_len = 960;     ///< 检测输入图像长边限制值（像素）
    std::string det_limit_type = "max"; ///< 缩放策略："max"=按最长边, "min"=按最短边
    std::string det_box_type = "quad";  ///< 输出文本框类型："quad"=四点

    float det_db_thresh       = 0.3f;   ///< DB 概率图二值化阈值
    float det_db_box_thresh   = 0.6f;   ///< 检测框置信度阈值（基于框内概率图均值）
    float det_db_unclip_ratio = 1.5f;   ///< DB Unclip 外扩比例

    // ── 识别（rec）参数 ──
    int rec_img_c = 3;         ///< 识别输入通道数
    int rec_img_h = 48;        ///< 识别输入固定高度（像素）
    int rec_img_w = 320;       ///< 识别输入最大宽度（像素）
    int rec_batch_num = 6;     ///< 识别批处理数量

    // ── 方向分类（cls）参数 ──
    int cls_img_c = 3;         ///< 分类输入通道数
    int cls_img_h = 48;        ///< 分类输入固定高度（像素）
    int cls_img_w = 192;       ///< 分类输入最大宽度（像素）
    int cls_batch_num = 6;     ///< 分类批处理数量
    float cls_thresh = 0.9f;   ///< 方向分类置信度阈值

    float drop_score = 0.5f;   ///< 低于此分数的识别结果将被丢弃

    // ── 字典与白名单 ──
    std::string rec_char_dict_path;      ///< 全量识别字典路径（如 PP-OCRv5 inference.yml）
    std::string rec_allowed_chars_path;  ///< 允许输出的数学字符白名单路径

    // ── OCR 数学题后置筛选参数 ──
    bool ocr_math_use_grayscale = false; ///< 是否将 OCR 输入转为三通道灰度图
    bool ocr_roi_enabled = false; ///< 是否只对配置的 OCR ROI 做检测识别
    std::string ocr_roi_mode = "full"; ///< OCR ROI 模式：full/quadrant/ratio
    std::string ocr_roi_quadrant = "full"; ///< OCR ROI 象限：full/top_left/top_right/bottom_left/bottom_right
    cv::Rect2d ocr_roi_rect_ratio{0.0, 0.0, 1.0, 1.0}; ///< 自定义 OCR ROI 比例坐标 x/y/w/h
    cv::Rect2d ocr_left_roi_rect_ratio{0.0, 0.0, 1.0, 1.0}; ///< 左侧任务 OCR ROI 比例坐标 x/y/w/h
    cv::Rect2d ocr_right_roi_rect_ratio{0.0, 0.0, 1.0, 1.0}; ///< 右侧任务 OCR ROI 比例坐标 x/y/w/h
    std::string ocr_roi_runtime_label; ///< 运行时 ROI 标签，用于日志显示 left/right
    double ocr_math_min_surround_white_ratio = 0.50; ///< 最小外围白色比例
    double ocr_math_surround_margin_ratio = 0.50; ///< 外围环带宽度/平均文字高度
    int ocr_math_white_s_max = 110; ///< HSV 白色区域 S 通道上限
    int ocr_math_white_v_min = 50; ///< HSV 白色区域 V 通道下限

    // ── OCR 测试模式窗口显示参数 ──
    bool ocr_test_show_visual = true; ///< test 模式是否显示整帧 OCR 结果窗口
    bool ocr_test_show_ocr_roi = true; ///< test 模式是否显示当前最优算术候选窗口
    bool ocr_test_show_debug_panels = true; ///< test 模式是否显示 OCR 调试拼图窗口

    // ── 类别名称（JSON 中 cls0~cls3 字段） ──
    std::string class0; ///< 第 0 类名称
    std::string class1; ///< 第 1 类名称
    std::string class2; ///< 第 2 类名称
    std::string class3; ///< 第 3 类名称

    // ── YOLO 图像增强参数 ──
    bool yolo_enhance_enabled = true;             ///< 是否启用 YOLO 输入图像增强
    float yolo_enhance_clahe_clip_limit = 2.0f;   ///< CLAHE 对比度限幅
    int yolo_enhance_clahe_tile_grid_size = 8;    ///< CLAHE 网格尺寸（像素块边长）
    float yolo_enhance_saturation_scale = 1.3f;   ///< 饱和度缩放系数

    // ── 输出保存参数 ──
    bool save_ppocr_video = true; ///< 是否保存 PP-OCR 推理可视化视频
    std::string ppocr_video_save_dir; ///< PP-OCR 推理视频保存目录
    double ppocr_video_fps = 20.0; ///< PP-OCR 推理视频帧率
    int max_ppocr_videos = 10; ///< PP-OCR 推理视频最多保留条数
    bool save_ocr_result_images = true; ///< 是否保存 OCR 稳定结果标注图
    std::string ocr_result_image_dir; ///< OCR 稳定结果图保存目录
    int max_ocr_result_images = 30; ///< OCR 稳定结果图最多保留张数
    bool save_yolo_test_video = true; ///< 是否保存 YOLO 准确率测试视频

    bool enable_undistort = true; ///< 是否启用鱼眼去畸变
    float D_matrix[4]; ///< 鱼眼畸变参数 D 矩阵（4×1）
    
}s_detector_params;

/**
 * @brief YOLO 检测结果（对齐到 4 字节，便于 SIMD 拷贝）
 */
struct alignas(float) Detection
{
    float bbox[4];   ///< 边界框 [x, y, width, height]（原图坐标系）
    float conf;      ///< 检测置信度 (0.0 ~ 1.0)
    float class_id;  ///< 类别编号
};

/**
 * @brief 海康相机参数
 */
typedef struct
{
    int device_id; ///< 相机设备编号（枚举索引）
    int width;     ///< 采集宽度（像素）
    int height;    ///< 采集高度（像素）
    int offset_x;  ///< ROI 水平偏移
    int offset_y;  ///< ROI 垂直偏移
    int exposure;  ///< 曝光时间（微秒）

}s_hikcamera_params;

/**
 * @brief USB 相机参数
 */
typedef struct
{
    std::string device_path; ///< USB 相机设备路径（可选，如 /dev/video0）
    int device_id; ///< 相机设备编号
    int width;     ///< 采集宽度（像素）
    int height;    ///< 采集高度（像素）
    int fps;       ///< 目标帧率
    double brightness; ///< 硬件亮度；负值表示不覆盖驱动默认值
    double exposure; ///< 硬件曝光；负值表示启用自动曝光
    double contrast; ///< 硬件对比度；负值表示不覆盖驱动默认值
    double software_brightness_offset; ///< OpenCV 像素亮度偏移；0 表示不处理

}s_usbcamera_params;

/**
 * @brief 应用总配置，包含检测、相机选择、海康相机和 USB 相机参数
 */
typedef struct
{
    s_detector_params detect_config;          ///< 检测器全局参数
    std::string camera_type = "usb";          ///< 当前相机类型（当前仅启用 usb）
    int usb_camera_index = 0;                 ///< USB 相机配置索引（0-3）
    s_hikcamera_params hikcamera_config;      ///< 海康相机参数（当前停用）
    s_usbcamera_params usbcamera_config[4];   ///< USB 相机参数（最多 4 个）

}Appconfig;

/**
 * @brief YOLO 输出层信息
 */
typedef struct
{
    int idx;        ///< 输出层索引
    int stride;     ///< 该层的下采样步长
    int num_anchor; ///< 该层的 anchor 数量
    int num_out;    ///< 该层每个 anchor 的输出维度
}s_OutLayer;

/**
 * @brief 多帧融合时用于存储单个捡取点的合并信息
 */
typedef struct{
    int id;                            ///< 捡取点编号
    std::vector<cv::Point2f> merge_pts;    ///< 合并后的轮廓点集
    std::vector<float> merge_confs;        ///< 各点的置信度
}pick_merge_store;

/**
 * @brief OCR 配置（目前为空占位）
 */
typedef struct  {

}OCRConfig;

/**
 * @brief OCR 四点文本框（左上 → 右上 → 右下 → 左下）
 */
typedef struct  {
    std::array<cv::Point2f, 4> pts; ///< 四个顶点坐标
}OCRBox;

/**
 * @brief OCR 单行文字识别结果
 */
typedef struct  {
    std::string text;  ///< 识别文本内容
    float score = 0.0f;///< 置信度 (0.0 ~ 1.0)
}OCRRecResult;

/**
 * @brief OCR 完整结果：检测框 + 识别文本
 */
typedef struct  {
    OCRBox box;     ///< 文本检测框
    OCRRecResult rec; ///< 识别结果
}OCRItem;

/**
 * @brief 从整帧 OCR 结果中筛选出的单行算术题候选。
 */
typedef struct {
    std::vector<size_t> item_indices; ///< 构成候选的 OCRItem 索引
    std::string raw_text;             ///< OCR 原始拼接文本
    std::string expression;           ///< 归一化后的严格算术表达式
    double result = 0.0;              ///< 表达式计算结果
    float mean_score = 0.0f;          ///< 候选内文字平均识别置信度
    float surround_white_ratio = 0.0f;///< 文字外围环带白色比例
    float rank_score = 0.0f;          ///< 候选综合排序分数
    cv::Rect text_bounds;             ///< 所有候选文字框的合并矩形
    cv::Rect surround_bounds;         ///< 裁剪到图像边界后的环带外矩形
    bool white_pass = false;          ///< 是否通过白色比例门槛
} OCRMathCandidate;

/**
 * @brief 文本检测预处理阶段的缩放元信息
 *
 * 用于后处理时将检测图中的坐标映射回原图。
 */
typedef struct  {
    int src_h ;      ///< 原图高度
    int src_w ;      ///< 原图宽度
    float ratio_h ;  ///< 高度缩放比（resize_h / src_h）
    float ratio_w ;  ///< 宽度缩放比（resize_w / src_w）
    int resize_h ;   ///< 缩放后高度（已对齐 32 的倍数）
    int resize_w ;   ///< 缩放后宽度（已对齐 32 的倍数）
}DetResizeMeta;


#endif
