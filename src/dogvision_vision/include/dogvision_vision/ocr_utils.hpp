#pragma once

#include <opencv2/opencv.hpp>
#include <string>

#include <dogvision_vision/common_structs.h>

/**
 * @brief 准备实际送入 OCR 的整帧图像。
 * @param input 原始图像，支持 1/3/4 通道。
 * @param use_grayscale 是否转换为三通道灰度图。
 * @retval cv::Mat 与输入同尺寸的三通道 BGR 图。
 */
cv::Mat prepare_ocr_input(const cv::Mat& input, bool use_grayscale);

/**
 * @brief 使用透视变换裁剪四点 OCR 文本区域。
 * @param src 源 BGR 图像。
 * @param box 四点文本区域框。
 * @retval cv::Mat 裁剪后的文本图像；区域无效时返回空矩阵。
 */
cv::Mat crop_text_region(const cv::Mat& src, const OCRBox& box);

/**
 * @brief 在图像上绘制一个 OCR 框和识别标签。
 * @param vis 需要原地修改的可视化图像。
 * @param box OCR 检测框。
 * @param rec 与检测框对应的识别结果。
 * @retval void
 */
void draw_ocr_result(cv::Mat& vis, const OCRBox& box, const OCRRecResult& rec);

/**
 * @brief 从 OCR 文本中提取并计算第一个算术表达式。
 * @param text 合并后的 OCR 文本。
 * @param result 输出表达式计算值。
 * @param expr_str 输出归一化后的表达式字符串。
 * @retval bool 找到并成功计算算术表达式时返回 true。
 */
bool parse_simple_expr(const std::string& text, double& result, std::string& expr_str);

/**
 * @brief 计算 OCR 四点框的轴对齐外接矩形。
 */
cv::Rect ocr_box_bounds(const OCRBox& box, const cv::Size& image_size);

/**
 * @brief 计算文字合并框外围环带中的白色像素比例。
 * @param image 原始彩色图像。
 * @param text_bounds 文字合并矩形。
 * @param average_text_height 候选文字框平均高度。
 * @param config OCR 数学题筛选配置。
 * @param surround_bounds 可选输出裁剪后的环带外矩形。
 */
float calculate_surround_white_ratio(
    const cv::Mat& image,
    const cv::Rect& text_bounds,
    float average_text_height,
    const s_detector_params& config,
    cv::Rect* surround_bounds = nullptr);

/**
 * @brief 从整帧 OCR 结果中组合并筛选单行算术题候选。
 * @return 按通过状态和综合分数从高到低排列的候选。
 */
std::vector<OCRMathCandidate> find_math_candidates(
    const cv::Mat& original_image,
    const std::vector<OCRItem>& items,
    const s_detector_params& config);

/**
 * @brief 在本地 OpenCV 窗口中显示 OCR 算术结果。
 * @param expr_str 识别到的表达式字符串。
 * @param mod_result 非负的模 4 结果。
 * @retval void
 */
void show_result_window(const std::string& expr_str, int mod_result);

/**
 * @brief 在输入图像中定位白底算术题区域。
 * @param input 源 BGR 图像。
 * @param mask_out 可选输出参数：白色区域精确二值掩码（CV_8UC1）。
 *                 只在找到区域时写入，传入 nullptr 可忽略。
 * @param white_s_max  HSV 中 S 通道上限（默认 110，值越大容忍越"近似的白色"）。
 * @param white_v_min  HSV 中 V 通道下限（默认 50，值越小容忍越"暗的白色"）。
 * @retval cv::Rect2f 外接矩形；空矩形表示未找到区域。
 */
cv::Rect2f find_math_proble(const cv::Mat& input,
                            cv::Mat* mask_out = nullptr,
                            int white_s_max = 110,
                            int white_v_min = 50);

/**
 * @brief 按配置图像尺寸初始化鱼眼去畸变映射表。
 * @param image_width 相机图像宽度。
 * @param image_height 相机图像高度。
 * @retval bool 映射表初始化成功时返回 true。
 */
bool init_fisheye_undistort(int image_width, int image_height);

/**
 * @brief 对图像执行鱼眼去畸变。
 * @param input 源图像。
 * @retval cv::Mat 去畸变后的图像；映射表不可用时返回源图克隆；输入为空时返回空图。
 */
cv::Mat undistort_image(const cv::Mat& input);
