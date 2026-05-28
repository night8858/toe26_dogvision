#pragma once

#include <opencv2/opencv.hpp>
#include <string>

#include <dogvision_vision/common_structs.h>

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
 * @brief 在本地 OpenCV 窗口中显示 OCR 算术结果。
 * @param expr_str 识别到的表达式字符串。
 * @param mod_result 非负的模 4 结果。
 * @retval void
 */
void show_result_window(const std::string& expr_str, int mod_result);

/**
 * @brief 在输入图像中定位白底算术题区域。
 * @param input 源 BGR 图像。
 * @retval cv::Rect2f 外接矩形；空矩形表示未找到区域。
 */
cv::Rect2f find_math_proble(const cv::Mat& input);

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
