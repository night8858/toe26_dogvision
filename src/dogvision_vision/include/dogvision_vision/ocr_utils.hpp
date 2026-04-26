#pragma once

#include <opencv2/opencv.hpp>
#include <string>

#include <dogvision_vision/common_structs.h>

// ============================================================
//  ocr_utils.hpp
//  ppocr_node 中的 OCR 辅助工具函数
//  职责：图像裁剪、结果可视化、算术表达式解析
// ============================================================

/// 从原图中用透视变换裁剪4点文字区域
/// 若裁剪后竖排（高 > 宽 × 1.5），自动旋转90°使其横排
cv::Mat crop_text_region(const cv::Mat& src, const OCRBox& box);

/// 在图像上绘制4点检测框及识别文本标签
void draw_ocr_result(cv::Mat& vis, const OCRBox& box, const OCRRecResult& rec);

/// 从 OCR 识别的合并文本中提取并计算第一个算术表达式
/// 支持符号：+ - * / × ÷ 以及中文全角字符，自动归一化后解析
/// @param text     OCR 合并输出字符串
/// @param result   输出：计算结果（double）
/// @param expr_str 输出：识别到的原始表达式字符串
/// @return true 表示成功找到并计算了表达式
bool parse_simple_expr(const std::string& text, double& result, std::string& expr_str);

/// 以独立 OpenCV 窗口展示算术计算结果
/// @param expr_str  识别到的表达式字符串
/// @param mod_result 计算结果对 4 取模（非负）
void show_result_window(const std::string& expr_str, int mod_result);

/// 在输入图像中定位"白底黑字算术题区域"并返回其矩形
/// 策略：HSV 白色掩码 → 形态学填孔去噪 → 轮廓筛选（面积、宽高比、白色覆盖率）
/// 返回空 Rect2f（area()==0）表示未找到合适区域
/// @note 裁剪示例：cv::Mat roi_img = img(cv::Rect(find_math_proble(img)));
cv::Rect2f find_math_proble(const cv::Mat& input);
