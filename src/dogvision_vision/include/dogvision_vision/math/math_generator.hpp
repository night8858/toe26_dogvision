#ifndef MATH_GENERATOR_HPP
#define MATH_GENERATOR_HPP

#include <string>
#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief 四则运算数学题生成器
 *
 * 负责随机生成整数结果的四则运算题、渲染白底黑字图片、
 * 以及按序追加写入 YAML 文件。
 */
class MathGenerator
{
public:
    /**
     * @brief 构造函数
     * @param yaml_path  YAML 输出文件路径（绝对或相对路径）
     * @param min_val    操作数最小值（含）
     * @param max_val    操作数最大值（含）
     * @retval 无
     */
    MathGenerator(const std::string &yaml_path,
                  int min_val = 1, int max_val = 100);

    ~MathGenerator() = default;

    // 禁止拷贝
    MathGenerator(const MathGenerator &) = delete;
    MathGenerator &operator=(const MathGenerator &) = delete;

    /**
     * @brief 生成一道复合四则运算题
     * @param 无
     * @retval std::tuple<std::string, int, int> 题目字符串、正确答案、答案 mod4
     *
     * 格式: a + b * c - d / e + f
     * 同时包含 +、-、*、/ 四种运算符，遵循先乘除后加减。
     * 除法保证分母整除分子，结果必为整数。
     */
    std::tuple<std::string, int, int> generateProblem();

    /**
     * @brief 将题目文字渲染为白底黑字图片
     * @param text  要显示的文本（如 "12 + 8 * 3 - 4 / 2 + 5 = "）
     * @retval cv::Mat  渲染好的 BGR 图像
     *
     * 使用 FONT_HERSHEY_DUPLEX 粗体渲染，文字居中对齐。
     * 自动适配窗口分辨率。
     */
    cv::Mat renderImage(const std::string &text) const;

    /**
     * @brief 将一道题的题目和答案（含 mod4）追加写入 YAML 文件
     * @param problem  题目字符串
     * @param answer   正确答案
     * @param mod4     答案对 4 取模的结果
     * @retval void
     *
     * 每次调用以追加模式写入一条记录（含 id/question/answer/mod4），立即关闭文件。
     */
    void appendToYaml(const std::string &problem, int answer, int mod4);

private:
    std::string yaml_path_;    // YAML 输出路径
    int min_val_;              // 操作数下限
    int max_val_;              // 操作数上限
    int problem_index_;        // 题目序号（从 1 开始）
    bool yaml_header_written_; // YAML 文件头是否已写入

    /**
     * @brief 获取适合当前屏幕的字体缩放比例
     * @param image_width  图像宽度
     * @retval double      字体缩放系数
     */
    double getFontScale(int image_width) const;
};

#endif // MATH_GENERATOR_HPP
