#include <dogvision_vision/math/math_generator.hpp>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

// ──────────────────────────────────────────────
// 构造函数
// ──────────────────────────────────────────────
MathGenerator::MathGenerator(const std::string &yaml_path,
                             int min_val, int max_val)
    : yaml_path_(yaml_path)
    , min_val_(min_val)
    , max_val_(max_val)
    , problem_index_(0)
    , yaml_header_written_(false)
{
    // 初始化随机种子（仅首次调用有效）
    static bool seeded = false;
    if (!seeded) {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        seeded = true;
    }
}

/**
 * @brief 在闭区间内生成随机整数。
 * @param min_val 下界。
 * @param max_val 上界。
 * @retval int 位于 [min_val, max_val] 范围内的随机值。
 */
static inline int randRange(int min_val, int max_val)
{
    if (min_val > max_val) std::swap(min_val, max_val);
    return min_val + std::rand() % (max_val - min_val + 1);
}

// ──────────────────────────────────────────────
// 生成一道复合四则运算题（四种运算符随机排列）
// 格式如: a + b * c - d / e + f  或  a * b / c + d - e  等
// 始终遵循 先乘除后加减，除法保证整除
// ──────────────────────────────────────────────
std::tuple<std::string, int, int> MathGenerator::generateProblem()
{
    // 使用较小的数值范围使复合运算更友好
    int lo = min_val_;
    int hi = std::min(max_val_, 20);

    // 最大重试次数，防止无法找到合适除数导致死循环
    const int MAX_TRIES = 50;

    for (int attempt = 0; attempt < MAX_TRIES; attempt++)
    {
        // ── 1. 运算符随机排列 (0:+, 1:-, 2:*, 3:/) ──
        std::vector<int> ops = {0, 1, 2, 3};
        static std::mt19937 rng{std::random_device{}()};
        std::shuffle(ops.begin(), ops.end(), rng);

        // ── 2. 生成 5 个初始操作数 ──
        std::vector<int> val(5);
        for (auto &v : val) v = randRange(lo, hi);

        // ── 3. 左到右扫描，按 PEMDAS 求值 ──
        // 维护当前"项"(term)：累积 * 和 /  的结果
        // 遇到 + 或 - 时把 term 推入总和，开始新 term
        int sum = 0;
        int term = val[0];
        bool valid = true;

        // 构建表达式字符串
        std::ostringstream oss;
        oss << val[0];

        for (int i = 0; i < 4; i++)
        {
            int next_val = val[i + 1];

            // 为除法做预处理：确保 term 能被 next_val 整除
            if (ops[i] == 3)  // /
            {
                int abs_term = std::abs(term);
                // 收集 abs_term 在 [lo, hi] 范围内的所有因数
                std::vector<int> divisors;
                if (abs_term > 0)
                {
                    int limit = std::min(hi, abs_term);
                    for (int d = lo; d <= limit; d++)
                    {
                        if (abs_term % d == 0)
                            divisors.push_back(d);
                    }
                }
                // 从可用因数中随机选一个（确保非零）
                int good_divisor = 1;
                if (!divisors.empty())
                    good_divisor = divisors[std::rand() % divisors.size()];
                next_val = good_divisor;
            }

            // 输出运算符和调整后的操作数
            const char op_chars[] = {'+', '-', '*', '/'};
            oss << " " << op_chars[ops[i]] << " " << next_val;

            // 根据运算符合并到 term 或推入 sum
            if (ops[i] == 2)       // *
            {
                term *= next_val;
            }
            else if (ops[i] == 3)  // /
            {
                if (next_val == 0)
                {
                    valid = false;
                    break;
                }
                term /= next_val;
            }
            // 检测中间结果过大 → 重试避免数字失控
            if (std::abs(term) > 5000)
            {
                valid = false;
                break;
            }
            else if (ops[i] == 0)  // +
            {
                sum += term;
                term = next_val;
            }
            else if (ops[i] == 1)  // -
            {
                sum += term;
                term = -next_val;
            }
        }

        if (!valid) continue;

        sum += term;
        int answer = sum;
        int mod4   = ((answer % 4) + 4) % 4;

        oss << " = ";
        problem_index_++;

        return {oss.str(), answer, mod4};
    }

    // 极端情况：多次重试后仍失败（几乎不可能），返回兜底题目
    std::cerr << "generateProblem() exhausted retries, using fallback." << std::endl;
    std::string fallback = "1 + 2 * 3 - 4 / 2 + 5 = ";
    int answer = 1 + 2 * 3 - 4 / 2 + 5;  // = 10
    int mod4   = 10 % 4;                  // = 2
    problem_index_++;
    return {fallback, answer, mod4};
}

// ──────────────────────────────────────────────
// 渲染白底黑字图片（粗体 + 精确居中）
// ──────────────────────────────────────────────
cv::Mat MathGenerator::renderImage(const std::string &text) const
{
    // 获取屏幕分辨率（尝试读取当前窗口尺寸，失败则用 1920×1080 兜底）
    int screen_w = 1920;
    int screen_h = 1080;

    cv::namedWindow("_tmp_resolution", cv::WINDOW_NORMAL);
    cv::setWindowProperty("_tmp_resolution",
                          cv::WND_PROP_FULLSCREEN,
                          cv::WINDOW_FULLSCREEN);
    cv::Rect rect = cv::getWindowImageRect("_tmp_resolution");
    if (rect.width > 0 && rect.height > 0)
    {
        screen_w = rect.width;
        screen_h = rect.height;
    }
    cv::destroyWindow("_tmp_resolution");

    // 创建白色背景
    cv::Mat image(screen_h, screen_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // ── 粗体渲染 ──
    // FONT_HERSHEY_DUPLEX 本身比 SIMPLEX 更粗
    // thickness 再放大一级，达到醒目的黑体效果
    int font_face   = cv::FONT_HERSHEY_DUPLEX;
    double font_scale = getFontScale(screen_w);
    int thickness   = std::max(4, screen_w / 240);

    // ── 精确居中：循环微调直到文字完全位于图像中心区域 ──
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);

    // 水平居中 + 垂直居中（考虑 baseline 偏移）
    int text_x = (screen_w - text_size.width) / 2;
    int text_y = (screen_h + text_size.height - baseline) / 2;

    // ── 白色背景上绘制黑色文字 ──
    cv::putText(image, text,
                cv::Point(text_x, text_y),
                font_face, font_scale,
                cv::Scalar(0, 0, 0),   // 黑色
                thickness,
                cv::LINE_AA);

    return image;
}

// ──────────────────────────────────────────────
// 将题目+答案（含 mod4）追加写入 YAML 文件
// ──────────────────────────────────────────────
void MathGenerator::appendToYaml(const std::string &problem, int answer, int mod4)
{
    std::ofstream ofs(yaml_path_, std::ios::app);
    if (!ofs.is_open())
    {
        std::cerr << "Cannot open YAML file for append: " << yaml_path_ << std::endl;
        return;
    }

    // 首次写入时加文件头
    if (!yaml_header_written_)
    {
        ofs << "math_problems:" << std::endl;
        yaml_header_written_ = true;
    }

    // 写入一条记录（含编号、题目、答案、mod4）
    ofs << "  - id: "       << problem_index_          << std::endl;
    ofs << "    question: \"" << problem << "\""        << std::endl;
    ofs << "    answer: "   << answer                  << std::endl;
    ofs << "    mod4: "     << mod4                    << std::endl;

    ofs.close();
}

// ──────────────────────────────────────────────
// 获取字体缩放比例
// ──────────────────────────────────────────────
double MathGenerator::getFontScale(int image_width) const
{
    // 以 1920 宽度为基准，比例缩放
    // 1920px → 4.0  1080px → 2.5  等
    double scale = static_cast<double>(image_width) / 480.0;
    return std::max(1.0, std::min(scale, 10.0));
}
