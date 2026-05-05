#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <string>
#include <tuple>
#include <chrono>

#include "dogvision_Math_generator/math_generator.hpp"

int main(int argc, char **argv)
{
    // ── ROS 初始化 ──
    ros::init(argc, argv, "math_generator_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // ── 读取参数 ──
    std::string yaml_path;
    int min_val, max_val, interval_sec;

    // YAML 输出路径：默认放在 package 的 result/ 目录下
    std::string pkg_path = ros::package::getPath("dogvision_Math_generator");
    pnh.param<std::string>("yaml_path", yaml_path,
                           pkg_path + "/result/math_results.yaml");
    pnh.param<int>("min_val",    min_val,    1);
    pnh.param<int>("max_val",    max_val,    100);
    pnh.param<int>("interval",   interval_sec, 10);

    ROS_INFO("Math Generator Node started");
    ROS_INFO("  YAML output : %s", yaml_path.c_str());
    ROS_INFO("  number range: [%d, %d]", min_val, max_val);
    ROS_INFO("  interval    : %d sec", interval_sec);
    ROS_INFO("  Press Q or ESC to exit.");

    // ── 初始化生成器 ──
    MathGenerator generator(yaml_path, min_val, max_val);

    // ── 创建全屏窗口 ──
    const std::string win_name = "Math Generator";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::setWindowProperty(win_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    // ── 主循环：每 interval 秒生成一道新题 ──
    auto last_gen_time = std::chrono::steady_clock::now();
    bool first_frame = true;

    while (ros::ok())
    {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           now - last_gen_time)
                           .count();

        if (first_frame || elapsed >= interval_sec)
        {
            // 生成题目（复合四则运算，包含 +-*/）
            auto [problem, answer, mod4] = generator.generateProblem();

            // 渲染白底黑字图片
            cv::Mat img = generator.renderImage(problem);

            // 全屏显示
            cv::imshow(win_name, img);

            // 追加写入 YAML（含 id / answer / mod4）
            generator.appendToYaml(problem, answer, mod4);

            ROS_INFO("[%s] answer=%d, mod4=%d", problem.c_str(), answer, mod4);

            last_gen_time = now;
            first_frame   = false;
        }

        // 键盘检测 — 100ms 轮询，不阻塞 10s 周期
        int key = cv::waitKey(100);
        if (key == 'q' || key == 'Q' || key == 27) // 27 = ESC
        {
            ROS_INFO("Exit signal received. Shutting down.");
            break;
        }

        ros::spinOnce();
    }

    // ── 清理 ──
    cv::destroyAllWindows();
    ROS_INFO("Math Generator Node finished.");
    return 0;
}