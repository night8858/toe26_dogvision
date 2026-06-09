#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <tuple>

#include <dogvision_vision/math/math_generator.hpp>

/**
 * @brief 运行 ROS2 数学题生成节点。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @retval int 进程退出码。
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("math_generator_node");
    auto logger = node->get_logger();

    const std::string share_dir = ament_index_cpp::get_package_share_directory("dogvision_vision");
    node->declare_parameter<std::string>("yaml_path", share_dir + "/data/math_generator/math_results.yaml");
    node->declare_parameter<int>("min_val", 1);
    node->declare_parameter<int>("max_val", 100);
    node->declare_parameter<int>("interval", 10);
    node->declare_parameter<int>("canvas_width", 1920);
    node->declare_parameter<int>("canvas_height", 1080);

    const std::string yaml_path = node->get_parameter("yaml_path").as_string();
    const int min_val = static_cast<int>(node->get_parameter("min_val").as_int());
    const int max_val = static_cast<int>(node->get_parameter("max_val").as_int());
    const int interval_sec = static_cast<int>(node->get_parameter("interval").as_int());
    const int canvas_w = static_cast<int>(node->get_parameter("canvas_width").as_int());
    const int canvas_h = static_cast<int>(node->get_parameter("canvas_height").as_int());

    RCLCPP_INFO(logger, "Math Generator Node started");
    RCLCPP_INFO(logger, "  YAML output : %s", yaml_path.c_str());
    RCLCPP_INFO(logger, "  number range: [%d, %d]", min_val, max_val);
    RCLCPP_INFO(logger, "  interval    : %d sec", interval_sec);
    RCLCPP_INFO(logger, "  canvas      : %d x %d", canvas_w, canvas_h);
    RCLCPP_INFO(logger, "  Press Q or ESC to exit.");

    MathGenerator generator(yaml_path, min_val, max_val);

    const std::string win_name = "Math Generator";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    cv::setWindowProperty(win_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    auto last_gen_time = std::chrono::steady_clock::now();
    bool first_frame = true;

    while (rclcpp::ok())
    {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_gen_time).count();

        if (first_frame || elapsed >= interval_sec)
        {
            auto [problem, answer, mod4] = generator.generateProblem();
            cv::Mat img = generator.renderImage(problem, canvas_w, canvas_h);
            cv::imshow(win_name, img);
            generator.appendToYaml(problem, answer, mod4);
            RCLCPP_INFO(logger, "[%s] answer=%d, mod4=%d", problem.c_str(), answer, mod4);

            last_gen_time = now;
            first_frame = false;
        }

        const int key = cv::waitKey(100);
        if (key == 'q' || key == 'Q' || key == 27)
        {
            RCLCPP_INFO(logger, "Exit signal received. Shutting down.");
            break;
        }

        rclcpp::spin_some(node);
    }

    cv::destroyAllWindows();
    RCLCPP_INFO(logger, "Math Generator Node finished.");
    rclcpp::shutdown();
    return 0;
}
