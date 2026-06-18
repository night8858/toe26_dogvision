#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

/**
 * @brief 打印终端支持的控制命令。
 * @param 无
 * @retval void
 */
static void print_help()
{
    std::cout << "\n"
              << "  机械臂控制终端\n"
              << "  高层指令（发往 /arm/mission_cmd）\n"
              << "    STOW[,ALL|alias]        收起\n"
              << "    PICK[,ALL|alias]        吸取位置\n"
              << "    PLACE,ALL|alias|id,X,Y  放置\n"
              << "    VALVE/V,id|ALL,ON/OFF   电磁阀\n"
              << "    PUMP/P,ON[,speed]|OFF   气泵\n"
              << "  低层指令（加 $ 前缀发往 /arm_internation/cmd）\n"
              << "    启动 4DOF: ros2 launch dogvision_arm arm_test.launch protocol:=4dof\n"
              << "    AA 平面臂: $LF,X:10,Y:20\n"
              << "    BB 4DOF:  $4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4\n"
              << "    BB 4DOF:  $4POSE,R,0.1,0.2,0.3,0.4\n"
              << "    BB 4DOF:  $4ACT,0/1\n"
              << "    BB 4DOF:  $4PICK,L,0.45,0.42,-0.21\n"
              << "    BB 4DOF:  $4PLACE1,R,0.45,-0.40,-0.21\n"
              << "    BB 4DOF:  $4PLACE2,L,X:0.45,Y:0.40,Z:0.04\n"
              << "    BB 4DOF:  $4PUTBACK,L / $4GETBACK,R\n"
              << "    BB 4DOF:  $4PICKALL,0.45,0.42,-0.21,0.45,-0.42,-0.21\n"
              << "    BB 4DOF:  $4PUTBACKALL\n"
              << "    BB 4DOF:  $START,0,0,0      带初始偏移启动(mm)\n"
              << "    $V,id,ON/OFF\n"
              << "    $P,ON,speed\n"
              << "  系统\n"
              << "    help    显示本帮助\n"
              << "    quit    退出\n"
              << std::endl;
}

/**
 * @brief 去除字符串首尾的 ASCII 空白字符。
 * @param line 输入字符串。
 * @retval std::string 去除首尾空白后的字符串。
 */
static std::string trim_line(const std::string& line)
{
    const auto start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
    {
        return {};
    }
    const auto end = line.find_last_not_of(" \t\r\n");
    return line.substr(start, end - start + 1);
}

/**
 * @brief 运行 ROS2 终端命令发布节点。
 *
 * @section 架构设计
 * 本节点提供人机交互界面（stdin 终端输入），将用户命令路由到两个不同层级的话题：
 *
 *   stdin 输入
 *      │
 *      ├── 以 '$' 开头 ──▶ /arm_internation/cmd      （低层协议指令）
 *      │                   例：$LF,X:10,Y:20
 *      │                       $4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4
 *      │                       $V,1,ON
 *      │
 *      └── 无前缀 ──────▶ /arm/mission_cmd          （高层任务指令）
 *                          例：STOW,ALL
 *                              PICK,RF
 *                              PLACE,ALL
 *
 * $ 前缀的设计意图：
 * - 区分"调试/手动控制"（低层）与"任务编排"（高层）两类使用场景
 * - 低层命令直接透传 arm_internation 协议引擎，不做任务级语义解析
 * - 高层命令由 arm_mission_node 拆解为一系列低层命令序列
 *
 * @section 输入线程模型
 * std::getline() 在独立线程中阻塞等待用户输入，通过 atomic<bool> running
 * 与主 ROS 线程同步退出。避免阻塞 spin_some() 导致节点无法响应 ROS 事件。
 *
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @retval int 进程退出码。
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("arm_cmd_terminal_node");
    auto logger = node->get_logger();

    node->declare_parameter<std::string>("cmd_topic", "/arm_internation/cmd");
    node->declare_parameter<std::string>("mission_topic", "/arm/mission_cmd");
    const std::string cmd_topic = node->get_parameter("cmd_topic").as_string();
    const std::string mission_topic = node->get_parameter("mission_topic").as_string();

    auto cmd_pub = node->create_publisher<std_msgs::msg::String>(cmd_topic, rclcpp::QoS(10));
    auto mission_pub = node->create_publisher<std_msgs::msg::String>(mission_topic, rclcpp::QoS(10));

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    RCLCPP_INFO(logger, "[arm_cmd_terminal_node] Ready.");
    RCLCPP_INFO(logger, "[arm_cmd_terminal_node]   mission pub -> %s", mission_topic.c_str());
    RCLCPP_INFO(logger, "[arm_cmd_terminal_node]   low-level cmd pub -> %s", cmd_topic.c_str());
    RCLCPP_INFO(logger, "[arm_cmd_terminal_node] \"help\" 查看帮助，\"quit\"/\"exit\" 退出");

    std::atomic<bool> running{true};
    std::thread input_thread([&]() {
        std::string line;
        while (running.load() && rclcpp::ok())
        {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line))
            {
                running.store(false);
                break;
            }

            line = trim_line(line);
            if (line.empty())
            {
                continue;
            }

            if (line == "quit" || line == "exit")
            {
                RCLCPP_INFO(logger, "[arm_cmd_terminal_node] Exiting...");
                running.store(false);
                rclcpp::shutdown();
                break;
            }

            if (line == "help" || line == "h")
            {
                print_help();
                continue;
            }

            std_msgs::msg::String msg;
            msg.data = line;
            if (!line.empty() && line[0] == '$')
            {
                msg.data = line.substr(1);
                cmd_pub->publish(msg);
                RCLCPP_INFO(logger, "[arm_cmd_terminal_node] [CMD] %s", msg.data.c_str());
            }
            else
            {
                mission_pub->publish(msg);
                RCLCPP_INFO(logger, "[arm_cmd_terminal_node] [MISSION] %s", line.c_str());
            }
        }
    });

    rclcpp::WallRate rate(20);
    while (rclcpp::ok() && running.load())
    {
        rclcpp::spin_some(node);
        rate.sleep();
    }

    running.store(false);
    if (input_thread.joinable())
    {
        input_thread.join();
    }
    rclcpp::shutdown();
    return 0;
}
