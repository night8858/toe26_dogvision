#include <ros/ros.h>
#include <std_msgs/String.h>

#include <atomic>
#include <iostream>
#include <string>
#include <thread>

// ============================================================
//  arm_cmd_terminal_node  终端测试节点
// ------------------------------------------------------------
//  启动后在终端等待用户输入命令（回车确认），支持两类指令：
//
//  普通指令 -> 发布到 /arm/mission_cmd（高层指令）
//    STOW,ALL         -> 所有臂收起
//    START,ALL        -> 所有臂到启动位置
//    PICK,ALL         -> 所有臂到吸取位置
//    PLACE,LF         -> LF 臂到放置位置
//    VALVE,0,ON       -> 打开电磁阀 0
//    PUMP,ON,2500     -> 启动气泵
//    PLACE_END        -> 放置结束
//
//  $ 前缀指令 -> 发布到 /arm_internation/cmd（低层指令）
//    $LF,X:10,Y:20    -> 直接控制机械臂
//    $V,1,ON          -> 直接控制电磁阀
//    $P,ON,2500       -> 直接控制气泵
//
//  输入 "quit" 或 "exit" 退出节点。
// ============================================================

int main(int argc, char **argv)
{
    ros::init(argc, argv, "arm_cmd_terminal_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string cmd_topic = "/arm_internation/cmd";
    std::string mission_topic = "/arm/mission_cmd";
    pnh.param<std::string>("cmd_topic", cmd_topic, cmd_topic);
    pnh.param<std::string>("mission_topic", mission_topic, mission_topic);

    ros::Publisher cmd_pub = nh.advertise<std_msgs::String>(cmd_topic, 10);
    ros::Publisher mission_pub = nh.advertise<std_msgs::String>(mission_topic, 10);

    ros::Duration(0.5).sleep();

    ROS_INFO_STREAM("[arm_cmd_terminal_node] Ready.");
    ROS_INFO_STREAM("[arm_cmd_terminal_node]   mission pub -> " << mission_topic);
    ROS_INFO_STREAM("[arm_cmd_terminal_node]   low-level cmd pub -> " << cmd_topic);
    ROS_INFO_STREAM("[arm_cmd_terminal_node] \"help\" 查看帮助，\"quit\"/\"exit\" 退出");

    std::atomic<bool> running{true};

    auto print_help = []() {
        std::cout << "\n"
                  << "  ╔══════════════════════════════════════╗\n"
                  << "  ║  机械臂控制终端                      ║\n"
                  << "  ╚══════════════════════════════════════╝\n"
                  << "  ── 高层指令（发往 /arm/mission_cmd）──\n"
                  << "     STOW[,ALL|alias]     收起\n"
                  << "     PICK[,ALL|alias]     吸取位置\n"
                  << "     PLACE,ALL|alias|id,X,Y  放置\n"
                  << "     VALVE/V,id|ALL,ON/OFF  电磁阀\n"
                  << "     PUMP/P,ON[,speed]|OFF  气泵\n"
                  << "  ── 低层指令（加 $ 前缀发往 /arm_internation/cmd）──\n"
                  << "     $LF,X:10,Y:20  控制机械臂\n"
                  << "     $V,id,ON/OFF   控制电磁阀\n"
                  << "     $P,ON,speed    控制气泵\n"
                  << "  ── 系统 ──\n"
                  << "     help    显示本帮助\n"
                  << "     quit    退出\n"
                  << "  ────────────────────────────────────────\n"
                  << std::endl;
    };

    // 用独立线程读取 stdin，避免阻塞 ROS spin
    std::thread input_thread([&]() {
        std::string line;
        while (running.load() && ros::ok()) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) {
                running.store(false);
                break;
            }

            // 去除首尾空白
            const auto start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) continue;
            const auto end = line.find_last_not_of(" \t\r\n");
            line = line.substr(start, end - start + 1);

            if (line == "quit" || line == "exit") {
                ROS_INFO("[arm_cmd_terminal_node] Exiting...");
                running.store(false);
                ros::shutdown();
                break;
            }

            if (line == "help" || line == "h") {
                print_help();
                continue;
            }

            std_msgs::String msg;
            msg.data = line;

            if (!line.empty() && line[0] == '$') {
                // $ 前缀 → 低层指令（去掉 $ 前缀后发送）
                msg.data = line.substr(1);
                cmd_pub.publish(msg);
                ROS_INFO_STREAM("[arm_cmd_terminal_node] [CMD] " << msg.data);
            } else {
                // 普通指令 → 高层指令
                mission_pub.publish(msg);
                ROS_INFO_STREAM("[arm_cmd_terminal_node] [MISSION] " << line);
            }
        }
    });

    // 主线程处理 ROS 回调（此节点无订阅，但保持心跳/日志正常）
    ros::Rate rate(20);
    while (ros::ok() && running.load()) {
        ros::spinOnce();
        rate.sleep();
    }

    if (input_thread.joinable()) {
        input_thread.join();
    }

    return 0;
}
