#include <ros/ros.h>
#include <std_msgs/String.h>

#include <atomic>
#include <iostream>
#include <string>
#include <thread>

// ============================================================
//  arm_cmd_terminal_node  使用说明
// ------------------------------------------------------------
//  1) 启动后在终端等待用户输入一行命令（回车确认）
//  2) 将命令原文发布到 /arm_internation/cmd（可通过 ~cmd_topic 覆盖）
//  3) 输入 "quit" 或 "exit" 可正常退出节点
//
//  示例命令（与 Arm_internation_node 协议一致）：
//    RL,X:10,Y:10   -> 控制机械臂
//    G,0,0          -> 控制云台 yaw/pitch
//    V,1            -> 翻转电磁阀 1
//    V,1,ON         -> 打开电磁阀 1
// ============================================================

int main(int argc, char **argv)
{
    ros::init(argc, argv, "arm_cmd_terminal_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string cmd_topic = "/arm_internation/cmd";
    pnh.param<std::string>("cmd_topic", cmd_topic, cmd_topic);

    ros::Publisher cmd_pub = nh.advertise<std_msgs::String>(cmd_topic, 10);

    // 稍作等待，确保发布者向 master 注册完毕
    ros::Duration(0.5).sleep();

    ROS_INFO_STREAM("[arm_cmd_terminal_node] Ready. Publishing to: " << cmd_topic);
    ROS_INFO_STREAM("[arm_cmd_terminal_node] Type a command and press Enter. "
                    "Input \"quit\" or \"exit\" to stop.");

    std::atomic<bool> running{true};

    // 用独立线程读取 stdin，避免阻塞 ROS spin
    std::thread input_thread([&]() {
        std::string line;
        while (running.load() && ros::ok()) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) {
                // EOF（Ctrl+D）
                running.store(false);
                break;
            }

            // 去除首尾空白
            const auto start = line.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) {
                continue;  // 空行跳过
            }
            const auto end = line.find_last_not_of(" \t\r\n");
            line = line.substr(start, end - start + 1);

            if (line == "quit" || line == "exit") {
                ROS_INFO("[arm_cmd_terminal_node] Exiting...");
                running.store(false);
                ros::shutdown();
                break;
            }

            std_msgs::String msg;
            msg.data = line;
            cmd_pub.publish(msg);
            ROS_INFO_STREAM("[arm_cmd_terminal_node] Sent: " << line);
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
