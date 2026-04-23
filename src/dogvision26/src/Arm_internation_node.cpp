#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "internation/arm_internation.hpp"

// ============================================================
//  Arm_internation_node 使用说明
// ------------------------------------------------------------
//  1) 启动后会按 hw_id 在 /dev/ttyACM* 中查找设备并连接（失败重试）
//  2) 订阅 cmd_topic（默认 /arm_internation/cmd）字符串命令
//  3) 每次收到命令后，交给 arm_internation::handle_text_command 解析
//  4) 以 20Hz 发布 /arm_internation/data 全量状态数据
//  5) 主循环持续调用 receive_once() 以接收下位机反馈并更新内部状态
//
//  解码比例参数：
//    ~pos_scale    : 机械臂坐标 raw->float 比例（默认 0.01）
//    ~angle_scale  : 云台角度 raw->float 比例（默认 0.01）
//
//  支持命令示例：
//    RL,X:10,Y:10   -> 控制机械臂
//    G,0,0          -> 控制云台 yaw/pitch
//    V,1            -> 翻转电磁阀 1
//    V,1,ON         -> 打开电磁阀 1
// ============================================================

int main(int argc, char **argv)
{
    // 初始化 ros
    ros::init(argc, argv, "Arm_internation_node");

    // nh: 公共命名空间句柄；pnh: 私有命名空间句柄（~）
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // 串口通信对象：负责串口连接、协议收发、文本命令解析。
    arm_internation my_arm;

    // ---------------- 参数区 ----------------
    // 可通过 rosparam 覆盖：
    //   ~hw_id
    //   ~baud_rate
    //   ~cmd_topic
    std::string hw_id = "0483:5740";
    int baud_rate = 115200;
    std::string cmd_topic = "/arm_internation/cmd";
    std::string data_topic = "/arm_internation/data";
    double pos_scale = 0.01;
    double angle_scale = 0.01;

    pnh.param<std::string>("hw_id", hw_id, hw_id);
    pnh.param<int>("baud_rate", baud_rate, baud_rate);
    pnh.param<std::string>("cmd_topic", cmd_topic, cmd_topic);
    pnh.param<std::string>("data_topic", data_topic, data_topic);
    pnh.param<double>("pos_scale", pos_scale, pos_scale);
    pnh.param<double>("angle_scale", angle_scale, angle_scale);

    // ---------------- 串口连接区 ----------------
    ROS_INFO_STREAM("[Arm_internation_node] wait for ttyACM* with hw_id=" << hw_id
                    << ", baud=" << baud_rate);

    // 阻塞直到连接成功。第三个参数为重试周期（毫秒）。
    my_arm.open_by_HWid(hw_id, baud_rate, 1000);
    my_arm.set_decode_scale(static_cast<float>(pos_scale), static_cast<float>(angle_scale));

    // ---------------- 话题订阅区 ----------------
    // 收到字符串命令后统一交给通信类解析。
    ros::Subscriber cmd_sub = nh.subscribe<std_msgs::String>(
        cmd_topic, 20,
        [&](const std_msgs::String::ConstPtr& msg) {
            // 若格式错误或发送失败，会返回 false 并打印告警。
            if (!my_arm.handle_text_command(msg->data)) {
                ROS_WARN_STREAM("[Arm_internation_node] invalid cmd: " << msg->data);
            }
        }
    );

    // ---------------- 数据发布区 ----------------
    // 发布格式（单行）：
    // LF:x,y;RF:x,y;LB:x,y;RB:x,y;YAW:v;PITCH:v;VALVE_BITS:b1;MICRO_BITS:b2
    // 其中 VALVE_BITS / MICRO_BITS 分别对应协议 [20]/[21] 与 [22]/[23] 的位语义。
    ros::Publisher data_pub = nh.advertise<std_msgs::String>(data_topic, 20);
    const ros::Duration publish_period(1.0 / 20.0);  // 20Hz
    ros::Time next_publish_time = ros::Time::now();

    // ---------------- 主循环区 ----------------
    // 200Hz 原则上足够覆盖串口反馈处理与 ROS 回调调度。
    ros::Rate loop_rate(200);
    
    while (ros::ok())
    {
        // 读取并解析串口数据（成功解析时会刷新内部状态缓存）。
        my_arm.receive_once();

        // 按 20Hz 发布当前最新状态快照。
        const ros::Time now = ros::Time::now();
        if (now >= next_publish_time) {
            const ArmEndPosFloat lf = my_arm.get_arm_pos_float(0);
            const ArmEndPosFloat rf = my_arm.get_arm_pos_float(1);
            const ArmEndPosFloat lb = my_arm.get_arm_pos_float(2);
            const ArmEndPosFloat rb = my_arm.get_arm_pos_float(3);
            const GimbalAngleFloat gim = my_arm.get_gimbal_float();
            const SensorStatus sensor = my_arm.get_sensor();

            int valve_bits = 0;
            int micro_bits = 0;
            
            // 4 位电磁阀 + 4 位微动开关的位掩码
            for (int i = 0; i < 4; ++i) {
                if (sensor.valve[i]) {
                    valve_bits |= (1 << i);
                }
                if (sensor.microswitch[i]) {
                    micro_bits |= (1 << i);
                }
            }

            std_msgs::String msg;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3)
                << "LF:" << lf.x << "," << lf.y
                << ";RF:" << rf.x << "," << rf.y
                << ";LB:" << lb.x << "," << lb.y
                << ";RB:" << rb.x << "," << rb.y
                << ";YAW:" << gim.yaw
                << ";PITCH:" << gim.pitch
                << ";VALVE_BITS:" << valve_bits
                << ";MICRO_BITS:" << micro_bits;
            msg.data = oss.str();
            data_pub.publish(msg);

            next_publish_time = now + publish_period;
            ROS_INFO_STREAM("[Arm_internation_node] published state: " << msg.data);
        }

        // 处理订阅回调（执行命令解析与发送）。
        ros::spinOnce();
        loop_rate.sleep();
    }

    // 显式保留变量，避免部分编译器/配置下的未使用告警。
    (void)cmd_sub;
    (void)data_pub;

    // 正常退出时关闭串口。
    my_arm.close();
    return 0;

}