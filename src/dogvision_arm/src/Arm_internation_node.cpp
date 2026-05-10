#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <dogvision_arm/arm_internation.hpp>

// ============================================================
//  Arm_internation_node 使用说明
// ------------------------------------------------------------
//  订阅 /arm_internation/cmd，将字符串命令转发到串口下位机，
//  并以 20Hz 发布 /arm_internation/data 全量状态数据。
//
//  串口连接方式（二选一）：
//    1) 按硬件 ID 自动扫描（默认）
//       在 /dev/ttyACM* /dev/ttyUSB* 中匹配 ~hw_id（默认 0483:5740）
//       连接成功后才进入主循环；若设备未插入则等待。
//
//    2) 直接指定串口路径（用于虚拟串口 / 非标准设备）
//       设置 ~port 参数，例如 /dev/pts/3 或 /dev/ttyUSB0
//       此时不会阻塞等待，由主循环自动重连。
//
//  参数：
//    ~hw_id        硬件 ID（默认 0483:5740）
//    ~baud_rate    波特率（默认 115200）
//    ~port         串口路径（可选，设置后绕过 hw_id 扫描）
//    ~cmd_topic    订阅话题（默认 /arm_internation/cmd）
//    ~data_topic   发布话题（默认 /arm_internation/data）
//    ~pos_scale    坐标缩放（默认 0.01）
//    ~angle_scale  角度缩放（默认 0.01）
// ============================================================

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Arm_internation_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    arm_internation my_arm;

    // ---------------- 参数区 ----------------
    std::string hw_id = "0483:5740";
    int baud_rate = 115200;
    std::string serial_port = "";   // 为空时使用 hw_id 自动扫描
    std::string cmd_topic = "/arm_internation/cmd";
    std::string data_topic = "/arm_internation/data";
    double pos_scale = 0.01;
    double angle_scale = 0.01;

    pnh.param<std::string>("hw_id", hw_id, hw_id);
    pnh.param<int>("baud_rate", baud_rate, baud_rate);
    pnh.param<std::string>("port", serial_port, serial_port);
    pnh.param<std::string>("cmd_topic", cmd_topic, cmd_topic);
    pnh.param<std::string>("data_topic", data_topic, data_topic);
    pnh.param<double>("pos_scale", pos_scale, pos_scale);
    pnh.param<double>("angle_scale", angle_scale, angle_scale);

    // ---- 先创建话题，再连接串口 ----
    // 确保即使串口未连接，话题也能被 rostopics list 看到。

    // 订阅低层命令
    ros::Subscriber cmd_sub = nh.subscribe<std_msgs::String>(
        cmd_topic, 20,
        [&](const std_msgs::String::ConstPtr& msg) {
            if (!my_arm.handle_text_command(msg->data)) {
                ROS_WARN_STREAM("[Arm_internation_node] invalid cmd: " << msg->data);
            }
        }
    );

    // 发布状态数据 (20Hz)
    ros::Publisher data_pub = nh.advertise<std_msgs::String>(data_topic, 20);
    const ros::Duration publish_period(1.0 / 20.0);

    my_arm.set_decode_scale(static_cast<float>(pos_scale),
                            static_cast<float>(angle_scale));

    // ---------------- 串口连接 ----------------
    if (!serial_port.empty())
    {
        // 直接按路径连接（不阻塞，由主循环重连）
        ROS_INFO_STREAM("[Arm_internation_node] connecting to port: " << serial_port
                        << " @ " << baud_rate);
        if (!my_arm.open(serial_port, baud_rate)) {
            ROS_WARN("[Arm_internation_node] open port failed, will retry in loop");
        }
    }
    else
    {
        // 按硬件 ID 自动扫描（阻塞直到成功）
        ROS_INFO_STREAM("[Arm_internation_node] wait for device hw_id=" << hw_id
                        << ", baud=" << baud_rate);
        my_arm.open_by_HWid(hw_id, baud_rate, 1000);
    }

    // ---------------- 主循环 ----------------
    ros::Rate loop_rate(200);
    ros::Time next_publish_time  = ros::Time::now();
    ros::Time next_conn_check_time = ros::Time::now();

    while (ros::ok())
    {
        const ros::Time now = ros::Time::now();

        // ---- 每秒检查设备在线状态 ----
        if (now >= next_conn_check_time)
        {
            next_conn_check_time = now + ros::Duration(1.0);
            if (!my_arm.is_open())
            {
                ROS_WARN_THROTTLE(5, "[Arm_internation_node] serial disconnected, reconnecting...");
                my_arm.try_reconnect_once();
            }
        }

        // 读取并解析串口反馈
        my_arm.receive_once();

        // 20Hz 发布状态
        if (now >= next_publish_time)
        {
            const ArmEndPosFloat lf = my_arm.get_arm_pos_float(0);
            const ArmEndPosFloat rf = my_arm.get_arm_pos_float(1);
            const ArmEndPosFloat lb = my_arm.get_arm_pos_float(2);
            const ArmEndPosFloat rb = my_arm.get_arm_pos_float(3);
            const GimbalAngleFloat gim = my_arm.get_gimbal_float();
            const SensorStatus sensor = my_arm.get_sensor();

            int valve_bits = 0;
            int micro_bits = 0;
            for (int i = 0; i < 4; ++i) {
                if (sensor.valve[i]) valve_bits |= (1 << i);
                if (sensor.microswitch[i]) micro_bits |= (1 << i);
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
        }

        ros::spinOnce();
        loop_rate.sleep();
    }

    (void)cmd_sub;
    (void)data_pub;
    my_arm.close();
    return 0;
}