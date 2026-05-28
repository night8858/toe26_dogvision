#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

#include <dogvision_arm/arm_internation.hpp>

using namespace std::chrono_literals;

/**
 * @brief 构建兼容旧接口的机械臂状态字符串。
 * @param my_arm 用于读取当前状态的机械臂通信对象。
 * @retval std::string 用于 /arm_internation/data 的状态数据。
 */
static std::string build_status_payload(const arm_internation& my_arm)
{
    const ArmEndPosFloat lf = my_arm.get_arm_pos_float(0);
    const ArmEndPosFloat rf = my_arm.get_arm_pos_float(1);
    const ArmEndPosFloat lb = my_arm.get_arm_pos_float(2);
    const ArmEndPosFloat rb = my_arm.get_arm_pos_float(3);
    const GimbalAngleFloat gim = my_arm.get_gimbal_float();
    const SensorStatus sensor = my_arm.get_sensor();

    int valve_bits = 0;
    int micro_bits = 0;
    for (int i = 0; i < 4; ++i)
    {
        if (sensor.valve[i])
        {
            valve_bits |= (1 << i);
        }
        if (sensor.microswitch[i])
        {
            micro_bits |= (1 << i);
        }
    }

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
    return oss.str();
}

/**
 * @brief 运行 ROS2 机械臂串口通信节点。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @retval int 进程退出码。
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("arm_internation_node");
    auto logger = node->get_logger();

    node->declare_parameter<std::string>("hw_id", "0483:5740");
    node->declare_parameter<int>("baud_rate", 115200);
    node->declare_parameter<std::string>("port", "");
    node->declare_parameter<std::string>("cmd_topic", "/arm_internation/cmd");
    node->declare_parameter<std::string>("data_topic", "/arm_internation/data");
    node->declare_parameter<double>("pos_scale", 0.01);
    node->declare_parameter<double>("angle_scale", 0.01);

    const std::string hw_id = node->get_parameter("hw_id").as_string();
    const int baud_rate = static_cast<int>(node->get_parameter("baud_rate").as_int());
    const std::string serial_port = node->get_parameter("port").as_string();
    const std::string cmd_topic = node->get_parameter("cmd_topic").as_string();
    const std::string data_topic = node->get_parameter("data_topic").as_string();
    const double pos_scale = node->get_parameter("pos_scale").as_double();
    const double angle_scale = node->get_parameter("angle_scale").as_double();

    arm_internation my_arm;
    my_arm.set_decode_scale(static_cast<float>(pos_scale), static_cast<float>(angle_scale));

    auto cmd_sub = node->create_subscription<std_msgs::msg::String>(
        cmd_topic, rclcpp::QoS(20),
        [&](const std_msgs::msg::String::SharedPtr msg) {
            if (!my_arm.handle_text_command(msg->data))
            {
                RCLCPP_WARN(logger, "[arm_internation_node] invalid cmd: %s", msg->data.c_str());
            }
        });
    auto data_pub = node->create_publisher<std_msgs::msg::String>(data_topic, rclcpp::QoS(20));

    if (!serial_port.empty())
    {
        RCLCPP_INFO(logger, "[arm_internation_node] connecting to port: %s @ %d",
                    serial_port.c_str(), baud_rate);
        if (!my_arm.open(serial_port, baud_rate))
        {
            RCLCPP_WARN(logger, "[arm_internation_node] open port failed, will retry in loop");
        }
    }
    else
    {
        RCLCPP_INFO(logger, "[arm_internation_node] wait for device hw_id=%s, baud=%d",
                    hw_id.c_str(), baud_rate);
        my_arm.open_by_HWid(hw_id, baud_rate, 1000);
    }

    rclcpp::WallRate loop_rate(200);
    auto next_publish_time = std::chrono::steady_clock::now();
    auto next_conn_check_time = std::chrono::steady_clock::now();

    while (rclcpp::ok())
    {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_conn_check_time)
        {
            next_conn_check_time = now + 1s;
            if (!my_arm.is_open())
            {
                RCLCPP_WARN(logger, "[arm_internation_node] serial disconnected, reconnecting...");
                my_arm.try_reconnect_once();
            }
        }

        my_arm.receive_once();

        if (now >= next_publish_time)
        {
            std_msgs::msg::String msg;
            msg.data = build_status_payload(my_arm);
            data_pub->publish(msg);
            next_publish_time = now + 50ms;
        }

        rclcpp::spin_some(node);
        loop_rate.sleep();
    }

    (void)cmd_sub;
    my_arm.close();
    rclcpp::shutdown();
    return 0;
}
