#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8.hpp>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

#include <dogvision_arm/arm_internation.hpp>

using namespace std::chrono_literals;

/**
 * @brief 构建机械臂状态字符串，用于 /arm_internation/data 话题发布。
 *
 * 设计意图：将 arm_internation 内部状态缓存序列化为单行文本，
 * 便于 ROS 话题监控、日志记录和下游节点解析。
 *
 * 输出格式固定为：
 * MODE:4DOF;L4:x,y,z,pitch;R4:x,y,z,pitch;VALVE_BITS:n;MICRO_BITS:n
 *
 * @param my_arm 用于读取当前状态的机械臂通信对象。
 * @retval std::string 用于 /arm_internation/data 的状态数据。
 */
static std::string build_diagnostic_payload(const Arm4DofDiagnostic &diagnostic)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "DIAG,arm=" << static_cast<unsigned>(diagnostic.arm_id)
        << ",reason=" << static_cast<unsigned>(diagnostic.reason)
        << ",mask=" << static_cast<unsigned>(diagnostic.joint_mask)
        << ",req=" << diagnostic.requested_pose.x << "/"
        << diagnostic.requested_pose.y << "/"
        << diagnostic.requested_pose.z << "/"
        << diagnostic.requested_pose.pitch
        << ",lim=" << diagnostic.limited_pose.x << "/"
        << diagnostic.limited_pose.y << "/"
        << diagnostic.limited_pose.z << "/"
        << diagnostic.limited_pose.pitch;
    return oss.str();
}

static std::string build_status_payload(const arm_internation &my_arm)
{
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
    oss << std::fixed << std::setprecision(3);
    const Arm4DofPoseFloat left4 = my_arm.get_4dof_pose_float(0);
    const Arm4DofPoseFloat right4 = my_arm.get_4dof_pose_float(1);
    oss << "MODE:4DOF"
        << ";L4:" << left4.x << "," << left4.y << "," << left4.z << "," << left4.pitch
        << ";R4:" << right4.x << "," << right4.y << "," << right4.z << "," << right4.pitch
        << ";VALVE_BITS:" << valve_bits
        << ";MICRO_BITS:" << micro_bits;
    return oss.str();
}

/**
 * @brief 运行 ROS2 机械臂串口通信节点。
 *
 * @section 节点架构
 * 本节点是串口通信栈的最底层节点，负责：
 * 1) 串口生命周期管理（连接/断线检测/自动重连）
 * 2) 反馈帧接收与状态发布（200Hz 主循环 → 20Hz 数据发布 + DONE 事件发布）
 * 3) 命令订阅与转发（/arm_internation/cmd → handle_text_command）
 *
 * @section 主循环时序
 *   200Hz 周期内执行：
 *   ┌─ 1s 间隔检查 ─────────────────────────────────────┐
 *   │  if (!is_open()) → try_reconnect_once()           │
 *   └──────────────────────────────────────────────────┘
 *   ┌─ 每次循环 ────────────────────────────────────────┐
 *   │  receive_once() → 读串口 + 帧解析                 │
 *   │  consume_done_feedback_count() → 发布 DONE 事件   │
 *   └──────────────────────────────────────────────────┘
 *   ┌─ 50ms 间隔 ───────────────────────────────────────┐
 *   │  build_status_payload() → publish()               │
 *   └──────────────────────────────────────────────────┘
 *   ┌─ 每次循环 ────────────────────────────────────────┐
 *   │  spin_some() → 处理订阅回调                       │
 *   └──────────────────────────────────────────────────┘
 *
 * @section 异常处理
 * - open() 失败：不退出节点，保持存活等待重连
 * - 无效命令：WARN 日志，不崩溃
 *
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @retval int 进程退出码。
 */
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("arm_internation_node");
    auto logger = node->get_logger();

    node->declare_parameter<std::string>("hw_id", "0483:5740"); // 设备HWID，默认STM32F407的USB VID:PID

    node->declare_parameter<int>("baud_rate", 115200);                           // 串口波特率
    node->declare_parameter<std::string>("port", "");                            // 串口设备路径，留空则自动根据hw_id查找
    node->declare_parameter<std::string>("cmd_topic", "/arm_internation/cmd");   // 接收命令的ROS话题
    node->declare_parameter<std::string>("data_topic", "/arm_internation/data"); // 发布状态的ROS话题，主要是机械臂的位置数据的上报
    node->declare_parameter<std::string>("state_topic", "/arm_internation/state"); // 发布一次性事件，如下位机 BB CC 完成反馈
    node->declare_parameter<std::string>("ocr_answer_topic", "/ocr/answer");     // OCR 稳定答案，UInt8 0..3
    node->declare_parameter<double>("pos_scale", 0.01);                          // 位置解码缩放，默认0.01即1cm单位

    const std::string hw_id = node->get_parameter("hw_id").as_string();
    const int baud_rate = static_cast<int>(node->get_parameter("baud_rate").as_int());
    const std::string serial_port = node->get_parameter("port").as_string();
    const std::string cmd_topic = node->get_parameter("cmd_topic").as_string();
    const std::string data_topic = node->get_parameter("data_topic").as_string();
    const std::string state_topic = node->get_parameter("state_topic").as_string();
    const std::string ocr_answer_topic =
        node->get_parameter("ocr_answer_topic").as_string();
    const double pos_scale = node->get_parameter("pos_scale").as_double();
        //
    arm_internation my_arm;
    my_arm.set_decode_scale(static_cast<float>(pos_scale));
    RCLCPP_INFO(logger, "[arm_internation_node] protocol: %s", my_arm.protocol_name());

    auto cmd_sub = node->create_subscription<std_msgs::msg::String>(
        cmd_topic, rclcpp::QoS(20),
        [&](const std_msgs::msg::String::SharedPtr msg)
        {
            if (!my_arm.handle_text_command(msg->data))
            {
                RCLCPP_WARN(logger, "[arm_internation_node] invalid cmd: %s", msg->data.c_str());
            }
        });
    auto answer_qos =
        rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    auto ocr_answer_sub = node->create_subscription<std_msgs::msg::UInt8>(
        ocr_answer_topic, answer_qos,
        [&](const std_msgs::msg::UInt8::SharedPtr msg)
        {
            const uint8_t answer = msg->data;
            if (answer > 3)
            {
                RCLCPP_ERROR(logger,
                             "[arm_internation_node] OCR answer out of range [0,3]: %u",
                             static_cast<unsigned>(answer));
                return;
            }
            if (!my_arm.is_open())
            {
                RCLCPP_ERROR(logger,
                             "[arm_internation_node] cannot send OCR answer %u: serial disconnected",
                             static_cast<unsigned>(answer));
                return;
            }
            if (!my_arm.send_answer_cmd(answer))
            {
                RCLCPP_ERROR(logger,
                             "[arm_internation_node] failed to send BB 05 answer=%u",
                             static_cast<unsigned>(answer));
                return;
            }
            RCLCPP_INFO(logger,
                        "[arm_internation_node] sent BB 05 answer=%u from %s",
                        static_cast<unsigned>(answer), ocr_answer_topic.c_str());
        });
    auto data_pub = node->create_publisher<std_msgs::msg::String>(data_topic, rclcpp::QoS(20));
    auto state_pub = node->create_publisher<std_msgs::msg::String>(state_topic, rclcpp::QoS(20));

    if (!serial_port.empty())
    {
        RCLCPP_INFO(logger, "[arm_internation_node] connecting to port: %s @ %d",
                    serial_port.c_str(), baud_rate);
        if (!my_arm.open(serial_port, baud_rate))
        {
            RCLCPP_WARN(logger, "[arm_internation_node] open port failed, node will stay alive");
        }
    }
    else
    {
        RCLCPP_INFO(logger, "[arm_internation_node] auto reconnect configured for hw_id=%s, baud=%d",
                    hw_id.c_str(), baud_rate);
        my_arm.configure_auto_reconnect(hw_id, baud_rate, 1000);
    }

    rclcpp::WallRate loop_rate(200);
    auto next_publish_time = std::chrono::steady_clock::now();
    auto next_conn_check_time = std::chrono::steady_clock::now();

    while (rclcpp::ok())
    {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_conn_check_time)
        {
            // 定期检查串口连接状态，适用于非 auto_reconnect 场景的掉线检测。
            next_conn_check_time = now + 1s;
            if (!my_arm.is_open())
            {
                RCLCPP_WARN(logger, "[arm_internation_node] serial disconnected, reconnecting...");
                my_arm.try_reconnect_once();
            }
        }

        // 主循环核心：接收并解析串口反馈帧，更新内部状态缓存。
        // 若收到 BB CC FF EE CRC8，arm_internation 不会把它塞进 /data，
        // 而是记为一次性完成事件；这里消费事件并发布给雷达/任务侧监听者。
        my_arm.receive_once();
        const size_t done_count = my_arm.consume_done_feedback_count();
        for (size_t i = 0; i < done_count; ++i)
        {
            std_msgs::msg::String msg;
            msg.data = "DONE";
            state_pub->publish(msg);
        }

        const size_t diagnostic_count = my_arm.consume_diagnostic_feedback_count();
        if (diagnostic_count > 0)
        {
            const Arm4DofDiagnostic diagnostic = my_arm.get_last_diagnostic();
            std_msgs::msg::String msg;
            msg.data = build_diagnostic_payload(diagnostic);
            state_pub->publish(msg);
            RCLCPP_WARN(logger, "[arm_internation_node] %s", msg.data.c_str());
        }

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
    (void)ocr_answer_sub;
    (void)state_pub;
    my_arm.close();
    rclcpp::shutdown();
    return 0;
}
