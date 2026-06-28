#include <dogvision_arm/arm_mission_protocol.hpp>

#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

using namespace std::chrono_literals;

class ArmMissionNode final : public rclcpp::Node
{
public:
    ArmMissionNode()
        : Node("arm_mission_node")
    {
        declare_parameter<std::string>("mission_topic", "/arm/mission_cmd");        // 
        declare_parameter<std::string>("cmd_topic", "/arm_internation/cmd");        // 发送话题
        declare_parameter<std::string>("state_topic", "/arm_internation/state");    // 下位机反馈状态话题，主要是 DONE / DIAG
        declare_parameter<int>("timeout_ms", 10000);                                 // 超时毫秒数，设为 0 或负数则禁用超时

        mission_topic_ = get_parameter("mission_topic").as_string();
        cmd_topic_ = get_parameter("cmd_topic").as_string();
        state_topic_ = get_parameter("state_topic").as_string();
        const int timeout_ms = get_parameter("timeout_ms").as_int();

        mission_controller_.set_timeout_ms(timeout_ms);

        cmd_pub_ = create_publisher<std_msgs::msg::String>(cmd_topic_, rclcpp::QoS(10));
        feedback_pub_ = create_publisher<std_msgs::msg::String>(mission_topic_, rclcpp::QoS(10));
        mission_sub_ = create_subscription<std_msgs::msg::String>(
            mission_topic_, rclcpp::QoS(10),
            [this](const std_msgs::msg::String::SharedPtr msg)
            {
                on_mission_command(msg->data);
            });
        state_sub_ = create_subscription<std_msgs::msg::String>(
            state_topic_, rclcpp::QoS(10),
            [this](const std_msgs::msg::String::SharedPtr msg)
            {
                on_arm_state(msg->data);
            });

        // 5000ms 定时器轮询超时，超时后自动解除 busy
        timeout_timer_ = create_wall_timer(5000ms, [this]()
        {
            const auto result = mission_controller_.check_timeout();
            if (result.completed)
            {
                RCLCPP_WARN(get_logger(),
                    "[arm_mission_node] TIMEOUT for %s (no DONE received, auto-releasing busy)",
                    result.completed_command.c_str());
                publish_feedback(result.feedback);
            }
        });

        RCLCPP_INFO(get_logger(),
            "[arm_mission_node] mission=%s cmd=%s state=%s timeout_ms=%d",
            mission_topic_.c_str(), cmd_topic_.c_str(), state_topic_.c_str(), timeout_ms);
    }

private:
    void on_mission_command(const std::string &data)
    {
        const dogvision_arm::MissionCommandResult result =
            mission_controller_.handle_mission_command(data);
        if (result.action == dogvision_arm::MissionCommandAction::Ignore)
        {
            return;
        }

        if (result.action == dogvision_arm::MissionCommandAction::Busy)
        {
            RCLCPP_WARN(get_logger(), "[arm_mission_node] busy, rejected: %s", data.c_str());
            publish_feedback(result.feedback);
            return;
        }

        if (result.action == dogvision_arm::MissionCommandAction::Invalid)
        {
            RCLCPP_WARN(get_logger(), "[arm_mission_node] %s: %s", result.error.c_str(), data.c_str());
            return;
        }

        std_msgs::msg::String msg;
        msg.data = result.low_level;
        cmd_pub_->publish(msg);
        RCLCPP_INFO(get_logger(), "[arm_mission_node] >> %s", result.low_level.c_str());
    }

    void on_arm_state(const std::string &state)
    {
        const dogvision_arm::MissionStateResult result =
            mission_controller_.handle_arm_state(state);
        if (!result.completed)
        {
            return;
        }

        if (result.feedback.rfind("FEEDBACK:REJECTED:", 0) == 0)
        {
            RCLCPP_WARN(get_logger(),
                        "[arm_mission_node] REJECTED %s for %s",
                        result.feedback.c_str(),
                        result.completed_command.c_str());
        }
        else
        {
            RCLCPP_INFO(get_logger(), "[arm_mission_node] DONE for %s", result.completed_command.c_str());
        }
        publish_feedback(result.feedback);
    }

    void publish_feedback(const std::string &text)
    {
        std_msgs::msg::String msg;
        msg.data = text;
        feedback_pub_->publish(msg);
    }

    std::string mission_topic_;
    std::string cmd_topic_;
    std::string state_topic_;
    dogvision_arm::ArmMissionController mission_controller_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr feedback_pub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mission_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
    rclcpp::TimerBase::SharedPtr timeout_timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmMissionNode>());
    rclcpp::shutdown();
    return 0;
}
