#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <chrono>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

using namespace std::chrono_literals;

namespace
{
const char* kArmAlias[4] = {"LF", "RF", "LB", "RB"};
constexpr int kPumpSpeed = 2500;
constexpr double kCmdInterval = 0.2;

float g_stow_pos[4][2] = {};
float g_pick_pos[4][2] = {};
float g_place_pos[4][2] = {};
float g_start_pos[4][2] = {};
} // namespace

/**
 * @brief 将机械臂别名转换为数字编号。
 * @param alias 机械臂别名，例如 LF、RF、LB、RB、FL、FR、BL、BR。
 * @retval int 机械臂编号，范围为 [0,3]；未知别名返回 -1。
 */
static int arm_alias_to_id(const std::string& alias)
{
    std::string upper;
    for (char c : alias)
    {
        upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    if (upper == "LF" || upper == "FL") return 0;
    if (upper == "RF" || upper == "FR") return 1;
    if (upper == "LB" || upper == "BL") return 2;
    if (upper == "RB" || upper == "BR") return 3;
    return -1;
}

/**
 * @brief 从 ROS2 参数中声明并加载一组机械臂位置。
 *
 * 设计意图：每组位置（收起/吸取/放置/启动）按 `<prefix>.<alias>.<axis>` 三级嵌套声明，
 * 如 `stow_pos.LF.x`、`stow_pos.LF.y`。使用 declare_parameter + get_parameter 模式，
 * 允许 launch 文件或 YAML 覆盖默认值 10.0。
 *
 * @param node 持有参数的节点对象。
 * @param prefix 位置参数组前缀（如 "stow_pos"）。
 * @param[out] pos 输出位置表 [4][2]，按 LF/RF/LB/RB 顺序。
 * @retval void
 */
static void load_arm_positions(const rclcpp::Node::SharedPtr& node, const std::string& prefix, float pos[4][2])
{
    for (int i = 0; i < 4; ++i)
    {
        const std::string base = prefix + "." + kArmAlias[i];
        node->declare_parameter<double>(base + ".x", 10.0);
        node->declare_parameter<double>(base + ".y", 10.0);
        pos[i][0] = static_cast<float>(node->get_parameter(base + ".x").as_double());
        pos[i][1] = static_cast<float>(node->get_parameter(base + ".y").as_double());
    }
}

/**
 * @brief 加载所有配置的机械臂位置组。
 * @param node 持有参数的节点对象。
 * @retval void
 */
static void load_all_positions(const rclcpp::Node::SharedPtr& node)
{
    load_arm_positions(node, "stow_pos", g_stow_pos);
    load_arm_positions(node, "pick_pos", g_pick_pos);
    load_arm_positions(node, "place_pos", g_place_pos);
    load_arm_positions(node, "start_pos", g_start_pos);
}

/**
 * @brief 发布一条低层机械臂命令并等待指定间隔。
 * @param pub 低层命令发布器。
 * @param logger 用于命令跟踪的日志对象。
 * @param cmd 需要发布的命令字符串。
 * @param interval 发布后的等待时间，单位为秒。
 * @retval void
 */
static void publish_and_sleep(const rclcpp::Publisher<std_msgs::msg::String>::SharedPtr& pub,
                              const rclcpp::Logger& logger,
                              const std::string& cmd,
                              double interval)
{
    std_msgs::msg::String msg;
    msg.data = cmd;
    pub->publish(msg);
    RCLCPP_INFO(logger, "[arm_mission_node] >> %s", cmd.c_str());
    rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(interval)));
}

/**
 * @brief 构建低层机械臂运动命令。
 * @param arm_id 机械臂编号，范围为 [0,3]。
 * @param x 目标 x 坐标。
 * @param y 目标 y 坐标。
 * @retval std::string 低层命令字符串。
 */
static std::string make_arm_cmd(int arm_id, float x, float y)
{
    std::ostringstream oss;
    oss << kArmAlias[arm_id] << ",X:" << x << ",Y:" << y;
    return oss.str();
}

/**
 * @brief 构建低层电磁阀命令。
 * @param valve_id 电磁阀编号，范围为 [0,3]。
 * @param state true 表示打开，false 表示关闭。
 * @retval std::string 低层命令字符串。
 */
static std::string make_valve_cmd(int valve_id, bool state)
{
    std::ostringstream oss;
    oss << "V," << valve_id << "," << (state ? "ON" : "OFF");
    return oss.str();
}

/**
 * @brief 发布任务完成反馈。
 * @param pub 反馈发布器。
 * @param logger 用于反馈跟踪的日志对象。
 * @retval void
 */
static void send_feedback(const rclcpp::Publisher<std_msgs::msg::String>::SharedPtr& pub,
                          const rclcpp::Logger& logger)
{
    std_msgs::msg::String msg;
    msg.data = "FEEDBACK:DONE";
    pub->publish(msg);
    RCLCPP_INFO(logger, "[arm_mission_node] FEEDBACK:DONE");
}

/**
 * @brief 归一化并拆分任务命令字符串。
 *
 * 设计意图：兼容多种输入格式（空格、逗号、分号、中文标点），
 * 将所有 token 转为大写以实现大小写不敏感匹配。
 * 同时正确处理 UTF-8 多字节中文字符（如"收起"、"放置"）。
 *
 * @param data 原始任务命令。
 * @retval std::vector<std::string> 使用逗号或分号拆分后的大写 token 列表。
 */
static std::vector<std::string> tokenize_command(const std::string& data)
{
    std::vector<std::string> tokens;
    std::string tmp;
    for (size_t i = 0; i < data.size(); ++i)
    {
        const unsigned char uc = static_cast<unsigned char>(data[i]);
        if (uc == ' ' || uc == '\t')
        {
            continue;
        }
        if (uc == ',' || uc == ';')
        {
            if (!tmp.empty())
            {
                tokens.push_back(tmp);
                tmp.clear();
            }
            continue;
        }
        if (uc >= 0x80)
        {
            tmp.push_back(data[i]);
            ++i;
            while (i < data.size() && (static_cast<unsigned char>(data[i]) & 0xC0) == 0x80)
            {
                tmp.push_back(data[i]);
                ++i;
            }
            --i;
            continue;
        }
        tmp.push_back(static_cast<char>(std::toupper(uc)));
    }
    if (!tmp.empty())
    {
        tokens.push_back(tmp);
    }
    return tokens;
}

/**
 * @brief 处理高层任务命令并发布低层命令序列。
 *
 * @section 任务命令状态机
 * 本函数是 arm_mission_node 的核心，将高层语义命令拆解为低层串口指令序列。
 * 每条高层命令执行完毕后发布 "FEEDBACK:DONE" 确认。
 *
 * 命令分派决策树（首 token 匹配，支持中英文别名）：
 *
 *   tokens[0]
 *      │
 *   ┌──┼──────────┬──────────┬──────────┬──────────┬──────────┐
 *   ▼  ▼          ▼          ▼          ▼          ▼          ▼
 * STOW PICK      START      PLACE     VALVE/V    PUMP/P   PLACE_END
 * 收起 吸取       启动        放置      电磁阀     气泵     放置结束
 *   │  │          │          │          │          │          │
 *   └──┴──────────┴──────────┤          │          │          │
 *                            │          │          │          │
 *         tokens[1] 分派 ────┘          │          │          │
 *         ├─ ALL → 4臂循环             │          │          │
 *         ├─ LF/RF/LB/RB → 单臂        │          │          │
 *         └─ 数字,X,Y → 指定臂+坐标     │          │          │
 *                                       │          │          │
 *                    tokens[1] 分派 ────┘          │          │
 *                    ├─ ALL → 4阀循环              │          │
 *                    └─ 数字 → 单阀                 │          │
 *                                                   │          │
 *                            tokens[1] 分派 ────────┘          │
 *                            ├─ ON[,speed] → 开泵+设速         │
 *                            └─ OFF → 关泵                     │
 *                                                               │
 *                                       4阀关+泵关+反馈 ───────┘
 *
 * @section 命令间隔设计
 * kCmdInterval (0.2s) 保证下位机有足够时间执行上一指令，
 * 避免连续发送导致指令队列溢出或机械臂运动冲突。
 * 电磁阀命令间隔缩短为 0.3x (60ms)，因电磁阀响应远快于机械运动。
 *
 * @section 异常处理
 * - 空命令：WARN 后直接返回
 * - 未知命令：WARN 日志
 * - 无效别名/编号越界：WARN 后返回，不崩溃
 * - 数值解析失败（stoi/stof 异常）：catch 后 WARN，不崩溃
 *
 * @param data 原始任务命令数据。
 * @param cmd_pub 低层机械臂命令发布器。
 * @param feedback_pub 完成反馈发布器。
 * @param logger 用于警告和跟踪的日志对象。
 * @retval void
 */
static void mission_callback(const std::string& data,
                             const rclcpp::Publisher<std_msgs::msg::String>::SharedPtr& cmd_pub,
                             const rclcpp::Publisher<std_msgs::msg::String>::SharedPtr& feedback_pub,
                             const rclcpp::Logger& logger)
{
    const std::vector<std::string> tokens = tokenize_command(data);
    if (tokens.empty())
    {
        RCLCPP_WARN(logger, "[arm_mission_node] empty command");
        return;
    }

    const std::string& cmd = tokens[0];
    auto publish_arm_group = [&](float pos[4][2], const char* label) {
        if (tokens.size() < 2 || tokens[1] == "ALL" || tokens[1] == "所有")
        {
            RCLCPP_INFO(logger, "[arm_mission_node] %s,ALL", label);
            for (int i = 0; i < 4; ++i)
            {
                publish_and_sleep(cmd_pub, logger, make_arm_cmd(i, pos[i][0], pos[i][1]), kCmdInterval);
            }
        }
        else
        {
            const int id = arm_alias_to_id(tokens[1]);
            if (id < 0)
            {
                RCLCPP_WARN(logger, "[arm_mission_node] %s unknown alias: %s", label, tokens[1].c_str());
                return;
            }
            RCLCPP_INFO(logger, "[arm_mission_node] %s,%s", label, kArmAlias[id]);
            publish_and_sleep(cmd_pub, logger, make_arm_cmd(id, pos[id][0], pos[id][1]), kCmdInterval);
        }
        send_feedback(feedback_pub, logger);
    };

    if (cmd == "STOW" || cmd == "收起")
    {
        publish_arm_group(g_stow_pos, "STOW");
        return;
    }
    if (cmd == "PICK" || cmd == "吸取")
    {
        publish_arm_group(g_pick_pos, "PICK");
        return;
    }
    if (cmd == "START" || cmd == "启动")
    {
        publish_arm_group(g_start_pos, "START");
        return;
    }
    if (cmd == "PLACE" || cmd == "放置")
    {
        if (tokens.size() < 2)
        {
            RCLCPP_WARN(logger, "[arm_mission_node] PLACE requires target: ALL, alias, or id,X,Y");
            return;
        }
        if (tokens[1] == "ALL" || tokens[1] == "所有")
        {
            for (int i = 0; i < 4; ++i)
            {
                publish_and_sleep(cmd_pub, logger, make_arm_cmd(i, g_place_pos[i][0], g_place_pos[i][1]), kCmdInterval);
            }
            send_feedback(feedback_pub, logger);
            return;
        }
        const int alias_id = arm_alias_to_id(tokens[1]);
        if (alias_id >= 0)
        {
            publish_and_sleep(cmd_pub, logger, make_arm_cmd(alias_id, g_place_pos[alias_id][0], g_place_pos[alias_id][1]), kCmdInterval);
            send_feedback(feedback_pub, logger);
            return;
        }
        if (tokens.size() >= 4)
        {
            try
            {
                const int id = std::stoi(tokens[1]);
                const float x = std::stof(tokens[2]);
                const float y = std::stof(tokens[3]);
                if (id < 0 || id > 3)
                {
                    RCLCPP_WARN(logger, "[arm_mission_node] PLACE id out of range [0-3]: %d", id);
                    return;
                }
                publish_and_sleep(cmd_pub, logger, make_arm_cmd(id, x, y), kCmdInterval);
                send_feedback(feedback_pub, logger);
            }
            catch (...)
            {
                RCLCPP_WARN(logger, "[arm_mission_node] PLACE invalid args");
            }
            return;
        }
        RCLCPP_WARN(logger, "[arm_mission_node] PLACE: unrecognized arg '%s'", tokens[1].c_str());
        return;
    }
    if (cmd == "VALVE" || cmd == "V" || cmd == "电磁阀")
    {
        if (tokens.size() < 3)
        {
            RCLCPP_WARN(logger, "[arm_mission_node] VALVE requires: id,ON/OFF");
            return;
        }
        const bool state = (tokens[2] == "ON" || tokens[2] == "开");
        if (!state && tokens[2] != "OFF" && tokens[2] != "关")
        {
            RCLCPP_WARN(logger, "[arm_mission_node] VALVE invalid state: %s", tokens[2].c_str());
            return;
        }
        if (tokens[1] == "ALL" || tokens[1] == "所有")
        {
            for (int i = 0; i < 4; ++i)
            {
                publish_and_sleep(cmd_pub, logger, make_valve_cmd(i, state), kCmdInterval * 0.3);
            }
        }
        else
        {
            try
            {
                const int id = std::stoi(tokens[1]);
                if (id < 0 || id > 3)
                {
                    RCLCPP_WARN(logger, "[arm_mission_node] VALVE id out of range [0-3]: %d", id);
                    return;
                }
                publish_and_sleep(cmd_pub, logger, make_valve_cmd(id, state), kCmdInterval * 0.3);
            }
            catch (...)
            {
                RCLCPP_WARN(logger, "[arm_mission_node] VALVE invalid id: %s", tokens[1].c_str());
            }
        }
        send_feedback(feedback_pub, logger);
        return;
    }
    if (cmd == "PUMP" || cmd == "P" || cmd == "气泵")
    {
        if (tokens.size() < 2)
        {
            RCLCPP_WARN(logger, "[arm_mission_node] PUMP requires: ON[,speed] or OFF");
            return;
        }
        if (tokens[1] == "ON" || tokens[1] == "开")
        {
            int speed = kPumpSpeed;
            if (tokens.size() >= 3)
            {
                try { speed = std::stoi(tokens[2]); }
                catch (...) { RCLCPP_WARN(logger, "[arm_mission_node] PUMP invalid speed, using default"); }
            }
            publish_and_sleep(cmd_pub, logger, "P,ON," + std::to_string(speed), kCmdInterval);
        }
        else if (tokens[1] == "OFF" || tokens[1] == "关")
        {
            publish_and_sleep(cmd_pub, logger, "P,OFF", kCmdInterval);
        }
        else
        {
            RCLCPP_WARN(logger, "[arm_mission_node] PUMP unrecognized arg: %s", tokens[1].c_str());
            return;
        }
        send_feedback(feedback_pub, logger);
        return;
    }
    if (cmd == "PLACE_END" || cmd == "PLACEEND" || cmd == "放置结束")
    {
        for (int i = 0; i < 4; ++i)
        {
            publish_and_sleep(cmd_pub, logger, make_valve_cmd(i, false), kCmdInterval * 0.3);
        }
        rclcpp::sleep_for(100ms);
        publish_and_sleep(cmd_pub, logger, "P,OFF", kCmdInterval);
        send_feedback(feedback_pub, logger);
        return;
    }

    RCLCPP_WARN(logger, "[arm_mission_node] unknown mission cmd: %s", data.c_str());
}

/**
 * @brief 运行 ROS2 机械臂任务编排节点。
 *
 * @section 设计意图
 * 本节点位于"终端命令"与"串口协议"之间的中间层：
 *
 *   arm_cmd_terminal_node          arm_mission_node            arm_internation_node
 *   (stdin → 话题路由)    ──────▶   (任务拆解 → 指令序列)  ──▶  (串口发送)
 *   /arm/mission_cmd               /arm_internation/cmd         /dev/ttyUSB0
 *
 * 职责分离：
 * - 终端节点只做字符串路由（$ 前缀分流），不理解机械臂语义
 * - 任务节点理解"收起/吸取/放置"等高层语义，拆解为多步低层指令
 * - 通信节点只做协议打包与串口收发
 *
 * @section 位置参数加载
 * 通过 ROS2 参数系统从 YAML 文件加载四组预设位置：
 * - stow_pos:  收起位置（机械臂待机）
 * - pick_pos:  吸取位置（对准物块）
 * - place_pos: 放置位置（目标区域）
 * - start_pos: 启动位置（初始展开）
 * 每组 4 臂 × 2 坐标 (x, y)，通过 declare_parameter 声明默认值 10.0。
 *
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @retval int 进程退出码。
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("arm_mission_node");
    auto logger = node->get_logger();

    node->declare_parameter<std::string>("mission_topic", "/arm/mission_cmd");
    node->declare_parameter<std::string>("cmd_topic", "/arm_internation/cmd");
    const std::string mission_topic = node->get_parameter("mission_topic").as_string();
    const std::string cmd_topic = node->get_parameter("cmd_topic").as_string();

    load_all_positions(node);
    RCLCPP_INFO(logger, "[arm_mission_node] position config loaded");

    auto cmd_pub = node->create_publisher<std_msgs::msg::String>(cmd_topic, rclcpp::QoS(10));
    auto feedback_pub = node->create_publisher<std_msgs::msg::String>("/arm/mission_cmd", rclcpp::QoS(10));
    auto mission_sub = node->create_subscription<std_msgs::msg::String>(
        mission_topic, rclcpp::QoS(10),
        [cmd_pub, feedback_pub, logger](const std_msgs::msg::String::SharedPtr msg) {
            mission_callback(msg->data, cmd_pub, feedback_pub, logger);
        });

    RCLCPP_INFO(logger, "[arm_mission_node] Ready. Subscribing to: %s", mission_topic.c_str());
    RCLCPP_INFO(logger, "[arm_mission_node] Publishing low-level commands to: %s", cmd_topic.c_str());
    rclcpp::spin(node);

    (void)mission_sub;
    rclcpp::shutdown();
    return 0;
}
