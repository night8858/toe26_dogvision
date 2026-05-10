#include <ros/ros.h>
#include <std_msgs/String.h>
#include <XmlRpc.h>

#include <sstream>
#include <string>
#include <vector>

// ============================================================
//  arm_mission_node  高层指令编排节点
// ------------------------------------------------------------
//  订阅 /arm/mission_cmd（std_msgs/String），支持以下高层指令：
//
//    STOW[,ALL|alias]        -> 臂/所有臂移动到收起位置（纯运动）
//    START[,ALL|alias]       -> 臂/所有臂移动到启动位置（纯运动）
//    PICK[,ALL|alias]        -> 臂/所有臂移动到吸取位置（纯运动）
//    PLACE,ALL|alias|id,X,Y  -> 臂移动到放置/指定位置（纯运动）
//    VALVE/V,<id>|ALL,ON/OFF -> 电磁阀独立控制
//    PUMP/P,ON[,speed]|OFF   -> 气泵独立控制
//    PLACE_END               -> 关闭所有电磁阀 + 关气泵（复合指令）
//    
//  将高层指令拆解为低层指令序列，逐个发布到 /arm_internation/cmd，
//  由 Arm_internation_node 执行。
// ============================================================

namespace
{

// ---- 机械臂别名映射（与 arm_internation 一致）----
const char* kArmAlias[4] = {"LF", "RF", "LB", "RB"};


// ---- 位置配置（从 pos_set.yaml 加载）----
// 收起位置
float g_stow_pos[4][2] = {};
// 吸取物块位置
float g_pick_pos[4][2] = {};
// 放置物块位置
float g_place_pos[4][2] = {};
// 启动位置
float g_start_pos[4][2] = {};

// ---- 臂别名 → ID 映射（与 arm_internation 一致）----
int arm_alias_to_id(const std::string& alias)
{
    std::string upper;
    for (char c : alias)
        upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    if (upper == "LF" || upper == "FL") return 0;
    if (upper == "RF" || upper == "FR") return 1;
    if (upper == "LB" || upper == "BL") return 2;
    if (upper == "RB" || upper == "BR") return 3;
    return -1;
}

// ---- 从 ROS 私有参数加载位置配置 ----
void load_arm_positions(ros::NodeHandle& pnh, const std::string& prefix, float pos[4][2])
{
    // 默认值
    for (int i = 0; i < 4; ++i) { pos[i][0] = 10.0f; pos[i][1] = 10.0f; }

    XmlRpc::XmlRpcValue config;
    if (!pnh.getParam(prefix, config)) return;
    if (config.getType() != XmlRpc::XmlRpcValue::TypeStruct) return;

    const char* aliases[4] = {"LF", "RF", "LB", "RB"};
    for (int i = 0; i < 4; ++i)
    {
        if (config.hasMember(aliases[i]))
        {
            XmlRpc::XmlRpcValue& arm = config[aliases[i]];
            if (arm.getType() == XmlRpc::XmlRpcValue::TypeStruct)
            {
                if (arm.hasMember("x"))
                    pos[i][0] = static_cast<float>(static_cast<double>(arm["x"]));
                if (arm.hasMember("y"))
                    pos[i][1] = static_cast<float>(static_cast<double>(arm["y"]));
            }
        }
    }
}

void load_all_positions(ros::NodeHandle& pnh)
{
    load_arm_positions(pnh, "stow_pos", g_stow_pos);
    load_arm_positions(pnh, "pick_pos", g_pick_pos);
    load_arm_positions(pnh, "place_pos", g_place_pos);
    load_arm_positions(pnh, "start_pos", g_start_pos);
}

// 默认气泵速度
constexpr int kPumpSpeed = 2500;

// 指令间延时（秒），给下位机执行时间
constexpr double kCmdInterval = 0.2;

// ---- 工具函数：发布低层指令并等待 ----
void publish_and_sleep(ros::Publisher& pub, const std::string& cmd, double interval)
{
    std_msgs::String msg;
    msg.data = cmd;
    pub.publish(msg);
    ROS_INFO_STREAM("[arm_mission_node] >> " << cmd);
    ros::Duration(interval).sleep();
}

// ---- 构造机械臂低层指令字符串 ----
std::string make_arm_cmd(int arm_id, float x, float y)
{
    std::ostringstream oss;
    oss << kArmAlias[arm_id] << ",X:" << x << ",Y:" << y;
    return oss.str();
}

// ---- 构造电磁阀低层指令字符串 ----
std::string make_valve_cmd(int valve_id, bool state)
{
    std::ostringstream oss;
    oss << "V," << valve_id << "," << (state ? "ON" : "OFF");
    return oss.str();
}

// ---- 发布执行完成反馈 ----
void send_feedback(ros::Publisher* pub)
{
    if (!pub) return;
    std_msgs::String msg;
    msg.data = "FEEDBACK:DONE";
    pub->publish(msg);
    ROS_INFO("[arm_mission_node] FEEDBACK:DONE");
}

// ---- 高层指令回调函数 ----
void mission_callback(const std_msgs::String::ConstPtr& msg,
                      ros::Publisher* cmd_pub,
                      ros::Publisher* feedback_pub)
{
    const std::string& data = msg->data;

    // 简单按逗号分割（不依赖 arm_internation::normalize_cmd_text）
    std::vector<std::string> tokens;
    {
        std::string tmp;
        for (size_t i = 0; i < data.size(); ++i)
        {
            const unsigned char uc = static_cast<unsigned char>(data[i]);
            // 跳过空白字符
            if (uc == ' ' || uc == '\t')
                continue;

            // ASCII 分隔符
            if (uc == ',' || uc == ';')
            {
                if (!tmp.empty())
                {
                    tokens.push_back(tmp);
                    tmp.clear();
                }
                continue;
            }

            // 中文字符（UTF-8 多字节序列）—— 原样保留
            if (uc >= 0x80)
            {
                tmp.push_back(data[i]);  // 首字节
                ++i;
                while (i < data.size() && (static_cast<unsigned char>(data[i]) & 0xC0) == 0x80)
                {
                    tmp.push_back(data[i]);
                    ++i;
                }
                --i;  // for 循环会再次 ++i
                continue;
            }

            // ASCII 字母转大写
            if (uc >= 'a' && uc <= 'z')
                tmp.push_back(static_cast<char>(uc - 32));  // to upper
            else
                tmp.push_back(static_cast<char>(uc));
        }
        if (!tmp.empty())
            tokens.push_back(tmp);
    }

    if (tokens.empty())
    {
        ROS_WARN("[arm_mission_node] empty command");
        return;
    }

    const std::string& cmd = tokens[0];

    // ================================================================
    //  STOW：收起（纯运动，不操作电磁阀/气泵）
    //  STOW               -> 所有臂收起（向后兼容）
    //  STOW,ALL           -> 所有臂收起
    //  STOW,LF/RF/LB/RB   -> 指定臂收起
    // ================================================================
    if (cmd == "STOW" || cmd == "收起")
    {
        if (tokens.size() < 2 || tokens[1] == "ALL" || tokens[1] == "所有")
        {
            ROS_INFO("[arm_mission_node] STOW,ALL");
            for (int i = 0; i < 4; ++i)
                publish_and_sleep(*cmd_pub, make_arm_cmd(i, g_stow_pos[i][0], g_stow_pos[i][1]), kCmdInterval);
        }
        else
        {
            int id = arm_alias_to_id(tokens[1]);
            if (id < 0)
            {
                ROS_WARN("[arm_mission_node] STOW unknown alias: %s", tokens[1].c_str());
                return;
            }
            ROS_INFO_STREAM("[arm_mission_node] STOW," << kArmAlias[id]);
            publish_and_sleep(*cmd_pub, make_arm_cmd(id, g_stow_pos[id][0], g_stow_pos[id][1]), kCmdInterval);
        }
        send_feedback(feedback_pub);
        return;
    }

    // ================================================================
    //  PICK：吸取位置（纯运动，不操作电磁阀/气泵）
    //  PICK               -> 所有臂到吸取位置（向后兼容）
    //  PICK,ALL           -> 所有臂到吸取位置
    //  PICK,LF/RF/LB/RB   -> 指定臂到吸取位置
    // ================================================================
    if (cmd == "PICK" || cmd == "吸取")
    {
        if (tokens.size() < 2 || tokens[1] == "ALL" || tokens[1] == "所有")
        {
            ROS_INFO("[arm_mission_node] PICK,ALL");
            for (int i = 0; i < 4; ++i)
                publish_and_sleep(*cmd_pub, make_arm_cmd(i, g_pick_pos[i][0], g_pick_pos[i][1]), kCmdInterval);
        }
        else
        {
            int id = arm_alias_to_id(tokens[1]);
            if (id < 0)
            {
                ROS_WARN("[arm_mission_node] PICK unknown alias: %s", tokens[1].c_str());
                return;
            }
            ROS_INFO_STREAM("[arm_mission_node] PICK," << kArmAlias[id]);
            publish_and_sleep(*cmd_pub, make_arm_cmd(id, g_pick_pos[id][0], g_pick_pos[id][1]), kCmdInterval);
        }
        send_feedback(feedback_pub);
        return;
    }

    // ================================================================
    //  START：启动位置（纯运动，不操作电磁阀/气泵）
    //  START              -> 所有臂到启动位置（向后兼容）
    //  START,ALL          -> 所有臂到启动位置
    //  START,LF/RF/LB/RB  -> 指定臂到启动位置
    // ================================================================
    if (cmd == "START" || cmd == "启动")
    {
        if (tokens.size() < 2 || tokens[1] == "ALL" || tokens[1] == "所有")
        {
            ROS_INFO("[arm_mission_node] START,ALL");
            for (int i = 0; i < 4; ++i)
                publish_and_sleep(*cmd_pub, make_arm_cmd(i, g_start_pos[i][0], g_start_pos[i][1]), kCmdInterval);
        }
        else
        {
            int id = arm_alias_to_id(tokens[1]);
            if (id < 0)
            {
                ROS_WARN("[arm_mission_node] START unknown alias: %s", tokens[1].c_str());
                return;
            }
            ROS_INFO_STREAM("[arm_mission_node] START," << kArmAlias[id]);
            publish_and_sleep(*cmd_pub, make_arm_cmd(id, g_start_pos[id][0], g_start_pos[id][1]), kCmdInterval);
        }
        send_feedback(feedback_pub);
        return;
    }

    // ================================================================
    //  PLACE：放置位置（纯运动，不操作电磁阀/气泵）
    //  PLACE,ALL               -> 所有臂到放置位置
    //  PLACE,LF/RF/LB/RB       -> 指定臂到预设放置位置
    //  PLACE,<id>,<X>,<Y>      -> 指定 id 臂到显式坐标
    // ================================================================
    if (cmd == "PLACE" || cmd == "放置")
    {
        if (tokens.size() < 2)
        {
            ROS_WARN("[arm_mission_node] PLACE requires target: ALL, alias, or id,X,Y");
            return;
        }

        // 全部臂到放置位置
        if (tokens[1] == "ALL" || tokens[1] == "所有")
        {
            ROS_INFO("[arm_mission_node] PLACE,ALL");
            for (int i = 0; i < 4; ++i)
                publish_and_sleep(*cmd_pub, make_arm_cmd(i, g_place_pos[i][0], g_place_pos[i][1]), kCmdInterval);
            send_feedback(feedback_pub);
            return;
        }

        // 按别名到预设放置位置
        {
            int id = arm_alias_to_id(tokens[1]);
            if (id >= 0)
            {
                ROS_INFO_STREAM("[arm_mission_node] PLACE," << kArmAlias[id]
                                << " -> (" << g_place_pos[id][0] << ", " << g_place_pos[id][1] << ")");
                publish_and_sleep(*cmd_pub, make_arm_cmd(id, g_place_pos[id][0], g_place_pos[id][1]), kCmdInterval);
                send_feedback(feedback_pub);
                return;
            }
        }

        // 兼容旧格式：PLACE,id,X,Y
        if (tokens.size() >= 4)
        {
            try
            {
                int id = std::stoi(tokens[1]);
                float x = std::stof(tokens[2]);
                float y = std::stof(tokens[3]);
                if (id < 0 || id > 3)
                {
                    ROS_WARN("[arm_mission_node] PLACE id out of range [0-3]: %d", id);
                    return;
                }
                ROS_INFO_STREAM("[arm_mission_node] PLACE,id=" << id << " X=" << x << " Y=" << y);
                publish_and_sleep(*cmd_pub, make_arm_cmd(id, x, y), kCmdInterval);
                send_feedback(feedback_pub);
            }
            catch (...)
            {
                ROS_WARN("[arm_mission_node] PLACE invalid args: %s %s %s",
                         tokens[1].c_str(), tokens[2].c_str(), tokens[3].c_str());
            }
            return;
        }

        ROS_WARN("[arm_mission_node] PLACE: unrecognized arg '%s'", tokens[1].c_str());
        return;
    }

    // ================================================================
    //  VALVE：电磁阀独立控制
    //  VALVE/V,<id>,ON/OFF    或  VALVE/V,ALL,ON/OFF
    // ================================================================
    if (cmd == "VALVE" || cmd == "V" || cmd == "电磁阀")
    {
        if (tokens.size() < 3)
        {
            ROS_WARN("[arm_mission_node] VALVE requires: id,ON/OFF");
            return;
        }

        bool state;
        if (tokens[2] == "ON" || tokens[2] == "开")
            state = true;
        else if (tokens[2] == "OFF" || tokens[2] == "关")
            state = false;
        else
        {
            ROS_WARN("[arm_mission_node] VALVE invalid state: %s (use ON/OFF)", tokens[2].c_str());
            return;
        }

        if (tokens[1] == "ALL" || tokens[1] == "所有")
        {
            ROS_INFO_STREAM("[arm_mission_node] VALVE,ALL," << (state ? "ON" : "OFF"));
            for (int i = 0; i < 4; ++i)
                publish_and_sleep(*cmd_pub, make_valve_cmd(i, state), kCmdInterval * 0.3);
        }
        else
        {
            try
            {
                int id = std::stoi(tokens[1]);
                if (id < 0 || id > 3)
                {
                    ROS_WARN("[arm_mission_node] VALVE id out of range [0-3]: %d", id);
                    return;
                }
                ROS_INFO_STREAM("[arm_mission_node] VALVE," << id << "," << (state ? "ON" : "OFF"));
                publish_and_sleep(*cmd_pub, make_valve_cmd(id, state), kCmdInterval * 0.3);
            }
            catch (...)
            {
                ROS_WARN("[arm_mission_node] VALVE invalid id: %s", tokens[1].c_str());
            }
        }
        send_feedback(feedback_pub);
        return;
    }

    // ================================================================
    //  PUMP：气泵独立控制
    //  PUMP/P,ON[,<speed>]    或  PUMP/P,OFF
    // ================================================================
    if (cmd == "PUMP" || cmd == "P" || cmd == "气泵")
    {
        if (tokens.size() < 2)
        {
            ROS_WARN("[arm_mission_node] PUMP requires: ON[,speed] or OFF");
            return;
        }

        if (tokens[1] == "ON" || tokens[1] == "开")
        {
            int speed = kPumpSpeed;
            if (tokens.size() >= 3)
            {
                try { speed = std::stoi(tokens[2]); }
                catch (...) { ROS_WARN("[arm_mission_node] PUMP invalid speed: %s, using default", tokens[2].c_str()); }
            }
            std::ostringstream oss;
            oss << "P,ON," << speed;
            ROS_INFO_STREAM("[arm_mission_node] PUMP,ON," << speed);
            publish_and_sleep(*cmd_pub, oss.str(), kCmdInterval);
        }
        else if (tokens[1] == "OFF" || tokens[1] == "关")
        {
            ROS_INFO("[arm_mission_node] PUMP,OFF");
            publish_and_sleep(*cmd_pub, "P,OFF", kCmdInterval);
        }
        else
        {
            ROS_WARN("[arm_mission_node] PUMP unrecognized arg: %s", tokens[1].c_str());
        }
        send_feedback(feedback_pub);
        return;
    }

    // ================================================================
    //  PLACE_END：放置结束（复合指令，等效 VALVE,ALL,OFF + PUMP,OFF）
    // ================================================================
    if (cmd == "PLACE_END" || cmd == "PLACEEND" || cmd == "放置结束")
    {
        ROS_INFO("[arm_mission_node] PLACE_END: close all valves, stop pump");
        for (int i = 0; i < 4; ++i)
            publish_and_sleep(*cmd_pub, make_valve_cmd(i, false), kCmdInterval * 0.3);
        ros::Duration(kCmdInterval * 0.5).sleep();
        publish_and_sleep(*cmd_pub, "P,OFF", kCmdInterval);
        send_feedback(feedback_pub);
        return;
    }

    // ================================================================
    //  无法识别的命令
    // ================================================================
    ROS_WARN_STREAM("[arm_mission_node] unknown mission cmd: " << data);
}

} // namespace

// ================================================================
//  main
// ================================================================
int main(int argc, char** argv)
{
    ros::init(argc, argv, "arm_mission_node");

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string mission_topic = "/arm/mission_cmd";
    std::string cmd_topic = "/arm_internation/cmd";
    pnh.param<std::string>("mission_topic", mission_topic, mission_topic);
    pnh.param<std::string>("cmd_topic", cmd_topic, cmd_topic);

    // 从 pos_set.yaml 加载位置配置（通过 ROS 参数服务器）
    load_all_positions(pnh);
    ROS_INFO("[arm_mission_node] position config loaded from namespace '~'");

    // 发布低层指令到 Arm_internation_node
    ros::Publisher cmd_pub = nh.advertise<std_msgs::String>(cmd_topic, 10);
    // 发布执行完成反馈
    ros::Publisher feedback_pub = nh.advertise<std_msgs::String>("/arm/mission_cmd", 10);

    // 订阅高层指令
    ros::Subscriber mission_sub = nh.subscribe<std_msgs::String>(
        mission_topic, 10,
        [&cmd_pub, &feedback_pub](const std_msgs::String::ConstPtr& msg) {
            mission_callback(msg, &cmd_pub, &feedback_pub);
        });

    ROS_INFO_STREAM("[arm_mission_node] Ready. Subscribing to: " << mission_topic);
    ROS_INFO_STREAM("[arm_mission_node] Publishing low-level commands to: " << cmd_topic);
    ROS_INFO_STREAM("[arm_mission_node] Publishing feedback to: /arm/mission_cmd");
    ROS_INFO_STREAM("[arm_mission_node] Supported commands: STOW[,ALL|alias], START[,ALL|alias], PICK[,ALL|alias], PLACE,ALL|alias|id,X,Y, VALVE/V,id|ALL,ON/OFF, PUMP/P,ON[,speed]|OFF, PLACE_END");

    ros::spin();

    return 0;
}
