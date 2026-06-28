#ifndef DOGVISION_ARM_ARM_MISSION_PROTOCOL_HPP
#define DOGVISION_ARM_ARM_MISSION_PROTOCOL_HPP

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace dogvision_arm
{

inline std::string trim_copy(const std::string &s)
{
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin])))
    {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])))
    {
        --end;
    }
    return s.substr(begin, end - begin);
}

inline std::string to_upper_copy(std::string s)
{
    for (char &c : s)
    {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

inline std::vector<std::string> tokenize_command(const std::string &data)
{
    std::string normalized = data;
    auto replace_all = [&](const std::string &from, const std::string &to)
    {
        size_t pos = 0;
        while ((pos = normalized.find(from, pos)) != std::string::npos)
        {
            normalized.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replace_all("，", ",");
    replace_all("；", ",");
    replace_all("：", ":");

    std::vector<std::string> tokens;
    std::stringstream ss(normalized);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        item = trim_copy(item);
        if (!item.empty())
        {
            tokens.push_back(to_upper_copy(item));
        }
    }
    return tokens;
}

inline bool parse_float_token(const std::string &token, float &value)
{
    char *end_ptr = nullptr;
    const std::string t = trim_copy(token);
    if (t.empty())
    {
        return false;
    }
    const float parsed = std::strtof(t.c_str(), &end_ptr);
    if (end_ptr == t.c_str() || *end_ptr != '\0')
    {
        return false;
    }
    value = parsed;
    return true;
}

inline bool parse_4dof_arm_alias(const std::string &alias, std::string &canonical)
{
    const std::string a = to_upper_copy(trim_copy(alias));
    if (a == "0" || a == "L" || a == "LEFT" || a == "左" || a == "左臂")
    {
        canonical = "0";
        return true;
    }
    if (a == "1" || a == "R" || a == "RIGHT" || a == "右" || a == "右臂")
    {
        canonical = "1";
        return true;
    }
    return false;
}

inline std::string join_command(const std::vector<std::string> &parts)
{
    std::ostringstream oss;
    for (size_t i = 0; i < parts.size(); ++i)
    {
        if (i > 0)
        {
            oss << ',';
        }
        oss << parts[i];
    }
    return oss.str();
}

inline bool validate_xyz(const std::vector<std::string> &tokens, size_t begin)
{
    if (tokens.size() != begin + 3)
    {
        return false;
    }
    for (size_t i = begin; i < tokens.size(); ++i)
    {
        float value = 0.0f;
        if (!parse_float_token(tokens[i], value))
        {
            return false;
        }
    }
    return true;
}

inline bool validate_dual_xyz(const std::vector<std::string> &tokens)
{
    if (tokens.size() != 7)
    {
        return false;
    }
    for (size_t i = 1; i < tokens.size(); ++i)
    {
        float value = 0.0f;
        if (!parse_float_token(tokens[i], value))
        {
            return false;
        }
    }
    return true;
}

inline bool build_low_level_command(const std::vector<std::string> &tokens,
                                    std::string &low_level,
                                    std::string &error)
{
    if (tokens.empty())
    {
        error = "empty command";
        return false;
    }

    const std::string &cmd = tokens[0];
    if (cmd == "PICK" || cmd == "取块")
    {
        std::string arm_id;
        if (tokens.size() != 5 || !parse_4dof_arm_alias(tokens[1], arm_id) || !validate_xyz(tokens, 2))
        {
            error = "PICK requires: PICK,ID,x,y,z";
            return false;
        }
        low_level = join_command({"PICK", arm_id, tokens[2], tokens[3], tokens[4]});
        return true;
    }

    if (cmd == "PLACE" || cmd == "放置" || cmd == "放块")
    {
        std::string arm_id;
        if (tokens.size() != 5 || !parse_4dof_arm_alias(tokens[1], arm_id) || !validate_xyz(tokens, 2))
        {
            error = "PLACE requires: PLACE,ID,x,y,z";
            return false;
        }
        low_level = join_command({"PLACE", arm_id, tokens[2], tokens[3], tokens[4]});
        return true;
    }

    if (cmd == "PUTBACK" || cmd == "放回背部")
    {
        std::string arm_id;
        if (tokens.size() != 2 || !parse_4dof_arm_alias(tokens[1], arm_id))
        {
            error = "PUTBACK requires: PUTBACK,ID";
            return false;
        }
        low_level = join_command({"PUTBACK", arm_id});
        return true;
    }

    if (cmd == "GETBACK" || cmd == "背部取块")
    {
        std::string arm_id;
        if (tokens.size() != 2 || !parse_4dof_arm_alias(tokens[1], arm_id))
        {
            error = "GETBACK requires: GETBACK,ID";
            return false;
        }
        low_level = join_command({"GETBACK", arm_id});
        return true;
    }

    if (cmd == "PICKALL" || cmd == "双臂取块")
    {
        if (!validate_dual_xyz(tokens))
        {
            error = "PICKALL requires: PICKALL,lx,ly,lz,rx,ry,rz";
            return false;
        }
        low_level = join_command(tokens);
        return true;
    }

    if (cmd == "PLACEALL" || cmd == "双臂放置" || cmd == "双臂放块")
    {
        if (!validate_dual_xyz(tokens))
        {
            error = "PLACEALL requires: PLACEALL,lx,ly,lz,rx,ry,rz";
            return false;
        }
        low_level = join_command(tokens);
        return true;
    }

    if (cmd == "PUTBACKALL" || cmd == "双臂放回背部")
    {
        if (tokens.size() != 1)
        {
            error = "PUTBACKALL takes no arguments";
            return false;
        }
        low_level = "PUTBACKALL";
        return true;
    }

    if (cmd == "GETBACKALL" || cmd == "双臂背部取块")
    {
        if (tokens.size() != 1)
        {
            error = "GETBACKALL takes no arguments";
            return false;
        }
        low_level = "GETBACKALL";
        return true;
    }

    error = "unknown mission command";
    return false;
}

enum class MissionCommandAction
{
    Ignore,
    Accept,
    Busy,
    Invalid,
};

struct MissionCommandResult
{
    MissionCommandAction action = MissionCommandAction::Ignore;
    std::string low_level;
    std::string feedback;
    std::string error;
};

struct MissionStateResult
{
    bool completed = false;
    std::string feedback;
    std::string completed_command;
};

class ArmMissionController
{
public:
    /// @brief 设置超时时间（毫秒），默认 3000ms。设为 0 或负数则禁用超时。
    void set_timeout_ms(int ms)
    {
        timeout_ms_ = ms > 0 ? ms : 0;
    }

    MissionCommandResult handle_mission_command(const std::string &data)
    {
        const std::vector<std::string> tokens = tokenize_command(data);
        if (tokens.empty() || tokens[0].rfind("FEEDBACK:", 0) == 0)
        {
            return {};
        }

        if (busy_)
        {
            MissionCommandResult result;
            result.action = MissionCommandAction::Busy;
            result.feedback = "FEEDBACK:BUSY";
            return result;
        }

        MissionCommandResult result;
        std::string low_level;
        std::string error;
        if (!build_low_level_command(tokens, low_level, error))
        {
            result.action = MissionCommandAction::Invalid;
            result.error = error;
            return result;
        }

        busy_ = true;
        active_command_ = low_level;
        command_start_time_ = std::chrono::steady_clock::now();
        result.action = MissionCommandAction::Accept;
        result.low_level = low_level;
        return result;
    }

    MissionStateResult handle_arm_state(const std::string &state)
    {
        if (!busy_)
        {
            return {};
        }

        if (state == "DONE")
        {
            MissionStateResult result;
            result.completed = true;
            result.feedback = "FEEDBACK:DONE";
            result.completed_command = active_command_;
            busy_ = false;
            active_command_.clear();
            return result;
        }

        if (state.rfind("DIAG,", 0) == 0)
        {
            MissionStateResult result;
            result.completed = true;
            result.feedback = "FEEDBACK:REJECTED:" + extract_diag_reason(state);
            result.completed_command = active_command_;
            busy_ = false;
            active_command_.clear();
            return result;
        }

        return {};
    }

    /// @brief 检查是否超时。若 busy 且超时，返回等同于 DONE 的结果并清除 busy。
    /// @retval 未超时或未 busy 时 completed=false
    MissionStateResult check_timeout()
    {
        if (!busy_ || timeout_ms_ <= 0)
        {
            return {};
        }

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - command_start_time_).count();
        if (elapsed < timeout_ms_)
        {
            return {};
        }

        MissionStateResult result;
        result.completed = true;
        result.feedback = "FEEDBACK:DONE";
        result.completed_command = active_command_;
        busy_ = false;
        active_command_.clear();
        return result;
    }

    bool busy() const
    {
        return busy_;
    }

    const std::string &active_command() const
    {
        return active_command_;
    }

private:
    static std::string extract_diag_reason(const std::string &state)
    {
        const std::string key = "reason=";
        const size_t begin = state.find(key);
        if (begin == std::string::npos)
        {
            return "UNKNOWN";
        }
        const size_t value_begin = begin + key.size();
        const size_t value_end = state.find(',', value_begin);
        return state.substr(value_begin,
                            value_end == std::string::npos ? std::string::npos
                                                            : value_end - value_begin);
    }

    bool busy_ = false;
    std::string active_command_;
    int timeout_ms_ = 3000;  ///< 默认 3 秒超时
    std::chrono::steady_clock::time_point command_start_time_;
};

} // namespace dogvision_arm

#endif // DOGVISION_ARM_ARM_MISSION_PROTOCOL_HPP
