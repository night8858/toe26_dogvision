// ╔═══════════════════════════════════════════════════════════════╗
// ║  CC 云台协议 — 发送命令 & 文本命令解析                          ║
// ╚═══════════════════════════════════════════════════════════════╝
//
//  CC 99 — 启动相机云台（5 字节）
//    [0] 0xCC  [1] 0x99  [2] 0xFF  [3] 0xEE  [4] CRC8([0]~[3])
//
//  CC 01 — 运动云台到目标位置（17 字节）
//    [0] 0xCC  [1] 0x01  [2-5] J1(float32 LE, 度)
//    [6-9] PITCH(float32 LE, 度)  [10-13] YAW(float32 LE, 度)
//    [14] 0xFF  [15] 0xEE  [16] CRC8([0]~[15])
//
//  文本命令：
//    CAM_START / GIMBAL_START / 云台启动
//    CAM_MOVE,j1,pitch,yaw / GIMBAL_MOVE,j1,pitch,yaw / 云台运动,j1,pitch,yaw
// ============================================================

#include <dogvision_arm/arm_internation.hpp>
#include <cmath>

namespace {
    constexpr uint8_t kHeadC          = 0xCCu;
    constexpr uint8_t kCmdGimbalStart = 0x99u;
    constexpr uint8_t kCmdGimbalMove  = 0x01u;
}

// ════════════════════════════════════════════════════════════════
//  发送命令
// ════════════════════════════════════════════════════════════════

bool arm_internation::send_cc_gimbal_start_cmd()
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[5] = {kHeadC, kCmdGimbalStart, kTailA, kTailB, 0};
    buf[4] = calc_crc8(buf, 4);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_cc_gimbal_move_cmd(float j1, float pitch, float yaw)
{
    if (!std::isfinite(j1) || !std::isfinite(pitch) || !std::isfinite(yaw))
        return false;

    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[17] = {kHeadC, kCmdGimbalMove,
                       0, 0, 0, 0,  0, 0, 0, 0,
                       0, 0, 0, 0,  kTailA, kTailB, 0};
    encode_float_le(j1,    buf + 2);
    encode_float_le(pitch, buf + 6);
    encode_float_le(yaw,   buf + 10);
    buf[16] = calc_crc8(buf, 16);
    return write_bytes(buf, sizeof(buf));
}

// ════════════════════════════════════════════════════════════════
//  文本命令解析
// ════════════════════════════════════════════════════════════════

bool arm_internation::handle_text_command_cc(const std::vector<std::string>& tokens)
{
    if (tokens.empty()) return false;
    const std::string cmd = to_upper_copy(tokens[0]);

    if (cmd == "CAM_START" || cmd == "GIMBAL_START" || cmd == "云台启动")
        return send_cc_gimbal_start_cmd();

    if (cmd == "CAM_MOVE" || cmd == "GIMBAL_MOVE" || cmd == "云台运动")
    {
        if (tokens.size() < 4) return false;
        float j1 = 0.0f, pitch = 0.0f, yaw = 0.0f;
        if (!parse_float_token(tokens[1], j1) ||
            !parse_float_token(tokens[2], pitch) ||
            !parse_float_token(tokens[3], yaw))
            return false;
        return send_cc_gimbal_move_cmd(j1, pitch, yaw);
    }

    return false;
}
