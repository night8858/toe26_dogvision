// ╔═══════════════════════════════════════════════════════════════╗
// ║  BB/4DOF 双臂协议 — 帧解析 / 发送命令 / 文本命令解析            ║
// ╚═══════════════════════════════════════════════════════════════╝
//
//  协议约定：帧头 0xBB，帧尾 FF EE，CRC8 多项式 0x07
//
//  BB 01  ← 周期位姿反馈（46 字节）
//  BB 02  → 4DOF 位姿控制（22 字节）
//  BB 03  → 预设动作触发（6 字节）
//  BB 04  → 电磁阀控制（7 字节）
//  BB 05  → 答案/语音（8 字节）
//  BB 06  → 气泵控制（10 字节）
//  BB 08  ← 动作拒绝/裁剪诊断（81 字节）
//  BB 11  → 单臂取块（18 字节，xyz 单位 m）
//  BB 12  → 单臂放块（18 字节，xyz 单位 m）
//  BB 14  → 单臂放回背部（6 字节）
//  BB 15  → 单臂从背部取块（6 字节）
//  BB 21  → 双臂取块（29 字节，xyz 单位 m）
//  BB 22  → 双臂放回背部（5 字节）
//  BB 23  → 双臂放块（29 字节，xyz 单位 m）
//  BB 24  → 双臂从背部取块（5 字节）
//  BB 99  → 带偏移启动（17 字节，offset 单位 mm）
//  BB CC  ← 动作完成事件（5 字节）
// ============================================================

#include <dogvision_arm/arm_internation.hpp>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

static bool is_finite_xyz(float x, float y, float z)
{
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

static int16_t decode_i16_le_local(const uint8_t *src)
{
    const uint16_t raw = static_cast<uint16_t>(src[0]) |
                         (static_cast<uint16_t>(src[1]) << 8);
    return static_cast<int16_t>(raw);
}

static std::string frame_to_hex(const uint8_t *data, size_t len)
{
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (size_t i = 0; i < len; ++i)
    {
        if (i > 0) oss << ' ';
        oss << std::setw(2) << static_cast<unsigned>(data[i]);
    }
    return oss.str();
}

// ════════════════════════════════════════════════════════════════
//  反馈帧解析
// ════════════════════════════════════════════════════════════════

bool arm_internation::parse_4dof_feedback_frame()
{
    while (rx_len_ >= 2)
    {
        size_t start = 0;
        while (start + 1 < rx_len_ && rx_buf_[start] != kHeadB) ++start;

        if (start + 1 >= rx_len_)
        {
            if (start > 0)
            {
                rx_len_ -= start;
                std::memmove(rx_buf_, rx_buf_ + start, rx_len_);
            }
            return false;
        }

        const uint8_t frame_cmd = rx_buf_[start + 1];

        if (frame_cmd == kCmdFb)
        {
            if (start + k4DofFbFrameLen > rx_len_)
            {
                if (start > 0)
                {
                    rx_len_ -= start;
                    std::memmove(rx_buf_, rx_buf_ + start, rx_len_);
                }
                return false;
            }

            const bool tail_ok = rx_buf_[start + 43] == kTailA
                              && rx_buf_[start + 44] == kTailB;
            const bool crc_ok = tail_ok
                && calc_crc8(rx_buf_ + start, 45) == rx_buf_[start + 45];

            if (crc_ok)
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                dof4_pose_float_[0].x = decode_float_le(rx_buf_ + start + 2);
                dof4_pose_float_[0].y = decode_float_le(rx_buf_ + start + 6);
                dof4_pose_float_[0].z = decode_float_le(rx_buf_ + start + 10);
                dof4_pose_float_[0].pitch = decode_float_le(rx_buf_ + start + 14);
                dof4_pose_float_[1].x = decode_float_le(rx_buf_ + start + 18);
                dof4_pose_float_[1].y = decode_float_le(rx_buf_ + start + 22);
                dof4_pose_float_[1].z = decode_float_le(rx_buf_ + start + 26);
                dof4_pose_float_[1].pitch = decode_float_le(rx_buf_ + start + 30);

                arm_pos_float_[0].x = dof4_pose_float_[0].x;
                arm_pos_float_[0].y = dof4_pose_float_[0].y;
                arm_pos_float_[1].x = dof4_pose_float_[1].x;
                arm_pos_float_[1].y = dof4_pose_float_[1].y;
                arm_pos_float_[2] = ArmEndPosFloat{};
                arm_pos_float_[3] = ArmEndPosFloat{};

                for (int i = 0; i < 4; ++i)
                {
                    sensor_.valve[i] = (rx_buf_[start + 34 + i] & 0x01u) != 0;
                    sensor_.microswitch[i] = (rx_buf_[start + 38 + i] & 0x01u) != 0;
                }

                rx_len_ -= (start + k4DofFbFrameLen);
                std::memmove(rx_buf_, rx_buf_ + start + k4DofFbFrameLen, rx_len_);
                return true;
            }

            rx_len_ -= (start + 1);
            std::memmove(rx_buf_, rx_buf_ + start + 1, rx_len_);
            continue;
        }



        if (frame_cmd == kCmd4DofDiagnostic)
        {
            if (start + k4DofDiagnosticFrameLen > rx_len_)
            {
                if (start > 0)
                {
                    rx_len_ -= start;
                    std::memmove(rx_buf_, rx_buf_ + start, rx_len_);
                }
                return false;
            }

            const bool tail_ok = rx_buf_[start + 78] == kTailA
                              && rx_buf_[start + 79] == kTailB;
            const bool crc_ok = tail_ok
                && calc_crc8(rx_buf_ + start, 80) == rx_buf_[start + 80];

            if (crc_ok)
            {
                Arm4DofDiagnostic diagnostic{};
                diagnostic.arm_id = rx_buf_[start + 2];
                diagnostic.mode = rx_buf_[start + 3];
                diagnostic.reason = rx_buf_[start + 4];
                diagnostic.joint_mask = rx_buf_[start + 5];
                diagnostic.requested_pose.x = decode_float_le(rx_buf_ + start + 6);
                diagnostic.requested_pose.y = decode_float_le(rx_buf_ + start + 10);
                diagnostic.requested_pose.z = decode_float_le(rx_buf_ + start + 14);
                diagnostic.requested_pose.pitch = decode_float_le(rx_buf_ + start + 18);
                for (int i = 0; i < 4; ++i)
                {
                    diagnostic.requested_joints[i] = decode_float_le(rx_buf_ + start + 22 + i * 4);
                    diagnostic.limited_joints[i] = decode_float_le(rx_buf_ + start + 38 + i * 4);
                    diagnostic.target_servo_pos[i] = decode_i16_le_local(rx_buf_ + start + 70 + i * 2);
                }
                diagnostic.limited_pose.x = decode_float_le(rx_buf_ + start + 54);
                diagnostic.limited_pose.y = decode_float_le(rx_buf_ + start + 58);
                diagnostic.limited_pose.z = decode_float_le(rx_buf_ + start + 62);
                diagnostic.limited_pose.pitch = decode_float_le(rx_buf_ + start + 66);

                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    last_diagnostic_ = diagnostic;
                    ++pending_diagnostic_feedback_count_;
                }

                rx_len_ -= (start + k4DofDiagnosticFrameLen);
                std::memmove(rx_buf_, rx_buf_ + start + k4DofDiagnosticFrameLen, rx_len_);
                return true;
            }

            rx_len_ -= (start + 1);
            std::memmove(rx_buf_, rx_buf_ + start + 1, rx_len_);
            continue;
        }

        if (frame_cmd == kCmd4DofDone)
        {
            if (start + k4DofDoneFrameLen > rx_len_)
            {
                if (start > 0)
                {
                    rx_len_ -= start;
                    std::memmove(rx_buf_, rx_buf_ + start, rx_len_);
                }
                return false;
            }

            const bool tail_ok = rx_buf_[start + 2] == kTailA
                              && rx_buf_[start + 3] == kTailB;
            const bool crc_ok = tail_ok
                && calc_crc8(rx_buf_ + start, 4) == rx_buf_[start + 4];

            if (crc_ok)
            {
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    ++pending_done_feedback_count_;
                }
                rx_len_ -= (start + k4DofDoneFrameLen);
                std::memmove(rx_buf_, rx_buf_ + start + k4DofDoneFrameLen, rx_len_);
                return true;
            }

            rx_len_ -= (start + 1);
            std::memmove(rx_buf_, rx_buf_ + start + 1, rx_len_);
            continue;
        }

        rx_len_ -= (start + 1);
        std::memmove(rx_buf_, rx_buf_ + start + 1, rx_len_);
    }
    return false;
}

// ════════════════════════════════════════════════════════════════
//  发送命令
// ════════════════════════════════════════════════════════════════

bool arm_internation::send_valve_cmd(int valve_id, bool state)
{
    if (valve_id < 0 || valve_id > 3) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[7] = {kHeadB, kCmdValv, static_cast<uint8_t>(valve_id),
                      static_cast<uint8_t>(state), kTailA, kTailB, 0};
    buf[6] = calc_crc8(buf, 6);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_answer_cmd(uint8_t answer)
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[8] = {kHeadB, kCmdAns, answer, 0, 0, kTailA, kTailB, 0};
    buf[7] = calc_crc8(buf, 7);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_pump_cmd(bool on, int speed)
{
    if (speed < 0) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[10] = {kHeadB, kCmdPump,
                       on ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                       0, 0, 0, 0, kTailA, kTailB, 0};
    encode_float_le(static_cast<float>(speed), buf + 3);
    buf[9] = calc_crc8(buf, 9);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_4dof_pose_cmd(int arm_id, float x, float y, float z, float pitch)
{
    if (arm_id < 0 || arm_id > 1 || !std::isfinite(x) || !std::isfinite(y)
        || !std::isfinite(z) || !std::isfinite(pitch)) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[22] = {kHeadB, kCmdArm, static_cast<uint8_t>(arm_id),
                       0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, kTailA, kTailB, 0};
    encode_float_le(x, buf + 3);  encode_float_le(y, buf + 7);
    encode_float_le(z, buf + 11); encode_float_le(pitch, buf + 15);
    buf[21] = calc_crc8(buf, 21);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_4dof_action_cmd(uint8_t action_id)
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[6] = {kHeadB, kCmdAction, action_id, kTailA, kTailB, 0};
    buf[5] = calc_crc8(buf, 5);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_4dof_start_cmd(float ox, float oy, float oz)
{
    if (!std::isfinite(ox) || !std::isfinite(oy) || !std::isfinite(oz)) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[17] = {kHeadB, kCmd4DofStart,
                       0,0,0,0, 0,0,0,0, 0,0,0,0, kTailA, kTailB, 0};
    encode_float_le(ox, buf + 2); encode_float_le(oy, buf + 6); encode_float_le(oz, buf + 10);
    buf[16] = calc_crc8(buf, 16);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_4dof_pick_cmd(int arm_id, float x, float y, float z)
{
    if (arm_id < 0 || arm_id > 1 || !is_finite_xyz(x, y, z)) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[18] = {kHeadB, kCmd4DofPICK, static_cast<uint8_t>(arm_id),
                       0,0,0,0, 0,0,0,0, 0,0,0,0, kTailA, kTailB, 0};
    encode_float_le(x, buf + 3); encode_float_le(y, buf + 7); encode_float_le(z, buf + 11);
    buf[17] = calc_crc8(buf, 17);
    const bool ok = write_bytes(buf, sizeof(buf));
    std::cerr << "[arm_internation] TX BB 11 PICK arm=" << arm_id
              << " xyz=" << x << "," << y << "," << z
              << " crc=0x" << std::hex << std::uppercase << std::setw(2) << std::setfill('0')
              << static_cast<unsigned>(buf[17]) << std::dec << std::setfill(' ')
              << " ok=" << (ok ? "true" : "false")
              << " frame=" << frame_to_hex(buf, sizeof(buf)) << std::endl;
    return ok;
}

bool arm_internation::send_4dof_place_cmd(int arm_id, float x, float y, float z)
{
    if (arm_id < 0 || arm_id > 1 || !is_finite_xyz(x, y, z)) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[18] = {kHeadB, kCmd4DofPLACE, static_cast<uint8_t>(arm_id),
                       0,0,0,0, 0,0,0,0, 0,0,0,0, kTailA, kTailB, 0};
    encode_float_le(x, buf + 3); encode_float_le(y, buf + 7); encode_float_le(z, buf + 11);
    buf[17] = calc_crc8(buf, 17);
    const bool ok = write_bytes(buf, sizeof(buf));
    std::cerr << "[arm_internation] TX BB 12 PLACE arm=" << arm_id
              << " xyz=" << x << "," << y << "," << z
              << " crc=0x" << std::hex << std::uppercase << std::setw(2) << std::setfill('0')
              << static_cast<unsigned>(buf[17]) << std::dec << std::setfill(' ')
              << " ok=" << (ok ? "true" : "false")
              << " frame=" << frame_to_hex(buf, sizeof(buf)) << std::endl;
    return ok;
}

bool arm_internation::send_4dof_put_block_back_cmd(int arm_id)
{
    if (arm_id < 0 || arm_id > 1) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[6] = {kHeadB, kCmd4DofPUT_BLOCK_BACK, static_cast<uint8_t>(arm_id),
                      kTailA, kTailB, 0};
    buf[5] = calc_crc8(buf, 5);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_4dof_get_block_back_cmd(int arm_id)
{
    if (arm_id < 0 || arm_id > 1) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[6] = {kHeadB, kCmd4DofGET_BLOCK_BACK, static_cast<uint8_t>(arm_id),
                      kTailA, kTailB, 0};
    buf[5] = calc_crc8(buf, 5);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_4dof_pick_all_cmd(float lx, float ly, float lz,
                                             float rx, float ry, float rz)
{
    if (!is_finite_xyz(lx, ly, lz) || !is_finite_xyz(rx, ry, rz)) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[29] = {kHeadB, kCmd4DofPICK_ALL,
                       0,0,0,0, 0,0,0,0, 0,0,0,0,
                       0,0,0,0, 0,0,0,0, 0,0,0,0, kTailA, kTailB, 0};
    encode_float_le(lx, buf + 2);  encode_float_le(ly, buf + 6);
    encode_float_le(lz, buf + 10); encode_float_le(rx, buf + 14);
    encode_float_le(ry, buf + 18); encode_float_le(rz, buf + 22);
    buf[28] = calc_crc8(buf, 28);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_4dof_put_block_back_all_cmd()
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[5] = {kHeadB, kCmd4DofPUT_BLOCK_BACK_ALL, kTailA, kTailB, 0};
    buf[4] = calc_crc8(buf, 4);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_4dof_place_all_cmd(float lx, float ly, float lz,
                                              float rx, float ry, float rz)
{
    if (!is_finite_xyz(lx, ly, lz) || !is_finite_xyz(rx, ry, rz)) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[29] = {kHeadB, kCmd4DofPLACE_ALL,
                       0,0,0,0, 0,0,0,0, 0,0,0,0,
                       0,0,0,0, 0,0,0,0, 0,0,0,0, kTailA, kTailB, 0};
    encode_float_le(lx, buf + 2);  encode_float_le(ly, buf + 6);
    encode_float_le(lz, buf + 10); encode_float_le(rx, buf + 14);
    encode_float_le(ry, buf + 18); encode_float_le(rz, buf + 22);
    buf[28] = calc_crc8(buf, 28);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_4dof_get_block_back_all_cmd()
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[5] = {kHeadB, kCmd4DofGET_BLOCK_BACK_ALL, kTailA, kTailB, 0};
    buf[4] = calc_crc8(buf, 4);
    return write_bytes(buf, sizeof(buf));
}

// ════════════════════════════════════════════════════════════════
//  文本命令解析（BB/4DOF）
// ════════════════════════════════════════════════════════════════

bool arm_internation::handle_text_command_bb(const std::vector<std::string>& tokens)
{
    if (tokens.empty()) return false;
    const std::string cmd = to_upper_copy(tokens[0]);

    auto parse_xyz_tokens = [&](size_t begin, float &x, float &y, float &z) -> bool
    {
        bool has_x = false, has_y = false, has_z = false;
        for (size_t i = begin; i < tokens.size(); ++i)
        {
            float value = 0.0f;
            if (parse_float_after_prefix(tokens[i], "X", value))
                { x = value; has_x = true; continue; }
            if (parse_float_after_prefix(tokens[i], "Y", value))
                { y = value; has_y = true; continue; }
            if (parse_float_after_prefix(tokens[i], "Z", value))
                { z = value; has_z = true; continue; }
            if (parse_float_token(tokens[i], value))
            {
                if (!has_x)      { x = value; has_x = true; }
                else if (!has_y) { y = value; has_y = true; }
                else if (!has_z) { z = value; has_z = true; }
            }
        }
        return has_x && has_y && has_z;
    };

    auto parse_dual_xyz_tokens = [&](size_t begin,
                                     float &lx, float &ly, float &lz,
                                     float &rx, float &ry, float &rz) -> bool
    {
        bool has_lx = false, has_ly = false, has_lz = false;
        bool has_rx = false, has_ry = false, has_rz = false;
        for (size_t i = begin; i < tokens.size(); ++i)
        {
            float value = 0.0f;
            if (parse_float_after_prefix(tokens[i], "LX", value))
                { lx = value; has_lx = true; continue; }
            if (parse_float_after_prefix(tokens[i], "LY", value))
                { ly = value; has_ly = true; continue; }
            if (parse_float_after_prefix(tokens[i], "LZ", value))
                { lz = value; has_lz = true; continue; }
            if (parse_float_after_prefix(tokens[i], "RX", value))
                { rx = value; has_rx = true; continue; }
            if (parse_float_after_prefix(tokens[i], "RY", value))
                { ry = value; has_ry = true; continue; }
            if (parse_float_after_prefix(tokens[i], "RZ", value))
                { rz = value; has_rz = true; continue; }
            if (parse_float_token(tokens[i], value))
            {
                if (!has_lx)      { lx = value; has_lx = true; }
                else if (!has_ly) { ly = value; has_ly = true; }
                else if (!has_lz) { lz = value; has_lz = true; }
                else if (!has_rx) { rx = value; has_rx = true; }
                else if (!has_ry) { ry = value; has_ry = true; }
                else if (!has_rz) { rz = value; has_rz = true; }
            }
        }
        return has_lx && has_ly && has_lz && has_rx && has_ry && has_rz;
    };

    // ---- START (BB 99) ----
    if (cmd == "START" || cmd == "启动")
    {
        if (tokens.size() < 4) return false;
        float ox = 0.0f, oy = 0.0f, oz = 0.0f;
        bool has_x = false, has_y = false, has_z = false;
        for (size_t i = 1; i < tokens.size(); ++i)
        {
            float value = 0.0f;
            if (parse_float_after_prefix(tokens[i], "X", value))
                { ox = value; has_x = true; continue; }
            if (parse_float_after_prefix(tokens[i], "Y", value))
                { oy = value; has_y = true; continue; }
            if (parse_float_after_prefix(tokens[i], "Z", value))
                { oz = value; has_z = true; continue; }
            if (parse_float_token(tokens[i], value))
            {
                if (!has_x)      { ox = value; has_x = true; }
                else if (!has_y) { oy = value; has_y = true; }
                else if (!has_z) { oz = value; has_z = true; }
            }
        }
        if (!has_x || !has_y || !has_z) return false;
        return send_4dof_start_cmd(ox, oy, oz);
    }

    // ---- PICK (BB 11) ----
    if (cmd == "PICK" || cmd == "4PICK" || cmd == "取块" || cmd == "4取块")
    {
        if (tokens.size() < 5) return false;
        int arm_id = -1;
        if (!parse_4dof_arm_alias(tokens[1], arm_id)) return false;
        float x = 0, y = 0, z = 0;
        if (!parse_xyz_tokens(2, x, y, z)) return false;
        return send_4dof_pick_cmd(arm_id, x, y, z);
    }

    // ---- PLACE (BB 12) ----
    if (cmd == "PLACE" || cmd == "4PLACE" || cmd == "放置" || cmd == "放块" || cmd == "4放置")
    {
        if (tokens.size() < 5) return false;
        int arm_id = -1;
        if (!parse_4dof_arm_alias(tokens[1], arm_id)) return false;
        float x = 0, y = 0, z = 0;
        if (!parse_xyz_tokens(2, x, y, z)) return false;
        return send_4dof_place_cmd(arm_id, x, y, z);
    }

    // ---- PUTBACK (BB 14) ----
    if (cmd == "PUTBACK" || cmd == "4PUTBACK" || cmd == "放回背部" || cmd == "4放回背部")
    {
        if (tokens.size() < 2) return false;
        int arm_id = -1;
        if (!parse_4dof_arm_alias(tokens[1], arm_id)) return false;
        return send_4dof_put_block_back_cmd(arm_id);
    }

    // ---- GETBACK (BB 15) ----
    if (cmd == "GETBACK" || cmd == "4GETBACK" || cmd == "背部取块" || cmd == "4背部取块")
    {
        if (tokens.size() < 2) return false;
        int arm_id = -1;
        if (!parse_4dof_arm_alias(tokens[1], arm_id)) return false;
        return send_4dof_get_block_back_cmd(arm_id);
    }

    // ---- PICKALL (BB 21) ----
    if (cmd == "PICKALL" || cmd == "4PICKALL" || cmd == "双臂取块" || cmd == "4双臂取块")
    {
        if (tokens.size() < 7) return false;
        float lx = 0, ly = 0, lz = 0, rx = 0, ry = 0, rz = 0;
        if (!parse_dual_xyz_tokens(1, lx, ly, lz, rx, ry, rz)) return false;
        return send_4dof_pick_all_cmd(lx, ly, lz, rx, ry, rz);
    }

    // ---- PLACEALL (BB 23) ----
    if (cmd == "PLACEALL" || cmd == "4PLACEALL" || cmd == "双臂放置" || cmd == "双臂放块" || cmd == "4双臂放置")
    {
        if (tokens.size() < 7) return false;
        float lx = 0, ly = 0, lz = 0, rx = 0, ry = 0, rz = 0;
        if (!parse_dual_xyz_tokens(1, lx, ly, lz, rx, ry, rz)) return false;
        return send_4dof_place_all_cmd(lx, ly, lz, rx, ry, rz);
    }

    // ---- PUTBACKALL (BB 22) ----
    if (cmd == "PUTBACKALL" || cmd == "4PUTBACKALL" || cmd == "双臂放回背部" || cmd == "4双臂放回背部")
        return send_4dof_put_block_back_all_cmd();

    // ---- GETBACKALL (BB 24) ----
    if (cmd == "GETBACKALL" || cmd == "4GETBACKALL" || cmd == "双臂背部取块" || cmd == "4双臂背部取块")
        return send_4dof_get_block_back_all_cmd();

    // ---- 4POSE (BB 02) ----
    if (cmd == "4POSE" || cmd == "4P" || cmd == "DOF4POSE")
    {
        if (tokens.size() < 6) return false;
        int arm_id = -1;
        if (!parse_4dof_arm_alias(tokens[1], arm_id)) return false;
        float x = 0, y = 0, z = 0, pitch = 0;
        bool has_x = false, has_y = false, has_z = false, has_pitch = false;
        for (size_t i = 2; i < tokens.size(); ++i)
        {
            float value = 0.0f;
            if (parse_float_after_prefix(tokens[i], "X", value))
                { x = value; has_x = true; continue; }
            if (parse_float_after_prefix(tokens[i], "Y", value))
                { y = value; has_y = true; continue; }
            if (parse_float_after_prefix(tokens[i], "Z", value))
                { z = value; has_z = true; continue; }
            if (parse_float_after_prefix(tokens[i], "PITCH", value) ||
                parse_float_after_prefix(tokens[i], "P", value))
                { pitch = value; has_pitch = true; continue; }
            if (parse_float_token(tokens[i], value))
            {
                if (!has_x)      { x = value; has_x = true; }
                else if (!has_y) { y = value; has_y = true; }
                else if (!has_z) { z = value; has_z = true; }
                else if (!has_pitch) { pitch = value; has_pitch = true; }
            }
        }
        if (!has_x || !has_y || !has_z || !has_pitch) return false;
        return send_4dof_pose_cmd(arm_id, x, y, z, pitch);
    }

    // ---- 4ACT (BB 03) ----
    if (cmd == "4ACT" || cmd == "4ACTION" || cmd == "DOF4ACT")
    {
        if (tokens.size() < 2) return false;
        int action_id = -1;
        if (!parse_int_token(tokens[1], action_id) || action_id < 0 || action_id > 255)
            return false;
        return send_4dof_action_cmd(static_cast<uint8_t>(action_id));
    }

    // ---- V: 电磁阀 (BB 04) ----
    if (cmd == "V")
    {
        if (tokens.size() < 2) return false;
        bool state = false;
        bool has_state = false;
        if (tokens.size() >= 3)
        {
            const std::string s = to_upper_copy(tokens[2]);
            if (s == "1" || s == "ON" || s == "OPEN" || s == "TRUE" || s == "开")
                { state = true; has_state = true; }
            else if (s == "0" || s == "OFF" || s == "CLOSE" || s == "FALSE" || s == "关")
                { state = false; has_state = true; }
            else return false;
        }
        const std::string target = to_upper_copy(tokens[1]);
        if (target == "ALL" || target == "全部")
        {
            for (int vid = 0; vid < 4; ++vid)
            {
                bool ns = state;
                if (!has_state)
                {
                    std::lock_guard<std::mutex> lock(valve_cmd_mutex_);
                    ns = !valve_cmd_state_[vid];
                }
                if (!send_valve_cmd(vid, ns)) return false;
                std::lock_guard<std::mutex> lock(valve_cmd_mutex_);
                valve_cmd_state_[vid] = ns;
            }
            return true;
        }
        int vid = -1;
        if (!parse_int_token(tokens[1], vid) || vid < 0 || vid > 3) return false;
        if (!has_state)
        {
            std::lock_guard<std::mutex> lock(valve_cmd_mutex_);
            state = !valve_cmd_state_[vid];
        }
        if (!send_valve_cmd(vid, state)) return false;
        { std::lock_guard<std::mutex> lock(valve_cmd_mutex_); valve_cmd_state_[vid] = state; }
        return true;
    }

    // ---- A: 答案 (BB 05) ----
    if (cmd == "A")
    {
        if (tokens.size() < 2) return false;
        int answer = -1;
        if (!parse_int_token(tokens[1], answer) || answer < 0 || answer > 255) return false;
        return send_answer_cmd(static_cast<uint8_t>(answer));
    }

    // ---- P: 气泵 (BB 06) ----
    if (cmd == "P")
    {
        if (tokens.size() < 2) return false;
        const std::string action = to_upper_copy(tokens[1]);
        if (action == "ON" || action == "1" || action == "OPEN" || action == "TRUE")
        {
            int speed = 0;
            if (tokens.size() >= 3)
            {
                if (!parse_int_token(tokens[2], speed) || speed < 0) return false;
            }
            return send_pump_cmd(true, speed);
        }
        else if (action == "OFF" || action == "0" || action == "CLOSE" || action == "FALSE")
            return send_pump_cmd(false, 0);
        return false;
    }

    return false;
}
