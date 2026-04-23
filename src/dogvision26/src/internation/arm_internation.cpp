#include "arm_internation.hpp"
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// ============================================================
//  协议帧格式说明（见头文件注释）
// ============================================================
//  AA 01 反馈帧结构：
//    [0]  0xAA
//    [1]  0x01
//    [2-9]    LF x(4B), y(4B)
//    [10-17]  RF x(4B), y(4B)
//    [18-25]  LB x(4B), y(4B)
//    [26-33]  RB x(4B), y(4B)
//    [34-37]  YAW（4B）
//    [38-41]  PITCH（4B）
//    [42]     电磁阀1/2状态
//    [43]     电磁阀3/4状态
//    [44]     微动1/2状态
//    [45]     微动3/4状态
//    [46]     0xFF
//    [47]     0xEE
//    [48]     CRC8（覆盖 [0]~[47]）
//  兼容扩展版本：在 [45] 与 [46] 之间可能插入 4 字节保留位，
//  则帧尾/CRC 变为 [50]=0xFF [51]=0xEE [52]=CRC8（覆盖 [0]~[51]）。
// ============================================================

// ============================================================
//  代码阅读导航（建议先看这段）
// ------------------------------------------------------------
//  A. 连接部分
//     open()            : 打开并配置指定串口
//     open_by_hwid()    : 自动扫描 ttyACM* 并按硬件 ID 连接（失败重试）
//
//  B. 接收部分
//     receive_once()    : 从串口读入字节到缓存
//     parse_feedback_frame()
//                       : 在缓存里找完整帧、校验 CRC、刷新状态缓存
//
//  C. 发送部分
//     send_arm_cmd() / send_gimbal_cmd() / send_valve_cmd()
//                       : 按协议打包并发送
//
//  D. 上层适配
//     handle_text_command()
//                       : 将字符串命令解析为具体发送函数调用
// ============================================================

arm_internation::arm_internation() {}
arm_internation::~arm_internation() { close(); }

namespace
{
    // 反馈帧兼容长度：
    // - 49B: 2 + 40(float) + 4(status) + 2(tail) + 1(crc)
    // - 53B: 49B 基础上额外 4 字节保留位（常见于下位机扩展版本）
    static constexpr size_t kFbFrameLenV1 = 49;
    static constexpr size_t kFbFrameLenV2 = 53;


    // 将常见整型波特率转换为 termios 的 speed_t 常量。
    // 说明：统一映射可以避免调用方直接传入平台相关宏值。

    bool baud_to_termios(int baud_rate, speed_t &speed)
    {
        switch (baud_rate)
        {
        case 9600:
            speed = B9600;
            return true;
        case 19200:
            speed = B19200;
            return true;
        case 38400:
            speed = B38400;
            return true;
        case 57600:
            speed = B57600;
            return true;
        case 115200:
            speed = B115200;
            return true;
        case 230400:
            speed = B230400;
            return true;
        case 460800:
            speed = B460800;
            return true;
        case 921600:
            speed = B921600;
            return true;
        default:
            return false;
        }
    }

    std::string to_lower_copy(std::string s)
    {
        // 使用无符号字符转换，避免高位字符导致未定义行为。
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return s;
    }

    std::string to_upper_copy(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                       { return static_cast<char>(std::toupper(c)); });
        return s;
    }

    std::string trim_copy(const std::string &s)
    {
        // 去掉前后空白，便于命令容错（例如 " G, 10 , 20 "）。
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

    std::vector<std::string> split_by_comma(const std::string &text)
    {
        // 以逗号拆分命令并逐段 trim，便于后续按 token 解析。
        std::vector<std::string> out;
        std::stringstream ss(text);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            out.push_back(trim_copy(item));
        }
        return out;
    }

    bool read_first_line(const std::filesystem::path &file_path, std::string &value)
    {
        // sysfs 里常见是单行文本，本函数统一读取并裁剪空白。
        std::ifstream in(file_path);
        if (!in.is_open())
        {
            return false;
        }
        std::getline(in, value);
        value = trim_copy(value);
        return !value.empty();
    }

    bool find_usb_vendor_product_for_tty(const std::string &tty_name, std::string &vendor, std::string &product)
    {
        // 思路：
        // 1) 从 /sys/class/tty/<tty>/device 出发
        // 2) 向父目录回溯，直到找到 idVendor/idProduct
        // 3) 读取两者用于硬件匹配
        std::filesystem::path current = std::filesystem::path("/sys/class/tty") / tty_name / "device";
        if (!std::filesystem::exists(current))
        {
            return false;
        }
        std::error_code ec;
        current = std::filesystem::weakly_canonical(current, ec);
        if (ec)
        {
            return false;
        }

        for (int i = 0; i < 10; ++i)
        {
            const std::filesystem::path vendor_path = current / "idVendor";
            const std::filesystem::path product_path = current / "idProduct";
            if (std::filesystem::exists(vendor_path) && std::filesystem::exists(product_path))
            {
                return read_first_line(vendor_path, vendor) && read_first_line(product_path, product);
            }
            if (!current.has_parent_path())
            {
                break;
            }
            current = current.parent_path();
        }
        return false;
    }

    // 在 /dev 目录筛选 ttyACM*，并按 idVendor:idProduct 匹配目标设备。
    bool parse_hw_id(const std::string &hw_id, std::string &vendor, std::string &product)
    {
        // 期望格式："0483:5740"
        const auto pos = hw_id.find(':');
        if (pos == std::string::npos)
        {
            return false;
        }
        vendor = to_lower_copy(trim_copy(hw_id.substr(0, pos)));
        product = to_lower_copy(trim_copy(hw_id.substr(pos + 1)));
        return !vendor.empty() && !product.empty();
    }

    bool parse_int_token(const std::string &token, int &value)
    {
        // 使用 strtol 做严格数值解析，要求整个 token 都是整数字符串。
        std::string t = trim_copy(token);
        if (t.empty())
        {
            return false;
        }
        char *end_ptr = nullptr;
        long parsed = std::strtol(t.c_str(), &end_ptr, 10);
        if (end_ptr == t.c_str() || *end_ptr != '\0')
        {
            return false;
        }
        value = static_cast<int>(parsed);
        return true;
    }

    bool parse_float_token_impl(const std::string &token, float &value)
    {
        // 使用 strtof 做严格数值解析，要求整个 token 都是浮点数字符串。
        std::string t = trim_copy(token);
        if (t.empty())
        {
            return false;
        }
        char *end_ptr = nullptr;
        const float parsed = std::strtof(t.c_str(), &end_ptr);
        if (end_ptr == t.c_str() || *end_ptr != '\0' || !std::isfinite(parsed))
        {
            return false;
        }
        value = parsed;
        return true;
    }

    // 从字符串 token 提取浮点数。
    union FloatBytes
    {
        float f;
        uint8_t b[4];
    };

    // 协议按小端存储 float 字节序；上下位机均为常见小端平台时可直接互通。
    float decode_float_le(const uint8_t *src)
    {
        FloatBytes fb{};
        fb.b[0] = src[0];
        fb.b[1] = src[1];
        fb.b[2] = src[2];
        fb.b[3] = src[3];
        return fb.f;
    }

    void encode_float_le(float value, uint8_t *dst)
    {
        FloatBytes fb{};
        fb.f = value;
        dst[0] = fb.b[0];
        dst[1] = fb.b[1];
        dst[2] = fb.b[2];
        dst[3] = fb.b[3];
    }

    int16_t float_to_scaled_int16(float value, float scale)
    {
        if (!(scale > 0.0f) || !std::isfinite(value))
        {
            return 0;
        }
        const float raw = value / scale;
        const float lo = static_cast<float>(std::numeric_limits<int16_t>::min());
        const float hi = static_cast<float>(std::numeric_limits<int16_t>::max());
        const float clamped = std::max(lo, std::min(hi, raw));
        return static_cast<int16_t>(std::lround(clamped));
    }

    bool is_disconnect_errno(int err)
    {
        return err == EIO || err == ENODEV || err == ENXIO || err == EBADF || err == EPIPE;
    }

} // namespace

bool arm_internation::open(const std::string &port, int baud_rate)
{
    // 先关后开，避免重复打开同一对象的 fd 导致泄漏或状态不一致。
    close();

    speed_t speed = B115200;
    if (!baud_to_termios(baud_rate, speed))
    {
        std::cerr << "[arm_internation] Unsupported baud rate: " << baud_rate << std::endl;
        return false;
    }

    // O_NOCTTY: 防止该串口成为控制终端；O_SYNC: 写入尽快落到设备层。
    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0)
    {
        std::cerr << "[arm_internation] Failed to open " << port << std::endl;
        return false;
    }
    struct termios tty{};
    if (tcgetattr(fd_, &tty) != 0)
    {
        std::cerr << "[arm_internation] tcgetattr failed" << std::endl;
        close();
        return false;
    }
    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    if (tcsetattr(fd_, TCSANOW, &tty) != 0)
    {
        std::cerr << "[arm_internation] tcsetattr failed" << std::endl;
        close();
        return false;
    }
    return true;
}

bool arm_internation::open_by_HWid(const std::string &hw_id, int baud_rate, int retry_ms)
{
    reconnect_hw_id_ = hw_id;
    reconnect_baud_rate_ = baud_rate;
    reconnect_retry_ms_ = retry_ms > 0 ? retry_ms : 1000;
    auto_reconnect_enabled_.store(true);

    // 常驻重试策略：该函数只有在成功连接后才返回。
    // 适用于“设备可能晚插入”或“启动时串口尚未就绪”的场景。
    const int wait_ms = reconnect_retry_ms_;
    while (true)
    {
        const std::string dev = find_ttyacm_by_HWid(hw_id);
        if (!dev.empty() && open(dev, baud_rate))
        {
            std::cerr << "[arm_internation] Connected to " << dev << " for HW ID " << hw_id << std::endl;
            return true;
        }
        std::cerr << "[arm_internation] Cannot find/open HW ID " << hw_id
                  << ", retry in " << wait_ms << " ms" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    }
}

void arm_internation::close()
{
    if (fd_ >= 0)
    {
        ::close(fd_);
        fd_ = -1;
    }
}

bool arm_internation::is_open() const { return fd_ >= 0; }

// ---- CRC8/SMBUS ----
uint8_t arm_internation::calc_crc8(const uint8_t *data, size_t len)
{
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; ++i)
    {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j)
            crc = (crc & 0x80) ? (crc << 1) ^ 0x07 : (crc << 1);
    }
    return crc;
}

// ---- 解析反馈帧 ----
bool arm_internation::parse_feedback_frame()
{
    // 注意：接收缓存是“流式字节”，不保证 read 一次就是一帧。
    // 这里在同一次调用内持续重同步，直到找到有效帧或缓存不足最短帧。
    while (rx_len_ >= kFbFrameLenV1)
    {
        // 查找帧头
        size_t start = 0;
        while (start + kFbFrameLenV1 <= rx_len_ &&
               (rx_buf_[start] != kHeadA || rx_buf_[start + 1] != kCmdFb))
        {
            ++start;
        }

        // 没有足够字节构成完整候选帧：保留剩余字节，等待后续拼接。
        if (start + kFbFrameLenV1 > rx_len_)
        {
            if (start > 0)
            {
                rx_len_ -= start;
                std::memmove(rx_buf_, rx_buf_ + start, rx_len_);
            }
            return false;
        }

        // 同时尝试 49B 与 53B 两种帧长，选择 tail+crc 校验通过的版本。
        bool full_ok = false;
        size_t frame_len = 0;
        size_t tail_a_idx = 0;
        size_t tail_b_idx = 0;
        size_t crc_idx = 0;

        auto try_layout = [&](size_t candidate_len, size_t tail_a, size_t tail_b, size_t crc) -> bool {
            if (start + candidate_len > rx_len_) {
                return false;
            }
            if (rx_buf_[start + tail_a] != kTailA || rx_buf_[start + tail_b] != kTailB) {
                return false;
            }
            return calc_crc8(rx_buf_ + start, crc) == rx_buf_[start + crc];
        };

        // 优先旧版 49B，再尝试扩展 53B。
        if (try_layout(kFbFrameLenV1, 46, 47, 48)) {
            full_ok = true;
            frame_len = kFbFrameLenV1;
            tail_a_idx = 46;
            tail_b_idx = 47;
            crc_idx = 48;
        } else if (try_layout(kFbFrameLenV2, 50, 51, 52)) {
            full_ok = true;
            frame_len = kFbFrameLenV2;
            tail_a_idx = 50;
            tail_b_idx = 51;
            crc_idx = 52;
        }

        (void)tail_a_idx;
        (void)tail_b_idx;
        (void)crc_idx;

        if (full_ok)
        {
            // 解析完整帧：4臂/云台按 float(4B) 解码并直接写入 float 状态缓存。
            std::lock_guard<std::mutex> lock(state_mutex_);
            for (int i = 0; i < 4; ++i)
            {
                const size_t base = start + 2 + static_cast<size_t>(i) * 8;
                arm_pos_float_[i].x = decode_float_le(rx_buf_ + base);
                arm_pos_float_[i].y = decode_float_le(rx_buf_ + base + 4);
            }

            gimbal_float_.yaw = decode_float_le(rx_buf_ + start + 34);
            gimbal_float_.pitch = decode_float_le(rx_buf_ + start + 38);

            for (int i = 0; i < 2; ++i)
            {
                sensor_.valve[i] = (rx_buf_[start + 42] & (1 << i)) != 0;
            }
            for (int i = 2; i < 4; ++i)
            {
                sensor_.valve[i] = (rx_buf_[start + 43] & (1 << (i - 2))) != 0;
            }
            for (int i = 0; i < 2; ++i)
            {
                sensor_.microswitch[i] = (rx_buf_[start + 44] & (1 << i)) != 0;
            }
            for (int i = 2; i < 4; ++i)
            {
                sensor_.microswitch[i] = (rx_buf_[start + 45] & (1 << (i - 2))) != 0;
            }

            // 移除已处理帧。
            rx_len_ -= (start + frame_len);
            std::memmove(rx_buf_, rx_buf_ + start + frame_len, rx_len_);
            return true;
        }

        // 校验失败：丢弃当前帧头首字节，继续在本次调用内重同步。
        rx_len_ -= (start + 1);
        std::memmove(rx_buf_, rx_buf_ + start + 1, rx_len_);
    }
    return false;
}

// ---- 接收一次并尝试解析 ----
bool arm_internation::receive_once()
{
    if (fd_ < 0)
    {
        return reconnect_blocking();
    }

    // 极端情况下若缓存被占满，直接清空防止写越界。
    if (rx_len_ >= kBufSize)
    {
        rx_len_ = 0;
    }

    ssize_t n = ::read(fd_, rx_buf_ + rx_len_, kBufSize - rx_len_);
    if (n < 0)
    {
        const int err = errno;
        if (is_disconnect_errno(err))
        {
            std::cerr << "[arm_internation] serial disconnected on read, errno=" << err
                      << ", entering reconnect state" << std::endl;
            close();
            reconnect_blocking();
        }
        return false;
    }

    if (n > 0)
    {
        rx_len_ += static_cast<size_t>(n);
    }

    return parse_feedback_frame();
}

// ---- 状态读取 ----//
ArmEndPos arm_internation::get_arm_pos(int arm_id) const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (arm_id < 0 || arm_id >= 4)
    {
        return ArmEndPos{};
    }

    ArmEndPos out{};
    out.x = float_to_scaled_int16(arm_pos_float_[arm_id].x, pos_scale_);
    out.y = float_to_scaled_int16(arm_pos_float_[arm_id].y, pos_scale_);
    return out;
}
GimbalAngle arm_internation::get_gimbal() const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    GimbalAngle out{};
    out.yaw = float_to_scaled_int16(gimbal_float_.yaw, angle_scale_);
    out.pitch = float_to_scaled_int16(gimbal_float_.pitch, angle_scale_);
    return out;
}
SensorStatus arm_internation::get_sensor() const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    return sensor_;
}

ArmEndPosFloat arm_internation::get_arm_pos_float(int arm_id) const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    return (arm_id >= 0 && arm_id < 4) ? arm_pos_float_[arm_id] : ArmEndPosFloat{};
}

GimbalAngleFloat arm_internation::get_gimbal_float() const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    return gimbal_float_;
}

void arm_internation::set_decode_scale(float pos_scale, float angle_scale)
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    // 非正值没有物理意义，保持原配置。
    if (pos_scale > 0.0f)
    {
        pos_scale_ = pos_scale;
    }
    if (angle_scale > 0.0f)
    {
        angle_scale_ = angle_scale;
    }

    // 内部只保留 float 状态；比例仅影响 get_arm_pos/get_gimbal 的视图换算。
}

// ---- 写串口 ----//
bool arm_internation::write_bytes(const uint8_t *data, size_t len)
{
    if (fd_ < 0 && !reconnect_blocking())
    {
        return false;
    }

    // 串口 write 可能“部分写入”，这里循环直到全部写完或失败。
    size_t written = 0;
    while (written < len)
    {
        const ssize_t n = ::write(fd_, data + written, len - written);
        if (n <= 0)
        {
            const int err = errno;
            if (n < 0 && is_disconnect_errno(err))
            {
                std::cerr << "[arm_internation] serial disconnected on write, errno=" << err
                          << ", entering reconnect state" << std::endl;
                close();
                reconnect_blocking();
            }
            return false;
        }
        written += static_cast<size_t>(n);
    }
    return true;
}

////////////////////////// ---- 发送命令 ---- ////////////////////////
bool arm_internation::send_arm_cmd(int arm_id, float x, float y)
{
    // AA 02 浮点控制帧：
    // [0]AA [1]02 [2]arm_id [3..6]x(float LE) [7..10]y(float LE)
    // [11]CRC(覆盖[0]~[10]) [12]FF [13]EE
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[14] = {kHeadA, kCmdArm, static_cast<uint8_t>(arm_id),
                       0, 0, 0, 0,
                       0, 0, 0, 0,
                       0, kTailA, kTailB};
    encode_float_le(x, buf + 3);
    encode_float_le(y, buf + 7);
    buf[11] = calc_crc8(buf, 11);
    return write_bytes(buf, 14);
}

bool arm_internation::send_arm_cmd(int arm_id, int16_t x, int16_t y)
{
    // 兼容旧接口：旧代码仍可传 int16，内部统一走 float 帧。
    return send_arm_cmd(arm_id, static_cast<float>(x), static_cast<float>(y));
}

bool arm_internation::send_gimbal_cmd(int gimbal_id, int16_t yaw, int16_t pitch)
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[10] = {kHeadA, kCmdGim, (uint8_t)gimbal_id, (uint8_t)(yaw >> 8), (uint8_t)yaw, (uint8_t)(pitch >> 8), (uint8_t)pitch, 0, kTailA, kTailB};
    buf[7] = calc_crc8(buf, 7);
    return write_bytes(buf, 10);
}

bool arm_internation::send_valve_cmd(int valve_id, bool state)
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[8] = {kHeadA, kCmdValv, (uint8_t)valve_id, (uint8_t)state, 0, kTailA, kTailB};
    buf[4] = calc_crc8(buf, 4);
    return write_bytes(buf, 7);
}

std::string arm_internation::find_ttyacm_by_HWid(const std::string &hw_id)
{
    // 遍历 /dev 下 ttyACM*，逐个读取其 USB VID/PID 对比目标硬件 ID。
    std::string target_vendor;
    std::string target_product;
    if (!parse_hw_id(hw_id, target_vendor, target_product))
    {
        std::cerr << "[arm_internation] Invalid HW ID format: " << hw_id
                  << " (expected xxxx:xxxx)" << std::endl;
        return "";
    }

    const std::filesystem::path dev_path("/dev");
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator(dev_path, ec))
    {
        if (ec)
        {
            break;
        }
        const std::string tty_name = entry.path().filename().string();
        if (tty_name.rfind("ttyACM", 0) != 0)
        {
            continue;
        }

        std::string vendor;
        std::string product;
        if (!find_usb_vendor_product_for_tty(tty_name, vendor, product))
        {
            continue;
        }

        vendor = to_lower_copy(vendor);
        product = to_lower_copy(product);
        if (vendor == target_vendor && product == target_product)
        {
            return (dev_path / tty_name).string();
        }
    }
    return "";
}

std::string arm_internation::normalize_cmd_text(std::string text)
{
    // 统一命令格式，降低上层输入差异：
    // - 中文逗号/冒号/分号 -> 英文
    // - 删除空格
    // - 分号按逗号处理
    for (char &c : text)
    {
        if (c == ';')
        {
            c = ',';
        }
    }

    auto replace_all = [&](const std::string &from, const std::string &to)
    {
        size_t pos = 0;
        while ((pos = text.find(from, pos)) != std::string::npos)
        {
            text.replace(pos, from.size(), to);
            pos += to.size();
        }
    };

    replace_all("，", ",");
    replace_all("：", ":");
    replace_all("；", ",");
    replace_all(" ", "");
    return trim_copy(text);
}

bool arm_internation::parse_int_after_prefix(const std::string &token, const std::string &prefix, int &value)
{
    // 支持 X10 / X:10 / X=10 等写法。
    const std::string t = to_upper_copy(trim_copy(token));
    const std::string p = to_upper_copy(prefix);

    if (t.rfind(p, 0) != 0)
    {
        return false;
    }

    std::string numeric = t.substr(p.size());
    if (!numeric.empty() && (numeric[0] == ':' || numeric[0] == '='))
    {
        numeric = numeric.substr(1);
    }
    return parse_int_token(numeric, value);
}

bool arm_internation::parse_float_token(const std::string &token, float &value)
{
    return parse_float_token_impl(token, value);
}

bool arm_internation::parse_float_after_prefix(const std::string &token, const std::string &prefix, float &value)
{
    // 支持 X10.5 / X:10.5 / X=10.5 等写法。
    const std::string t = to_upper_copy(trim_copy(token));
    const std::string p = to_upper_copy(prefix);

    if (t.rfind(p, 0) != 0)
    {
        return false;
    }

    std::string numeric = t.substr(p.size());
    if (!numeric.empty() && (numeric[0] == ':' || numeric[0] == '='))
    {
        numeric = numeric.substr(1);
    }
    return parse_float_token_impl(numeric, value);
}

bool arm_internation::parse_arm_alias(const std::string &alias, int &arm_id) const
{
    // 机械臂 ID 映射：0=LF 1=RF 2=LB 3=RB
    // 兼容历史别名（例如 RL -> LB）。
    const std::string a = to_upper_copy(trim_copy(alias));
    if (a == "LF" || a == "FL")
    {
        arm_id = 0;
        return true;
    }
    if (a == "RF" || a == "FR")
    {
        arm_id = 1;
        return true;
    }
    if (a == "LB" || a == "BL")
    {
        arm_id = 2;
        return true;
    }
    if (a == "RB" || a == "BR")
    {
        arm_id = 3;
        return true;
    }
    return false;
}

bool arm_internation::reconnect_blocking()
{
    if (!auto_reconnect_enabled_.load() || reconnect_hw_id_.empty())
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(reconnect_mutex_);
    if (fd_ >= 0)
    {
        return true;
    }

    std::cerr << "[arm_internation] start auto reconnect by HWid=" << reconnect_hw_id_ << std::endl;
    return open_by_HWid(reconnect_hw_id_, reconnect_baud_rate_, reconnect_retry_ms_);
}

// 解析文本命令的主入口，支持机械臂、云台、电磁阀等多种命令格式，具有一定容错能力。
bool arm_internation::handle_text_command(const std::string &command_text)
{
    // 解析总入口：
    // 1) 先做文本规范化
    // 2) 再按首 token 判断命令类型（机械臂/云台/电磁阀）
    // 3) 最终调用 send_*_cmd
    const std::string normalized = normalize_cmd_text(command_text);
    if (normalized.empty())
    {
        return false;
    }

    const std::vector<std::string> tokens = split_by_comma(normalized);
    if (tokens.empty())
    {
        return false;
    }

    int arm_id = -1;
    if (parse_arm_alias(tokens[0], arm_id))
    {
        // 机械臂命令格式示例：
        // RL,X:10.5,Y:20.0
        // RF,10,20
        float x = 0.0f;
        float y = 0.0f;
        bool has_x = false;
        bool has_y = false;

        for (size_t i = 1; i < tokens.size(); ++i)
        {
            float value = 0.0f;
            if (parse_float_after_prefix(tokens[i], "X", value))
            {
                x = value;
                has_x = true;
                continue;
            }
            if (parse_float_after_prefix(tokens[i], "Y", value))
            {
                y = value;
                has_y = true;
                continue;
            }
            if (parse_float_token(tokens[i], value))
            {
                if (!has_x)
                {
                    x = value;
                    has_x = true;
                }
                else if (!has_y)
                {
                    y = value;
                    has_y = true;
                }
            }
        }

        if (!has_x || !has_y)
        {
            return false;
        }
        return send_arm_cmd(arm_id, x, y);
    }

    const std::string cmd = to_upper_copy(tokens[0]);
    if (cmd == "G")
    {
        // 云台命令：G,yaw,pitch（示例：G,0,0）
        if (tokens.size() < 3)
        {
            return false;
        }
        int yaw = 0;
        int pitch = 0;
        if (!parse_int_token(tokens[1], yaw) || !parse_int_token(tokens[2], pitch))
        {
            return false;
        }
        return send_gimbal_cmd(0, static_cast<int16_t>(yaw), static_cast<int16_t>(pitch));
    }

    if (cmd == "V")
    {
        // 电磁阀命令：
        // V,id           -> 翻转该阀当前“命令态”
        // V,id,state     -> 显式设置
        if (tokens.size() < 2)
        {
            return false;
        }

        int valve_id = -1;
        if (!parse_int_token(tokens[1], valve_id) || valve_id < 0 || valve_id > 3)
        {
            return false;
        }

        bool state = false;
        bool has_state = false;
        if (tokens.size() >= 3)
        {
            const std::string s = to_upper_copy(tokens[2]);
            if (s == "1" || s == "ON" || s == "OPEN" || s == "TRUE")
            {
                state = true;
                has_state = true;
            }
            else if (s == "0" || s == "OFF" || s == "CLOSE" || s == "FALSE")
            {
                state = false;
                has_state = true;
            }
        }

        if (!has_state)
        {
            // 若未给 state，则翻转缓存态，方便“一键切换”。
            std::lock_guard<std::mutex> lock(valve_cmd_mutex_);
            state = !valve_cmd_state_[valve_id];
        }

        if (!send_valve_cmd(valve_id, state))
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(valve_cmd_mutex_);
        // 成功发送后更新缓存态，保证下一次翻转有依据。
        valve_cmd_state_[valve_id] = state;
        return true;
    }

    return false;
}
