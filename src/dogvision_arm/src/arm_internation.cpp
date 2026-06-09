#include <dogvision_arm/arm_internation.hpp>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>    //终端控制接口。用于配置异步串行通信端口。
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
#include <cstdlib>
#include <limits>

#include <libusb-1.0/libusb.h> //检查硬件设备用

// ============================================================
//  协议帧格式说明（见头文件注释）
// ============================================================
//  AA 平面机械臂协议：
//  AA 01 反馈帧结构，CRC8 覆盖 [0] 到帧尾第二字节：
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
//
//  AA 下行帧：
//    AA 02 arm_id x(float LE) y(float LE) FF EE CRC8
//    AA 03 gimbal_id yaw(float LE) pitch(float LE) FF EE CRC8
//    AA 04 valve_id state FF EE CRC8
//    AA 05 answer 0 0 FF EE CRC8
//    AA 06 on_off speed(float LE) FF EE CRC8
//
//  BB 4DOF 双臂协议：
//  BB 01 反馈帧结构，固定 46 字节，CRC8 覆盖 [0]~[44]：
//    [0]  0xBB
//    [1]  0x01
//    [2-17]   左臂 x/y/z/pitch（4 个 float32 LE）
//    [18-33]  右臂 x/y/z/pitch（4 个 float32 LE）
//    [34-37]  valve0..3（每字节低 1 位有效）
//    [38-41]  microswitch0..3（当前 STM32 端预留为 0）
//    [42]     reserved
//    [43]     0xFF
//    [44]     0xEE
//    [45]     CRC8（覆盖 [0]~[44]）
//
//  BB 下行帧：
//    BB 02 arm_id x y z pitch（4 个 float32 LE）FF EE CRC8
//    BB 03 action_id FF EE CRC8
//    BB 04 valve_id state FF EE CRC8
//    BB 05 answer 0 0 FF EE CRC8
//    BB 06 on_off speed(float LE) FF EE CRC8
// ============================================================

// ============================================================
//  代码阅读导航（建议先看这段）
// ------------------------------------------------------------
//  A. 连接部分
//     open()            : 打开并配置指定串口（8N1 原始模式）
//     open_by_hwid()    : 自动扫描 ttyACM*/ttyUSB* 并按硬件 ID 连接（失败重试）
//     configure_auto_reconnect() / reconnect_once() / try_reconnect_once()
//                       : 自动重连有限状态机（FSM）
//
//  B. 接收部分
//     receive_once()    : 主接收循环入口（libusb 掉线预检 → read() → 帧解析）
//     parse_feedback_frame()
//                       : 协议分派 → parse_plane_feedback_frame / parse_4dof_feedback_frame
//                         帧同步策略：滑动窗口搜索帧头 → 双帧长 CRC 校验 → 数据不足保留/坏帧丢弃
//
//  C. 发送部分
//     send_arm_cmd() / send_gimbal_cmd() / send_valve_cmd()
//                       : 按协议打包帧 → write_bytes()（循环写，断线级联重连）
//
//  D. 上层适配
//     handle_text_command()
//                       : 规范化 → 分词 → 命令分派（自动按协议适配 AA/BB）
//
//  E. 异常处理总览
//     - 串口断开：read/write EIO/ENODEV/ENXIO → close() + clear_report_state() + reconnect
//     - 帧同步失败：跳过坏字节重新搜索，永不阻塞
//     - libusb 不可用：保守策略返回 true（不误触发掉线）
//     - 参数越界：返回 false / 零值，不崩溃
// ============================================================

arm_internation::arm_internation() {}
arm_internation::~arm_internation() { close(); }

namespace
{
    /**
     * @name 反馈帧兼容长度常量
     * @details AA 协议存在两个兼容版本：
     *          - V1 (49B): 标准版，2 头 + 40 浮点净荷 + 4 传感器 + 2 尾 + 1 CRC
     *          - V2 (53B): 扩展版，V1 基础上在微动状态与帧尾之间插入 4 字节保留位
     *          parse_plane_feedback_frame() 对候选帧头同时尝试两种帧长，
     *          以 tail + CRC 双校验通过的为准，兼容不同固件版本。
     *          BB 协议固定 46 字节，无版本兼容问题。
     * @{
     */
    static constexpr size_t kFbFrameLenV1 = 49;
    static constexpr size_t kFbFrameLenV2 = 53;
    static constexpr size_t k4DofFbFrameLen = 46;
    /** @} */

    /**
     * @brief 将常见整型波特率转换为 termios speed_t 常量。
     * @param baud_rate 整型波特率（9600 ~ 921600）
     * @param[out] speed 输出的 termios 速度常量
     * @retval true 转换成功
     * @retval false 不支持的波特率值
     * @note 统一映射避免调用方直接传入平台相关宏值（如 B115200），
     *       提高代码可移植性。
     */
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

    /**
     * @brief 字符串转小写（无符号安全版本）。
     * @details 使用 unsigned char 转换避免高位字符（如 UTF-8 多字节）导致未定义行为。
     *          仅用于 ASCII 协议标识符和命令 token，不处理完整 Unicode。
     */
    /**
     * @brief 字符串转小写（无符号安全版本）。
     * @details 使用 unsigned char 转换避免高位字符（如 UTF-8 多字节）导致未定义行为。
     *          仅用于 ASCII 协议标识符和命令 token，不处理完整 Unicode。
     */
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

    /**
     * @brief 查找 sysfs 中 tty 设备对应的 USB VID/PID。
     * @param tty_name tty 设备名（如 "ttyACM0"）
     * @param[out] vendor 输出的 idVendor 字符串（小写十六进制）
     * @param[out] product 输出的 idProduct 字符串（小写十六进制）
     * @retval true 成功读取 VID/PID
     * @retval false sysfs 路径不存在或无法读取
     * @details 从 /sys/class/tty/<tty>/device 出发，逐级向父目录回溯，
     *          最多 10 级，直到找到 idVendor/idProduct 文件。
     *          设计目的：避免硬编码设备路径，适应不同内核版本 sysfs 布局。
     */
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

    /**
     * @brief 解析 "VID:PID" 格式的硬件 ID 字符串。
     * @param hw_id 输入字符串（如 "0483:5740"）
     * @param[out] vendor 输出的 idVendor（小写）
     * @param[out] product 输出的 idProduct（小写）
     * @retval true 格式正确且非空
     * @retval false 无冒号分隔符或字段为空
     */
    // 在 /dev 目录筛选 ttyACM*/ttyUSB*，并按 idVendor:idProduct 匹配目标设备。
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

    /**
     * @brief 严格解析 4 位十六进制字符串为 uint16_t。
     * @param text 十六进制字符串（如 "0483"，不区分大小写）
     * @param[out] value 输出的无符号 16 位整数值
     * @retval true 成功解析
     * @retval false 空字符串/超长/非十六进制字符/溢出
     */
    bool parse_hex_u16(const std::string &text, uint16_t &value)
    {
        if (text.empty() || text.size() > 4)
        {
            return false;
        }

        char *end_ptr = nullptr;
        errno = 0;
        const unsigned long parsed = std::strtoul(text.c_str(), &end_ptr, 16);
        if (end_ptr == text.c_str() || *end_ptr != '\0' || errno != 0 || parsed > 0xFFFFUL)
        {
            return false;
        }

        value = static_cast<uint16_t>(parsed);
        return true;
    }

    bool parse_hw_id_to_vid_pid(const std::string &hw_id, uint16_t &vendor_id, uint16_t &product_id)
    {
        std::string vendor;
        std::string product;
        if (!parse_hw_id(hw_id, vendor, product))
        {
            return false;
        }
        return parse_hex_u16(vendor, vendor_id) && parse_hex_u16(product, product_id);
    }

    /**
     * @brief 通过 libusb 检查指定 VID/PID 的 USB 设备是否在线。
     * @param vendor_id USB 供应商 ID
     * @param product_id USB 产品 ID
     * @retval true 设备在线或 libusb 初始化失败（保守策略：不误触发掉线）
     * @retval false 设备不在 USB 总线上
     * @details libusb 直接枚举 USB 总线设备列表，不依赖串口驱动状态。
     *          当 libusb_init() 失败时（如权限不足），保守返回 true，
     *          避免因库不可用而错误触发断线重连。
     *          调用方应在每次检测后调用 libusb_exit() 清理上下文。
     */
    bool has_usb_device_via_libusb(uint16_t vendor_id, uint16_t product_id)
    {
        libusb_context *ctx = nullptr;
        if (libusb_init(&ctx) != 0)
        {
            // 无法初始化 libusb 时不判定为断线，避免误触发重连。
            return true;
        }

        libusb_device **dev_list = nullptr;
        const ssize_t count = libusb_get_device_list(ctx, &dev_list);
        if (count < 0)
        {
            libusb_exit(ctx);
            return true;
        }

        bool found = false;
        for (ssize_t i = 0; i < count; ++i)
        {
            libusb_device_descriptor desc{};
            if (libusb_get_device_descriptor(dev_list[i], &desc) == 0)
            {
                if (desc.idVendor == vendor_id && desc.idProduct == product_id)
                {
                    found = true;
                    break;
                }
            }
        }

        libusb_free_device_list(dev_list, 1);
        libusb_exit(ctx);
        return found;
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
        std::cerr << "[arm_internation] Failed to open " << port
                  << ": " << std::strerror(errno) << std::endl;
        return false;
    }
    struct termios tty{};
    if (tcgetattr(fd_, &tty) != 0)
    {
        std::cerr << "[arm_internation] tcgetattr failed on " << port
                  << ": " << std::strerror(errno) << std::endl;
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
        std::cerr << "[arm_internation] tcsetattr failed on " << port
                  << ": " << std::strerror(errno) << std::endl;
        close();
        return false;
    }
    return true;
}

bool arm_internation::open_by_HWid(const std::string &hw_id, int baud_rate, int retry_ms)
{
    configure_auto_reconnect(hw_id, baud_rate, retry_ms);

    // 常驻重试策略：该函数只有在成功连接后才返回。
    // 适用于“设备可能晚插入”或“启动时串口尚未就绪”的场景。
    const int wait_ms = retry_ms > 0 ? retry_ms : 1000;
    while (true)
    {
        if (open_matching_tty_once(hw_id, baud_rate, "open_by_HWid"))
        {
            return true;
        }
        std::cerr << "[arm_internation] open_by_HWid: retry HW ID " << hw_id
                  << " in " << wait_ms << " ms" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    }
}

void arm_internation::configure_auto_reconnect(const std::string &hw_id, int baud_rate, int retry_ms)
{
    std::lock_guard<std::mutex> lock(reconnect_mutex_);
    reconnect_hw_id_ = hw_id;
    reconnect_baud_rate_ = baud_rate;
    reconnect_retry_ms_ = retry_ms > 0 ? retry_ms : 1000;
    auto_reconnect_enabled_.store(!reconnect_hw_id_.empty());
    last_reconnect_attempt_tp_ = {};
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

bool arm_internation::set_protocol_from_string(const std::string &protocol_name)
{
    const std::string p = to_lower_copy(trim_copy(protocol_name));
    if (p == "aa" || p == "plane" || p == "plane_aa" || p == "平面")
    {
        protocol_ = ArmProtocol::PlaneAA;
        clear_report_state();
        return true;
    }
    if (p == "bb" || p == "4dof" || p == "dof4" || p == "dof4_bb" || p == "双臂")
    {
        protocol_ = ArmProtocol::Dof4BB;
        clear_report_state();
        return true;
    }
    return false;
}

ArmProtocol arm_internation::protocol() const
{
    return protocol_;
}

const char *arm_internation::protocol_name() const
{
    return protocol_ == ArmProtocol::Dof4BB ? "4dof" : "aa";
}

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
/**
 * @section parse_plane_feedback_frame 帧同步状态机
 *
 * 核心挑战：串口是流式设备，read() 不保证帧边界对齐。
 * 策略："滑动窗口 + 双帧长试探 + 坏帧逐字节丢弃"
 *
 * 状态转换图：
 *  ┌──────────┐   搜索到 0xAA 0x01    ┌──────────────┐
 *  │ 搜索帧头  │─────────────────────▶│ 候选帧校验    │
 *  │(滑动窗口) │◀────────────────────│              │
 *  └──────────┘  坏帧：丢弃1字节      └──────┬───────┘
 *        │          继续搜索                  │
 *        │ 数据不足                           │ tail + CRC 通过
 *        ▼                                    ▼
 *  ┌──────────┐                        ┌──────────────┐
 *  │ 等待拼接  │                        │ 解码 & 更新   │
 *  │(保留帧头) │                        │ 状态缓存      │
 *  └──────────┘                        └──────────────┘
 *
 * 关键设计决策：
 * - "数据不足"与"校验失败"的区分：
 *   若当前候选帧头 + 最长帧长（V2/53B）超出 rx_len_，判定为数据不足，
 *   保留帧头在缓存前端，等待下次 read() 拼接。不丢弃任何字节。
 *   若两种帧长都有足够字节但 tail/CRC 均不通过，则为真正坏帧头，
 *   丢弃当前帧头的一个字节，从下一字节继续搜索。
 *
 * - 双帧长兼容：
 *   优先尝试 V1(49B)，再尝试 V2(53B)。
 *   两种帧长各自独立校验 tail(0xFF 0xEE) + CRC8。
 *   这确保下位机固件升级（增加保留位）后无需修改代码。
 */
bool arm_internation::parse_feedback_frame()
{
    if (protocol_ == ArmProtocol::Dof4BB)
    {
        return parse_4dof_feedback_frame();
    }
    return parse_plane_feedback_frame();
}

bool arm_internation::parse_plane_feedback_frame()
{
    // 注意：接收缓存是"流式字节"，不保证 read 一次就是一帧。
    // 这里在同一次调用内持续重同步，直到找到有效帧或缓存不足最短帧。
    while (rx_len_ >= kFbFrameLenV1)
    {
        // 查找帧头
        size_t start = 0;
        while (start + kFbFrameLenV1 <= rx_len_ &&
               (rx_buf_[start] != 0xAA || rx_buf_[start + 1] != kCmdFb))
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
            if (rx_buf_[start + tail_a] != 0xFF || rx_buf_[start + tail_b] != 0xEE) {
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

        // full_ok == false：区分"字节不足"与"真正校验失败"两种情况。
        // 若连最长帧（V2/53B）都还凑不齐，说明只是读到了半帧——不能丢弃帧头，
        // 应把缓存紧凑到帧头处，等待下次 receive_once() 读入更多字节再重试。
        if (start + kFbFrameLenV2 > rx_len_)
        {
            // 数据不足，将帧头移到缓存起点后退出
            if (start > 0)
            {
                rx_len_ -= start;
                std::memmove(rx_buf_, rx_buf_ + start, rx_len_);
            }
            return false;
        }

        // 两种帧长都有足够字节却仍然校验失败：真正的坏帧头，丢弃一字节继续重同步。
        rx_len_ -= (start + 1);
        std::memmove(rx_buf_, rx_buf_ + start + 1, rx_len_);
    }
    return false;
}

bool arm_internation::parse_4dof_feedback_frame()
{
    // BB 01 固定 46 字节：
    // [0]BB [1]01 [2..33] 左/右臂 8 个 float [34..37]阀门
    // [38..41]微动 [42]预留 [43]FF [44]EE [45]CRC8。
    while (rx_len_ >= k4DofFbFrameLen)
    {
        size_t start = 0;
        while (start + k4DofFbFrameLen <= rx_len_ &&
               (rx_buf_[start] != kHeadB || rx_buf_[start + 1] != kCmdFb))
        {
            ++start;
        }

        if (start + k4DofFbFrameLen > rx_len_)
        {
            if (start > 0)
            {
                rx_len_ -= start;
                std::memmove(rx_buf_, rx_buf_ + start, rx_len_);
            }
            return false;
        }

        const size_t tail_a_idx = start + 43;
        const size_t tail_b_idx = start + 44;
        const size_t crc_idx = start + 45;
        const bool tail_ok = rx_buf_[tail_a_idx] == kTailA && rx_buf_[tail_b_idx] == kTailB;
        const bool crc_ok = tail_ok && calc_crc8(rx_buf_ + start, 45) == rx_buf_[crc_idx];

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

            // 将 4DOF 左/右臂 x/y 投影到旧 LF/RF 字段，保证旧状态消费者仍能读到主位姿。
            arm_pos_float_[0].x = dof4_pose_float_[0].x;
            arm_pos_float_[0].y = dof4_pose_float_[0].y;
            arm_pos_float_[1].x = dof4_pose_float_[1].x;
            arm_pos_float_[1].y = dof4_pose_float_[1].y;
            arm_pos_float_[2] = ArmEndPosFloat{};
            arm_pos_float_[3] = ArmEndPosFloat{};
            gimbal_float_ = GimbalAngleFloat{};

            for (int i = 0; i < 4; ++i)
            {
                sensor_.valve[i] = (rx_buf_[start + 34 + i] & 0x01u) != 0;
                sensor_.microswitch[i] = (rx_buf_[start + 38 + i] & 0x01u) != 0;
            }

            rx_len_ -= (start + k4DofFbFrameLen);
            std::memmove(rx_buf_, rx_buf_ + start + k4DofFbFrameLen, rx_len_);
            return true;
        }

        // 候选帧头存在但尾或 CRC 不对，丢弃当前帧头一个字节继续重同步。
        rx_len_ -= (start + 1);
        std::memmove(rx_buf_, rx_buf_ + start + 1, rx_len_);
    }
    return false;
}

// ---- 接收一次并尝试解析 ----
bool arm_internation::receive_once()
{
    /**
     * @section receive_once 状态机（三层级联检测）
     *
     * 本函数是"接收-解析-容错"一体化入口，按优先级串联三层检测：
     *
     * [1] libusb 掉线预检（仅 auto_reconnect 模式）
     *     条件：距上次检查 >= usb_check_interval_ms_（默认 500ms）
     *     动作：调用 is_bound_hwid_online_libusb() 查询 USB 总线
     *     掉线则：close() → clear_report_state() → reconnect_once()
     *
     * [2] 串口 read()
     *     正常：追加字节到 rx_buf_，累计 rx_len_
     *     缓冲区满（>= kBufSize=512）：直接清空，防止写越界
     *     read() 返回错误：判断 errno 是否为断开类（EIO/ENODEV/ENXIO）
     *       是 → 级联触发重连（同 [1] 的掉线路径）
     *       否 → 静默返回 false
     *
     * [3] parse_feedback_frame()
     *     按当前协议（AA/BB）搜索完整帧、校验 CRC
     *     成功 → 更新状态缓存，返回 true
     *     失败 → 返回 false（等待下次 receive_once 累积更多字节）
     *
     * 输出：true=本次调用解析出一帧；false=无新帧或错误
     * 异常：所有异常均内部消化，不向上抛出
     */
    if (fd_ < 0)
    {
        clear_report_state();
        return reconnect_once();
    }

    // 对绑定 HWID 的场景，使用 libusb 周期性确认 USB 设备是否仍在线。
    // 一旦确认掉线，立即清空上报缓存并进入自动重连。
    if (!reconnect_hw_id_.empty())
    {
        const auto now = std::chrono::steady_clock::now();
        if (last_usb_check_tp_.time_since_epoch().count() == 0 ||
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_usb_check_tp_).count() >= usb_check_interval_ms_)
        {
            last_usb_check_tp_ = now;
            if (!is_bound_hwid_online_libusb())
            {
                std::cerr << "[arm_internation] USB device " << reconnect_hw_id_
                          << " not found by libusb, entering reconnect state" << std::endl;
                close();
                clear_report_state();
                return reconnect_once();
            }
        }
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
            clear_report_state();
            reconnect_once();
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

Arm4DofPoseFloat arm_internation::get_4dof_pose_float(int arm_id) const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    return (arm_id >= 0 && arm_id < 2) ? dof4_pose_float_[arm_id] : Arm4DofPoseFloat{};
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
    if (fd_ < 0 && !reconnect_once())
    {
        clear_report_state();
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
                clear_report_state();
                reconnect_once();
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
    if (protocol_ == ArmProtocol::Dof4BB)
    {
        // 4DOF 模式必须使用显式 4POSE 命令，避免旧 LF/RF 文本被误当成双臂控制。
        return false;
    }

    // AA 02 浮点控制帧：
    // [0]AA [1]02 [2]arm_id [3..6]x(float LE) [7..10]y(float LE)
    //  [11]FF [12]EE [13]CRC(覆盖[0]~[12])
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[14] = {0xAA, kCmdArm, static_cast<uint8_t>(arm_id),
                       0, 0, 0, 0,
                       0, 0, 0, 0,
                       0xFF, 0xEE,
                       0};
    encode_float_le(x, buf + 3);
    encode_float_le(y, buf + 7);
    buf[13] = calc_crc8(buf, 13);
    return write_bytes(buf, 14);
}

bool arm_internation::send_gimbal_cmd(int gimbal_id, float yaw, float pitch)
{
    if (protocol_ == ArmProtocol::Dof4BB)
    {
        // BB 4DOF 协议没有云台命令，保守拒绝。
        return false;
    }

    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[14] =  {0xAA, kCmdGim, (uint8_t)gimbal_id, 
                        0, 0, 0, 0, 
                        0, 0, 0, 0, 
                        0xFF, 0xEE, 0};
    encode_float_le(yaw, buf + 3);
    encode_float_le(pitch, buf + 7);
    buf[13] = calc_crc8(buf, 13);
    return write_bytes(buf, 14);
}

bool arm_internation::send_valve_cmd(int valve_id, bool state)
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    const uint8_t head = protocol_ == ArmProtocol::Dof4BB ? kHeadB : kHeadA;
    uint8_t buf[7] = {head, kCmdValv, (uint8_t)valve_id, (uint8_t)state, 0xFF, 0xEE, 0};
    buf[6] = calc_crc8(buf, 6);
    return write_bytes(buf, 7);
}

//发送任务赛的答案0-3
bool arm_internation::send_answer_cmd(uint8_t answer)
{
    // AA: 任务赛答案；BB: CMD4_ANSWER_CONTROL 当前在 STM32 端预留，仍按协议发 3 字节 DATA。
    std::lock_guard<std::mutex> lock(send_mutex_);
    const uint8_t head = protocol_ == ArmProtocol::Dof4BB ? kHeadB : kHeadA;
    uint8_t buf[8] = {head, kCmdAns, answer,0, 0, 0xFF, 0xEE, 0};
    buf[7] = calc_crc8(buf, 7);
    return write_bytes(buf, 8);
}

//发送气泵控制命令
//帧格式：AA 06 [on:0/1] [speed_H] [speed_L] FF EE CRC
//on=1：开泵并设置速度；on=0：关泵
bool arm_internation::send_pump_cmd(bool on, int speed)
{
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t on_off = on ? 1 : 0;
    const uint8_t head = protocol_ == ArmProtocol::Dof4BB ? kHeadB : kHeadA;
    uint8_t buf[10] = {head, kCmdPump, on_off, 0, 0,0,0, 0xFF, 0xEE, 0};
    encode_float_le(static_cast<float>(speed), buf + 3);
    buf[9] = calc_crc8(buf, 9);
    return write_bytes(buf, 10);
}

bool arm_internation::send_4dof_pose_cmd(int arm_id, float x, float y, float z, float pitch)
{
    if (protocol_ != ArmProtocol::Dof4BB || arm_id < 0 || arm_id > 1)
    {
        return false;
    }

    // BB 02 位姿帧：
    // [0]BB [1]02 [2]arm_id [3..6]x [7..10]y [11..14]z [15..18]pitch [19]FF [20]EE [21]CRC
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[22] = {kHeadB, kCmdArm, static_cast<uint8_t>(arm_id),
                       0, 0, 0, 0,
                       0, 0, 0, 0,
                       0, 0, 0, 0,
                       0, 0, 0, 0,
                       kTailA, kTailB, 0};
    encode_float_le(x, buf + 3);
    encode_float_le(y, buf + 7);
    encode_float_le(z, buf + 11);
    encode_float_le(pitch, buf + 15);
    buf[21] = calc_crc8(buf, 21);
    return write_bytes(buf, sizeof(buf));
}

bool arm_internation::send_4dof_action_cmd(uint8_t action_id)
{
    if (protocol_ != ArmProtocol::Dof4BB)
    {
        return false;
    }

    // BB 03 动作帧：[0]BB [1]03 [2]action_id [3]FF [4]EE [5]CRC。
    std::lock_guard<std::mutex> lock(send_mutex_);
    uint8_t buf[6] = {kHeadB, kCmdGim, action_id, kTailA, kTailB, 0};
    buf[5] = calc_crc8(buf, 5);
    return write_bytes(buf, sizeof(buf));
}


std::vector<std::string> arm_internation::find_ttys_by_HWid(const std::string &hw_id)
{
    // 遍历 /dev 下 ttyACM*/ttyUSB*，逐个读取其 USB VID/PID 对比目标硬件 ID。
    std::vector<std::string> matches;
    std::string target_vendor;
    std::string target_product;
    if (!parse_hw_id(hw_id, target_vendor, target_product))
    {
        std::cerr << "[arm_internation] Invalid HW ID format: " << hw_id
                  << " (expected xxxx:xxxx)" << std::endl;
        return matches;
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
        if (tty_name.rfind("ttyACM", 0) != 0 && tty_name.rfind("ttyUSB", 0) != 0)
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
            matches.push_back((dev_path / tty_name).string());
        }
    }
    std::sort(matches.begin(), matches.end());
    return matches;
}

bool arm_internation::open_matching_tty_once(const std::string &hw_id, int baud_rate, const char *log_context)
{
    const std::vector<std::string> candidates = find_ttys_by_HWid(hw_id);
    const char *context = log_context != nullptr ? log_context : "open";
    if (candidates.empty())
    {
        std::cerr << "[arm_internation] " << context << ": no ttyACM/ttyUSB device matched HW ID "
                  << hw_id << std::endl;
        return false;
    }

    bool had_open_failure = false;
    for (const std::string &dev : candidates)
    {
        std::cerr << "[arm_internation] " << context << ": trying " << dev
                  << " for HW ID " << hw_id << std::endl;
        if (open(dev, baud_rate))
        {
            last_usb_check_tp_ = std::chrono::steady_clock::now();
            std::cerr << "[arm_internation] " << context << ": connected to " << dev
                      << " for HW ID " << hw_id << std::endl;
            return true;
        }
        had_open_failure = true;
    }

    if (had_open_failure)
    {
        std::cerr << "[arm_internation] " << context << ": all matched ttyACM/ttyUSB devices failed to open for HW ID "
                  << hw_id << std::endl;
    }
    return false;
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

// 从 token 中提取 float，要求 token 以特定前缀开头（如 "X" 或 "Y"），并支持多种分隔符。
bool arm_internation::parse_float_token(const std::string &token, float &value)
{
    return parse_float_token_impl(token, value);
}

// 从 token 中提取 float，要求 token 以特定前缀开头（如 "X" 或 "Y"），并支持多种分隔符。
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


// 解析机械臂别名，支持 LF/RF/LB/RB 以及历史兼容的 FL/FR/BL/BR，映射到 0-3 的 arm_id。
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

bool arm_internation::parse_4dof_arm_alias(const std::string &alias, int &arm_id)
{
    const std::string a = to_upper_copy(trim_copy(alias));
    if (a == "0" || a == "L" || a == "LEFT" || a == "左" || a == "左臂")
    {
        arm_id = 0;
        return true;
    }
    if (a == "1" || a == "R" || a == "RIGHT" || a == "右" || a == "右臂")
    {
        arm_id = 1;
        return true;
    }
    return false;
}

bool arm_internation::reconnect_once()
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

    const auto now = std::chrono::steady_clock::now();
    if (last_reconnect_attempt_tp_.time_since_epoch().count() != 0)
    {
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_reconnect_attempt_tp_).count();
        if (elapsed_ms < reconnect_retry_ms_)
        {
            return false;
        }
    }
    last_reconnect_attempt_tp_ = now;

    return open_matching_tty_once(reconnect_hw_id_, reconnect_baud_rate_, "auto_reconnect");
}

bool arm_internation::is_bound_hwid_online_libusb() const
{
    if (reconnect_hw_id_.empty())
    {
        return true;
    }

    uint16_t vendor_id = 0;
    uint16_t product_id = 0;
    if (!parse_hw_id_to_vid_pid(reconnect_hw_id_, vendor_id, product_id))
    {
        // HWID 配置异常时不触发掉线判定，交由串口读写错误路径处理。
        return true;
    }

    return has_usb_device_via_libusb(vendor_id, product_id);
}

void arm_internation::clear_report_state()
{
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        for (auto &arm_pos : arm_pos_float_)
        {
            arm_pos.x = 0.0f;
            arm_pos.y = 0.0f;
        }
        for (auto &pose : dof4_pose_float_)
        {
            pose = Arm4DofPoseFloat{};
        }

        gimbal_float_.yaw = 0.0f;
        gimbal_float_.pitch = 0.0f;
        sensor_ = SensorStatus{};
    }

    rx_len_ = 0;
}

// 供外部调用的非阻塞重连尝试，适用于 ROS 主循环周期调用。
bool arm_internation::try_reconnect_once()
{
    if (reconnect_hw_id_.empty())
    {
        return false;
    }

    // 加锁：避免与 receive_once() / write_bytes() 并发修改 fd_
    std::lock_guard<std::mutex> lock(reconnect_mutex_);
    // 已经连接时无需操作
    if (fd_ >= 0)
    {
        return true;
    }

    const auto now = std::chrono::steady_clock::now();
    if (last_reconnect_attempt_tp_.time_since_epoch().count() != 0)
    {
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_reconnect_attempt_tp_).count();
        if (elapsed_ms < reconnect_retry_ms_)
        {
            return false;
        }
    }
    last_reconnect_attempt_tp_ = now;

    return open_matching_tty_once(reconnect_hw_id_, reconnect_baud_rate_, "try_reconnect_once");
}

// 解析文本命令的主入口，支持机械臂、云台、电磁阀等多种命令格式，具有一定容错能力。
/**
 * @section handle_text_command 命令分派状态机
 *
 * 设计目的：将人类可读的文本命令（支持中英文别名、多种分隔符）转换为协议帧发送。
 * 这是上层 ROS 话题 /arm_internation/cmd 的唯一入口。
 *
 * 分派决策树（按首 token 优先级）：
 *
 *   normalized cmd
 *        │
 *   ┌────┼────┬────┬────┬────┬────┐
 *   ▼    ▼    ▼    ▼    ▼    ▼    ▼
 * 4POSE 4ACT LF/RF G    V    P    A
 *  (BB)  (BB) /LB/RB (AA) (both)(both)(both)
 *                  (AA)
 *
 * 输入容错设计：
 * - 中英文标点（，：；→ ,:;）自动转换
 * - 空格去除
 * - 前缀分隔符兼容：X:10 / X=10 / X10 均等效
 * - 电磁阀 V,id 无状态参数时翻转缓存态（一键切换）
 *
 * 异常处理：
 * - 空命令：返回 false
 * - token 数量不足：返回 false
 * - 数值解析失败：返回 false（strtol/strtof 严格模式）
 * - 协议不匹配：BB 协议下 AA 命令（arm/gimbal）返回 false
 * - 越界参数：arm_id>3、valve_id>3 返回 false
 */
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

    const std::string cmd = to_upper_copy(tokens[0]);

    if (cmd == "4POSE" || cmd == "4P" || cmd == "DOF4POSE")
    {
        // 4DOF 位姿命令：
        //   4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4
        //   4POSE,R,0.1,0.2,0.3,0.4
        if (tokens.size() < 6)
        {
            return false;
        }

        int dof4_arm_id = -1;
        if (!parse_4dof_arm_alias(tokens[1], dof4_arm_id))
        {
            return false;
        }

        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float pitch = 0.0f;
        bool has_x = false;
        bool has_y = false;
        bool has_z = false;
        bool has_pitch = false;

        for (size_t i = 2; i < tokens.size(); ++i)
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
            if (parse_float_after_prefix(tokens[i], "Z", value))
            {
                z = value;
                has_z = true;
                continue;
            }
            if (parse_float_after_prefix(tokens[i], "PITCH", value) ||
                parse_float_after_prefix(tokens[i], "P", value))
            {
                pitch = value;
                has_pitch = true;
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
                else if (!has_z)
                {
                    z = value;
                    has_z = true;
                }
                else if (!has_pitch)
                {
                    pitch = value;
                    has_pitch = true;
                }
            }
        }

        if (!has_x || !has_y || !has_z || !has_pitch)
        {
            return false;
        }
        return send_4dof_pose_cmd(dof4_arm_id, x, y, z, pitch);
    }

    if (cmd == "4ACT" || cmd == "4ACTION" || cmd == "DOF4ACT")
    {
        // 4ACT,0 中止；4ACT,1..N 触发下位机 action_state_4dof_e 对应动作。
        if (tokens.size() < 2)
        {
            return false;
        }

        int action_id = -1;
        if (!parse_int_token(tokens[1], action_id) || action_id < 0 || action_id > 255)
        {
            return false;
        }
        return send_4dof_action_cmd(static_cast<uint8_t>(action_id));
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

    if (cmd == "A")
    {
        // A,0..255：AA 模式作为任务赛答案，BB 模式作为 CMD4_ANSWER_CONTROL 预留字段发送。
        if (tokens.size() < 2)
        {
            return false;
        }

        int answer = -1;
        if (!parse_int_token(tokens[1], answer) || answer < 0 || answer > 255)
        {
            return false;
        }
        return send_answer_cmd(static_cast<uint8_t>(answer));
    }

    if (cmd == "P")
    {
        // 气泵命令：
        // P,ON,2500    -> 打开气泵，设置速度 2500
        // P,OFF        -> 关闭气泵
        if (tokens.size() < 2)
        {
            return false;
        }

        const std::string action = to_upper_copy(tokens[1]);
        if (action == "ON" || action == "1" || action == "OPEN" || action == "TRUE")
        {
            int speed = 0;
            if (tokens.size() >= 3)
            {
                if (!parse_int_token(tokens[2], speed) || speed < 0)
                {
                    return false;
                }
            }
            return send_pump_cmd(true, speed);
        }
        else if (action == "OFF" || action == "0" || action == "CLOSE" || action == "FALSE")
        {
            return send_pump_cmd(false, 0);
        }

        return false;
    }

    return false;
}
