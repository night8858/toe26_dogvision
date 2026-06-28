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

// ╔═══════════════════════════════════════════════════════════════╗
// ║                    协议帧格式说明                              ║
// ╚═══════════════════════════════════════════════════════════════╝
//  见头文件注释，以下为 C++ 源码内嵌入式文档便于快速查阅。
// ============================================================
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
//  BB 下行帧，固定 22 字节，CRC8 覆盖 [0]~[20]：
//    BB 02 arm_id x(4B LE) y(4B LE) z(4B LE) pitch(4B LE) FF EE CRC8
//      [0]    0xBB
//      [1]    0x02
//      [2]    arm_id（0=左臂, 1=右臂）
//      [3-6]  x（float32 LE）
//      [7-10] y（float32 LE）
//      [11-14] z（float32 LE）
//      [15-18] pitch（float32 LE）
//      [19]   0xFF
//      [20]   0xEE
//      [21]   CRC8（覆盖 [0]~[20]）
//
//  BB 03 动作帧，固定 6 字节，CRC8 覆盖 [0]~[4]：
//    BB 03 action_id FF EE CRC8
//      [0]    0xBB
//      [1]    0x03
//      [2]    action_id（对应下位机 action_state_4dof_e 枚举）
//      [3]    0xFF
//      [4]    0xEE
//      [5]    CRC8（覆盖 [0]~[4]）
//
//  BB 04 电磁阀帧，固定 7 字节，CRC8 覆盖 [0]~[5]：
//    BB 04 valve_id state FF EE CRC8
//      [0]    0xBB
//      [1]    0x04
//      [2]    valve_id（0..3）
//      [3]    state（0=关, 1=开）
//      [4]    0xFF
//      [5]    0xEE
//      [6]    CRC8（覆盖 [0]~[5]）
//
//  BB 05 答案帧，固定 8 字节，CRC8 覆盖 [0]~[6]：
//    BB 05 answer 0 0 FF EE CRC8
//      [0]    0xBB
//      [1]    0x05
//      [2]    answer（0..255，任务赛答案编号，STM32 端预留）
//      [3]    0x00（预留，填充 0）
//      [4]    0x00（预留，填充 0）
//      [5]    0xFF
//      [6]    0xEE
//      [7]    CRC8（覆盖 [0]~[6]）
//
//  BB 06 气泵帧，固定 10 字节，CRC8 覆盖 [0]~[8]：
//    BB 06 on_off speed(4B float LE) FF EE CRC8
//      [0]    0xBB
//      [1]    0x06
//      [2]    on_off（0=关泵, 1=开泵）
//      [3-6]  speed（float32 LE，气泵转速）
//      [7]    0xFF
//      [8]    0xEE
//      [9]    CRC8（覆盖 [0]~[8]）
//
//  BB 11/12/13 单臂动态目标动作帧，固定 18 字节，CRC8 覆盖 [0]~[16]：
//    BB cmd arm_id x(4B float LE) y(4B float LE) z(4B float LE) FF EE CRC8
//      cmd=0x11 取块；cmd=0x12 放块
//      arm_id=0 左臂，arm_id=1 右臂；xyz 单位 m，原样发送，不做 mm/m 换算。
//
//  BB 14/15 单臂背部固定动作帧，固定 6 字节，CRC8 覆盖 [0]~[4]：
//    BB cmd arm_id FF EE CRC8
//      cmd=0x14 放块到背部；cmd=0x15 从背部取块。
//
//  BB 21 双臂动态取块帧，固定 29 字节，CRC8 覆盖 [0]~[27]：
//    BB 21 Lx Ly Lz Rx Ry Rz FF EE CRC8
//      六个坐标均为 float32 小端、单位 m，左臂目标在前，右臂目标在后。
//
//  BB 22 双臂放块到背部固定动作帧，固定 5 字节，CRC8 覆盖 [0]~[3]：
//    BB 22 FF EE CRC8
//      无 DATA 段，实际路径由下位机动作模板决定。
//
//  BB 99 带初始偏移启动帧，固定 17 字节，CRC8 覆盖 [0]~[15]：
//    BB 99 offsetX(4B float LE) offsetY(4B float LE) offsetZ(4B float LE) FF EE CRC8
//      [0]     0xBB
//      [1]     0x99
//      [2-5]   offsetX（float32 LE，单位 mm，可为 0）
//      [6-9]   offsetY（float32 LE，单位 mm，可为 0）
//      [10-13] offsetZ（float32 LE，单位 mm，可为 0）
//      [14]    0xFF
//      [15]    0xEE
//      [16]    CRC8（覆盖 [0]~[15]）
//
//  BB CC 动作完成事件帧，固定 5 字节，CRC8 覆盖 [0]~[3]：
//    BB CC FF EE CRC8
//      [0] 0xBB
//      [1] 0xCC
//      [2] 0xFF
//      [3] 0xEE
//      [4] CRC8（覆盖 [0]~[3]）
// ============================================================

// ╔═══════════════════════════════════════════════════════════════╗
// ║                    代码阅读导航（建议先看这段）                  ║
// ╚═══════════════════════════════════════════════════════════════╝
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
//     send_*_cmd()      : 按 BB/4DOF 协议打包帧 → write_bytes()（循环写，断线级联重连）
//
//  D. 上层适配
//     handle_text_command()
//                       : 规范化 → 分词 → 4DOF 命令分派
//
//  E. 异常处理总览
//     - 串口断开：read/write EIO/ENODEV/ENXIO → close() + clear_report_state() + reconnect
//     - 帧同步失败：跳过坏字节重新搜索，永不阻塞
//     - libusb 不可用：保守策略返回 true（不误触发掉线）
//     - 参数越界：返回 false / 零值，不崩溃
// ============================================================

// ╔═══════════════════════════════════════════════════════════════╗
// ║              构造函数 / 析构函数                               ║
// ╚═══════════════════════════════════════════════════════════════╝
arm_internation::arm_internation() {}
arm_internation::~arm_internation() { close(); }

// ╔═══════════════════════════════════════════════════════════════╗
// ║              匿名命名空间：内部辅助工具函数                      ║
// ╚═══════════════════════════════════════════════════════════════╝
//  包含：波特率转换、字符串处理（大小写/trim/分割）、sysfs USB 查询、
//  数值解析（int/float/hex）、float 小端编解码、断线 errno 判断等。
namespace
{
    /**
     * @name 反馈帧兼容长度常量
     * @details BB/4DOF 位姿反馈固定 46 字节；动作完成事件 BB CC 是 5 字节短帧。
     * @{
     */
    static constexpr size_t k4DofFbFrameLen = 46;
    static constexpr size_t k4DofDoneFrameLen = 5;
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

// ╔═══════════════════════════════════════════════════════════════╗
// ║           串口连接与配置                                       ║
// ╚═══════════════════════════════════════════════════════════════╝
//  包括：open() 打开指定串口并配置 termios 8N1 原始模式，
//        open_by_HWid() 按硬件 ID 自动扫描并连接（含阻塞重试），
//        configure_auto_reconnect() 设置自动重连参数，
//        close() 关闭串口、is_open() 查询连接状态。
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
    return p == "compiled" ||
           p == "bb" ||
           p == "4dof" ||
           p == "dof4" ||
           p == "dof4_bb" ||
           p == "双臂";
}

ArmProtocol arm_internation::protocol() const
{
    return ArmProtocol::Dof4BB;
}

const char *arm_internation::protocol_name() const
{
    return "4dof";
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║           静态辅助方法（供 bb/cc 文件共享）                     ║
// ╚═══════════════════════════════════════════════════════════════╝
std::string arm_internation::to_upper_copy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                   { return static_cast<char>(std::toupper(c)); });
    return s;
}

bool arm_internation::parse_int_token(const std::string &token, int &value)
{
    std::string t = trim_copy(token);
    if (t.empty()) return false;
    char *end_ptr = nullptr;
    long parsed = std::strtol(t.c_str(), &end_ptr, 10);
    if (end_ptr == t.c_str() || *end_ptr != '\0') return false;
    value = static_cast<int>(parsed);
    return true;
}

union FloatBytes { float f; uint8_t b[4]; };

float arm_internation::decode_float_le(const uint8_t *src)
{
    FloatBytes fb{};
    fb.b[0] = src[0]; fb.b[1] = src[1];
    fb.b[2] = src[2]; fb.b[3] = src[3];
    return fb.f;
}

void arm_internation::encode_float_le(float value, uint8_t *dst)
{
    FloatBytes fb{}; fb.f = value;
    dst[0] = fb.b[0]; dst[1] = fb.b[1];
    dst[2] = fb.b[2]; dst[3] = fb.b[3];
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║           CRC8（SMBus 多项式 0x07）校验                        ║
// ╚═══════════════════════════════════════════════════════════════╝
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

// ╔═══════════════════════════════════════════════════════════════╗
// ║           反馈帧解析（接收路径）                                ║
// ╚═══════════════════════════════════════════════════════════════╝
bool arm_internation::parse_feedback_frame()
{
    return parse_4dof_feedback_frame();
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║           串口接收（三层级联检测）                              ║
// ╚═══════════════════════════════════════════════════════════════╝
//  receive_once(): libusb 掉线预检 → read() → 帧解析 → 状态更新
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
     *     搜索 BB/4DOF 完整帧、校验 CRC
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

// ╔═══════════════════════════════════════════════════════════════╗
// ║           状态读取接口                                         ║
// ╚═══════════════════════════════════════════════════════════════╝
//  get_arm_pos: 缩放为 int16_t 的兼容接口
//  get_arm_pos_float/get_4dof_pose_float: float 原值
//  get_sensor: 电磁阀 + 微动开关状态
//  consume_done_feedback_count: 取出一次性 BB CC 完成事件
//  set_decode_scale: 配置 int16_t 输出的缩放比例
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

Arm4DofPoseFloat arm_internation::get_4dof_pose_float(int arm_id) const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    return (arm_id >= 0 && arm_id < 2) ? dof4_pose_float_[arm_id] : Arm4DofPoseFloat{};
}

size_t arm_internation::consume_done_feedback_count()
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    const size_t count = pending_done_feedback_count_;
    pending_done_feedback_count_ = 0;
    return count;
}

Arm4DofDiagnostic arm_internation::get_last_diagnostic() const
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    return last_diagnostic_;
}

size_t arm_internation::consume_diagnostic_feedback_count()
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    const size_t count = pending_diagnostic_feedback_count_;
    pending_diagnostic_feedback_count_ = 0;
    return count;
}

void arm_internation::set_decode_scale(float pos_scale)
{
    std::lock_guard<std::mutex> lock(state_mutex_);

    // 非正值没有物理意义，保持原配置。
    if (pos_scale > 0.0f)
    {
        pos_scale_ = pos_scale;
    }

    // 内部只保留 float 状态；比例仅影响 get_arm_pos 的视图换算。
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║           串口写入（底层）                                      ║
// ╚═══════════════════════════════════════════════════════════════╝
//  write_bytes(): 循环写入直到全部完成或失败，断线时级联重连。
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


// ╔═══════════════════════════════════════════════════════════════╗
// ╔═══════════════════════════════════════════════════════════════╗
// ║           HWID 设备扫描与匹配                                   ║
// ╚═══════════════════════════════════════════════════════════════╝
//  find_ttys_by_HWid()  : 遍历 /dev 按 USB VID:PID 匹配 tty 设备
//  open_matching_tty_once(): 尝试打开第一个匹配设备
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

// ╔═══════════════════════════════════════════════════════════════╗
// ║           文本命令解析（分词 / 别名 / 数值提取）                 ║
// ╚═══════════════════════════════════════════════════════════════╝
//  normalize_cmd_text()     : 中英标点统一、去空格
//  parse_float_after_prefix(): 前缀式数值提取
//  parse_float_token()      : 纯浮点 token 解析
//  parse_4dof_arm_alias()   : BB 协议臂别名（L/R）
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

// ╔═══════════════════════════════════════════════════════════════╗
// ║           自动重连（有限状态机）                                 ║
// ╚═══════════════════════════════════════════════════════════════╝
//  reconnect_once()        : 带频率限制的阻塞重连（被内部调用）
//  try_reconnect_once()    : 非阻塞重连（供外部 ROS 循环调用）
//  is_bound_hwid_online_libusb(): libusb 掉线检测
//  clear_report_state()    : 断线后清空上报状态与接收缓存
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

        sensor_ = SensorStatus{};
        last_diagnostic_ = Arm4DofDiagnostic{};
        pending_done_feedback_count_ = 0;
        pending_diagnostic_feedback_count_ = 0;
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

// ╔═══════════════════════════════════════════════════════════════╗
// ║           文本命令分派（主入口）                                 ║
// ╚═══════════════════════════════════════════════════════════════╝
//  按首 token 判断协议类型，分派到对应的协议解析函数。
//  BB/4DOF 协议：handle_text_command_bb() → arm_internation_bb.cpp
//  CC 云台协议：handle_text_command_cc() → arm_internation_cc.cpp
bool arm_internation::handle_text_command(const std::string &command_text)
{
    const std::string normalized = normalize_cmd_text(command_text);
    if (normalized.empty()) return false;

    const std::vector<std::string> tokens = split_by_comma(normalized);
    if (tokens.empty()) return false;

    // 按协议分派：BB 优先（命令数量多），再尝试 CC
    if (handle_text_command_bb(tokens)) return true;
    if (handle_text_command_cc(tokens)) return true;
    return false;
}
