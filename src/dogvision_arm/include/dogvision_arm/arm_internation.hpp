#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <atomic>
#include <vector>

// ============================================================
//  协议数据结构
// ============================================================

/// 单根机械臂末端坐标（int16，单位由下位机协议确定）
struct ArmEndPos {
    int16_t x = 0;
    int16_t y = 0;
};

/// 机械臂末端坐标（float，已由原始编码值还原）
struct ArmEndPosFloat {
    float x = 0.0f;
    float y = 0.0f;
};

/// 4DOF 双臂末端位姿（float32，单位沿用下位机协议：m/rad 或实际固件定义）
struct Arm4DofPoseFloat {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float pitch = 0.0f;
};

/// 云台当前姿态角（int16，单位由下位机协议确定）
struct GimbalAngle {
    int16_t yaw   = 0;
    int16_t pitch = 0;
};

/// 云台姿态角（float，已由原始编码值还原）
struct GimbalAngleFloat {
    float yaw   = 0.0f;
    float pitch = 0.0f;
};

/// 电磁阀与微动开关的实时状态（4 路各自独立）
struct SensorStatus {
    std::array<bool, 4> valve       = {};  ///< 电磁阀 0-3：true = 开
    std::array<bool, 4> microswitch = {};  ///< 微动开关 0-3：true = 触发
};

struct PumpStatus {
        bool pump_on = false; ///< 泵状态：true = 开
        float speed = 0.0f;   ///< 泵速（单位由下位机协议确定）
};

/// 当前串口协议类型。
enum class ArmProtocol {
    PlaneAA, ///< 0xAA 包头：旧平面机械臂协议，兼容原有接口。
    Dof4BB,  ///< 0xBB 包头：4DOF 双臂协议，匹配 STM32 command_decode_4dof。
};

// ============================================================
//  arm_internation：串口通讯管理类
//
//  【职责分层】
//  1) 串口连接层：open/open_by_HWid/close/is_open
//     - 负责设备发现、参数配置、断线恢复（重试策略）
//  2) 协议收发层：receive_once/parse_feedback_frame/write_bytes/send_*_cmd
//     - 负责字节流读写、帧边界定位、CRC 校验、命令打包
//  3) 状态与命令适配层：get_* / handle_text_command
//     - 对外提供线程安全状态读取
//     - 将上层字符串命令适配为具体协议指令
//
//  协议帧说明（完整字节表见 arm_internation.cpp 顶部注释）：
//    AA 01  ← 下位机反馈（100 Hz）
//    AA 02  → 机械臂末端控制
//    AA 03  → 云台角度控制
//    AA 04  → 电磁阀控制
//    AA 06  → 泵控制
//    AA 05  → 任务赛答案控制
//
//    BB 01  ← 4DOF 双臂反馈
//    BB 02  → 4DOF 单臂位姿控制（arm_id + x/y/z/pitch）
//    BB 03  → 4DOF 预设动作触发
//    BB 04  → 4DOF 电磁阀控制
//    BB 05  → 4DOF 语音应答/答案控制（下位机预留）
//    BB 06  → 4DOF 气泵控制

//  用法（典型）：
//    arm_internation comm;
//    comm.set_protocol_from_string("4dof");  // 可选；默认是旧 AA 协议
//    comm.open("/dev/ttyUSB0", 115200);
//    // 接收线程：while(running) comm.receive_once();
//    comm.send_arm_cmd(0, 100, 200);
//    comm.send_4dof_pose_cmd(0, 0.1f, 0.2f, 0.3f, 0.4f);
//    ArmEndPos pos = comm.get_arm_pos(0);
//
//  文本命令示例（通常由 ROS 字符串话题传入）：
//    RL,X:10,Y:10      -> 控制机械臂（别名映射见 parse_arm_alias）
//    4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4 -> 4DOF 左臂位姿
//    4ACT,1            -> 4DOF 触发预设动作 1；4ACT,0 表示中止
//    G,0,0             -> 控制云台 yaw/pitch
//    V,1               -> 翻转电磁阀 1 的状态
//    V,1,ON            -> 显式打开电磁阀 1
//    P,ON,2000           -> 打开/关闭泵,并设置泵速（示例：P,ON,2000 打开泵并设置速度为 2000）
//    A,0               -> 任务赛答案控制（示例：A,0 设置答案为 0）
// ============================================================
class arm_internation {
public:
    /// 构造通信对象（默认未连接串口）。
    arm_internation();
    /// 析构时自动关闭串口句柄。
    ~arm_internation();

    // ---- 串口管理 -----------------------------------------------
    /// 打开串口；失败时返回 false 并在 stderr 输出错误原因。
    bool open(const std::string& port, int baud_rate);
    /// 通过硬件ID在 /dev/ttyACM* 与 /dev/ttyUSB* 中检索并连接，失败时按 retry_ms 重试。
    bool open_by_HWid(const std::string& hw_id, int baud_rate, int retry_ms = 1000);
    /// 只保存自动重连目标，不阻塞等待设备；适合 ROS 主循环中周期调用 try_reconnect_once。
    void configure_auto_reconnect(const std::string& hw_id, int baud_rate, int retry_ms = 1000);
    /// 关闭串口（析构时自动调用）。
    void close();
    /// 返回串口是否已打开。
    bool is_open() const;
    /// 非阻塞单次重连尝试：扫描 ttyACM*/ttyUSB* 并尝试按原 hw_id/baud 打开。
    /// 成功返回 true；未找到设备或打开失败立即返回 false，不等待。
    bool try_reconnect_once();

    // ---- 协议模式 ------------------------------------------------
    /// 根据字符串设置协议；支持 aa/plane 与 bb/4dof，非法字符串返回 false 且不改变当前模式。
    bool set_protocol_from_string(const std::string& protocol_name);
    /// 获取当前协议枚举。
    ArmProtocol protocol() const;
    /// 返回当前协议名："aa" 或 "4dof"。
    const char* protocol_name() const;

    // ---- 接收 & 解析 --------------------------------------------
    /// 从串口读取数据并按当前协议尝试解析反馈帧。
    /// 建议在独立接收线程中循环调用，成功解析时返回 true。
    bool receive_once();

    // ---- 状态读取（线程安全）-------------------------------------
    /// 以 int16 视图读取机械臂末端坐标（由 float 状态按 scale 换算得到）。
    /// arm_id: 0=LF 1=RF 2=LB 3=RB
    ArmEndPos    get_arm_pos(int arm_id) const;
    /// 以 int16 视图读取云台角度（由 float 状态按 scale 换算得到）。
    GimbalAngle  get_gimbal()            const;
    /// 读取电磁阀与微动开关状态位。
    SensorStatus get_sensor()            const;

    /// 获取还原后的 float 坐标与角度（scale 见 set_decode_scale）。
    ArmEndPosFloat   get_arm_pos_float(int arm_id) const;
    GimbalAngleFloat get_gimbal_float()             const;
    /// 获取 4DOF 双臂位姿；arm_id: 0=左臂, 1=右臂。
    Arm4DofPoseFloat get_4dof_pose_float(int arm_id) const;

    /// 配置 float->int16 兼容视图的换算比例。
    /// 说明：内部状态始终保存为 float；该比例仅影响 get_arm_pos/get_gimbal 的返回值。
    void set_decode_scale(float pos_scale, float angle_scale);

    // ---- 发送命令 ------------------------------------------------
    /// AA 02：控制 arm_id 机械臂末端移动至 (x, y)，float 坐标，arm_id: 0-3。
    bool send_arm_cmd(int arm_id, float x, float y);

    /// AA 03：控制云台角度（当前协议为 float 参数），gimbal_id 目前仅支持 0。
    bool send_gimbal_cmd(int gimbal_id, float yaw, float pitch);

    /// AA 04：控制电磁阀，valve_id: 0-3，state: true = 开 / false = 关。
    bool send_valve_cmd(int valve_id, bool state);

    /// AA 05：发送任务赛答案，answer: 0-3。
    bool send_answer_cmd(uint8_t answer);

    /// AA 06：控制气泵开关与速度，on=true 开泵并设 speed，on=false 关泵。
    bool send_pump_cmd(bool on, int speed);

    /// BB 02：4DOF 单臂位姿控制，arm_id: 0=左臂, 1=右臂。
    bool send_4dof_pose_cmd(int arm_id, float x, float y, float z, float pitch);

    /// BB 03：4DOF 预设动作控制，action_id=0 表示中止当前动作。
    bool send_4dof_action_cmd(uint8_t action_id);

    /// 解析并执行字符串命令：
    /// RL,X:10,Y:10 -> AA 机械臂；4POSE,L,... -> BB 4DOF；V/P 自动按协议打包。
    bool handle_text_command(const std::string& command_text);

private:
    // ---- 协议常量 -----------------------------------------------
    static constexpr uint8_t kHeadA   = 0xAAu;  ///< 帧头第一字节
    static constexpr uint8_t kHeadB   = 0xBBu;  ///< 4DOF 帧头
    static constexpr uint8_t kCmdFb   = 0x01u;  ///< 命令字：下位机反馈
    static constexpr uint8_t kCmdArm  = 0x02u;  ///< 命令字：机械臂控制
    static constexpr uint8_t kCmdGim  = 0x03u;  ///< 命令字：云台控制
    static constexpr uint8_t kCmdValv = 0x04u;  ///< 命令字：电磁阀控制
    static constexpr uint8_t kCmdAns  = 0x05u;  ///< 命令字：任务赛答案
    static constexpr uint8_t kCmdPump = 0x06u;  ///< 命令字：气泵控制（on/off + 速度）
    static constexpr uint8_t kTailA   = 0xFFu;  ///< 帧尾第一字节
    static constexpr uint8_t kTailB   = 0xEEu;  ///< 帧尾第二字节


    //收放位置
    static constexpr float aimPos_start_X[4] = {10.0f, 10.0f, 10.0f, 10.0f}; ///< 目标 X 坐标（示例值）
    static constexpr float aimPos_start_Y[4] = {10.0f, 10.0f, 10.0f, 10.0f}; ///< 目标 Y 坐标（示例值）

    //放置物块位置
    static constexpr float aimPos_place_X[4] = {10.0f, 10.0f, 10.0f, 10.0f}; ///< 目标 X 坐标（示例值）
    static constexpr float aimPos_place_Y[4] = {10.0f, 10.0f, 10.0f, 10.0f}; ///< 目标 Y 坐标（示例值）

    //取物块位置设定
    static constexpr float aimPos_block_get_X[4] = {10.0f, 10.0f, 10.0f, 10.0f}; ///< 目标 X 坐标（示例值）
    static constexpr float aimPos_block_get_Y[4] = {10.0f, 10.0f, 10.0f, 10.0f}; ///< 目标 Y 坐标（示例值）

    // 任务赛特定：交接物块
    static constexpr float aimPos_block_transfer_X[4] = {10.0f, 10.0f, 10.0f, 10.0f}; ///< 目标 X 坐标（示例值）
    static constexpr float aimPos_block_transfer_Y[4] = {10.0f, 10.0f, 10.0f, 10.0f}; ///< 目标 Y 坐标（示例值）



    /// AA 01 帧总字节数：2(头) + 44(净荷) + 2(尾) + 1(CRC) = 49
    static constexpr int kFbFrameLen  = 49;
    /// AA 01 净荷字节数：4臂*8(float x/y) + 2云台*4(float) + 4传感器 = 44
    static constexpr int kFbPayloadLen = 44;
    /// BB 01 帧总字节数：2(头/命令) + 41(DATA) + 2(尾) + 1(CRC) = 46
    static constexpr int k4DofFbFrameLen = 46;

    // ---- CRC-8/SMBUS（多项式 0x07）------------------------------
    /// 对 data[0..len-1] 计算 CRC-8。
    static uint8_t calc_crc8(const uint8_t* data, size_t len);

    // ---- 帧解析 & 写 --------------------------------------------
    /// 按当前协议解析反馈帧并更新内部状态。
    bool parse_feedback_frame();
    /// 在接收缓冲区中搜索完整 AA 01 帧并更新内部状态。
    bool parse_plane_feedback_frame();
    /// 在接收缓冲区中搜索完整 BB 01 帧并更新内部状态。
    bool parse_4dof_feedback_frame();
    /// 将 len 字节写入串口。
    bool write_bytes(const uint8_t* data, size_t len);

    /// 从 /dev 目录筛选 ttyACM*/ttyUSB*，并按 idVendor:idProduct 匹配目标设备。
    static std::vector<std::string> find_ttys_by_HWid(const std::string& hw_id);
    /// 尝试打开某个 HWID 下所有匹配串口，直到成功或候选耗尽。
    bool open_matching_tty_once(const std::string& hw_id, int baud_rate, const char* log_context);
    /// 规范化文本命令：处理中英文标点、空格和分隔符。
    static std::string normalize_cmd_text(std::string text);
    /// 从类似 X:10 / Y=20 的 token 中提取整数值（用于旧命令分支）。
    static bool parse_int_after_prefix(const std::string& token, const std::string& prefix, int& value);
    /// 从字符串 token 提取浮点数。
    static bool parse_float_token(const std::string& token, float& value);
    /// 从类似 X:10.5 / Y=20 的 token 中提取浮点值。
    static bool parse_float_after_prefix(const std::string& token, const std::string& prefix, float& value);
    /// 机械臂名称别名映射：LF/RF/LB/RB（兼容 RL 等历史写法）。
    bool parse_arm_alias(const std::string& alias, int& arm_id) const;
    /// 4DOF 双臂别名映射：L/LEFT/0 与 R/RIGHT/1。
    static bool parse_4dof_arm_alias(const std::string& alias, int& arm_id);

    /// 断开后自动重连（非阻塞，只尝试一轮可用候选）。
    bool reconnect_once();
    /// 使用 libusb 检查当前绑定 HWID 对应 USB 设备是否在线。
    bool is_bound_hwid_online_libusb() const;
    /// 断线期间将内部上报缓存清零，避免上层继续读取陈旧值。
    void clear_report_state();

    // ---- 串口文件描述符 -----------------------------------------
    int fd_ = -1;

    // ---- 接收缓冲区 ---------------------------------------------
    static constexpr size_t kBufSize = 512;
    uint8_t rx_buf_[kBufSize] = {};
    size_t  rx_len_           = 0;

    // ---- 状态缓存（受互斥锁保护）--------------------------------
    mutable std::mutex state_mutex_;
    ArmEndPosFloat     arm_pos_float_[4] = {};
    Arm4DofPoseFloat   dof4_pose_float_[2] = {};
    GimbalAngleFloat   gimbal_float_     = {};
    SensorStatus       sensor_     = {};

    // ---- 协议模式 ------------------------------------------------
    ArmProtocol protocol_ = ArmProtocol::PlaneAA;

    // ---- 解码比例（raw int16 -> float）--------------------------
    float pos_scale_ = 0.01f;
    float angle_scale_ = 0.01f;

    // ---- 发送互斥锁（防止多线程同时写串口）----------------------
    std::mutex send_mutex_;

    // ---- 电磁阀命令缓存（仅用于 V,ID 无状态参数时的翻转）---------
    mutable std::mutex valve_cmd_mutex_;
    std::array<bool, 4> valve_cmd_state_ = {false, false, false, false};

    // ---- 自动重连配置（由 open_by_HWid 设置）----------------------
    std::string reconnect_hw_id_;
    int reconnect_baud_rate_ = 115200;
    int reconnect_retry_ms_ = 1000;
    std::atomic<bool> auto_reconnect_enabled_{false};
    std::mutex reconnect_mutex_;
    std::chrono::steady_clock::time_point last_reconnect_attempt_tp_{};

    // ---- libusb 在线检测节流 -----------------------------------
    std::chrono::steady_clock::time_point last_usb_check_tp_{};
    int usb_check_interval_ms_ = 300;
};
