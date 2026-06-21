#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <atomic>
#include <vector>

#include <dogvision_arm/protocol_config.hpp>

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
//  【设计理念】
//  本类是整个机械臂通信栈的核心抽象，将"串口硬件"与"协议解析"解耦。
//  上层（ROS 节点）只需关注命令文本与状态读取，
//  无需关心字节级组帧、CRC 校验、断线恢复等底层细节。
//
//  【职责分层】
//  1) 串口连接层：open/open_by_HWid/close/is_open
//     - 负责设备发现（sysfs/libusb 双重校验）、参数配置、断线恢复（有限状态机）
//  2) 协议收发层：receive_once/parse_feedback_frame/write_bytes/send_*_cmd
//     - 负责字节流读写、帧边界定位、CRC 校验、命令打包
//  3) 状态与命令适配层：get_* / handle_text_command
//     - 对外提供线程安全状态读取（互斥锁保护所有共享缓存）
//     - 将上层字符串命令适配为具体协议指令
//
//  【线程安全模型】
//  - state_mutex_: 保护所有上报状态缓存（arm_pos_float_/dof4_pose_float_/gimbal_float_/sensor_）
//  - send_mutex_: 保证多线程写串口时不发生字节交叉
//  - valve_cmd_mutex_: 保护电磁阀命令翻转缓存（仅用于 V,id 无状态参数场景）
//  - reconnect_mutex_: 保护自动重连配置与 fd_ 的竞态访问
//
//  【自动重连状态机（FSM）】
//  状态转换条件：
//  ┌────────────┐    open/open_by_HWid 成功    ┌────────────┐
//  │ DISCONNECTED│────────────────────────────▶│  CONNECTED  │
//  │ (fd_ < 0)  │◀────────────────────────────│ (fd_ >= 0)  │
//  └─────┬──────┘  read/write 返回 EIO/ENODEV  └─────┬──────┘
//        │            或 libusb 检测掉线                │
//        │                                            │
//        │  receive_once() 级联调用                    │ receive_once() 内
//        │  reconnect_once()                          │ 每 usb_check_interval_ms_
//        │  (间隔≥retry_ms_ 时扫描候选)                │ 用 libusb 轮询 HWID 存在性
//        │                                            │
//        └──────────── 重试循环 ◀──────────────────────┘
//
//  libusb 辅助检测的设计意图：
//  Linux 串口驱动在 USB 设备拔出后可能延迟数十秒才返回 EIO，
//  单纯依赖 read() 错误码会导致上位机长时间读取陈旧状态。
//  libusb 可以直接查询 USB 总线设备列表，实现亚秒级掉线感知。
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
//    BB 11  → 4DOF 单臂取块动作（arm_id + x/y/z，单位 m）
//    BB 12  → 4DOF 单臂放块第一层动作（arm_id + x/y/z，单位 m）
//    BB 13  → 4DOF 单臂放块第二层动作（arm_id + x/y/z，单位 m）
//    BB 14  → 4DOF 单臂放块到背部固定动作（arm_id）
//    BB 15  → 4DOF 单臂从背部取块固定动作（arm_id）
//    BB 21  → 4DOF 双臂取块动作（left xyz + right xyz，单位 m）
//    BB 22  → 4DOF 双臂放块到背部固定动作
//    BB 99  → 4DOF 带初始偏移启动（offsetX/Y/Z，float32 小端，单位 mm）
//    BB CC  ← 4DOF 当前动作完成事件；ROS 节点会转发为 /arm_internation/state = "DONE"

//  用法（典型）：
//    arm_internation comm;
//    comm.set_protocol_from_string("compiled");  // 可选；仅校验编译时锁定的协议
//    comm.open("/dev/ttyUSB0", 115200);
//    // 接收线程：while(running) comm.receive_once();
//    comm.send_arm_cmd(0, 100, 200);
//    comm.send_4dof_pose_cmd(0, 0.1f, 0.2f, 0.3f, 0.4f);
//    comm.send_4dof_start_cmd(0.0f, 0.0f, 0.0f);  // 带初始偏移启动
//    ArmEndPos pos = comm.get_arm_pos(0);
//
//  文本命令示例（通常由 ROS 字符串话题传入）：
//    RL,X:10,Y:10      -> 控制机械臂（别名映射见 parse_arm_alias）
//    4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4 -> 4DOF 左臂位姿
//    4ACT,1            -> 4DOF 触发预设动作 1；4ACT,0 表示中止
//    4PICK,L,0.45,0.42,-0.21       -> 4DOF 左臂按 PC 目标点取块
//    4PLACE1,R,0.45,-0.40,-0.21    -> 4DOF 右臂放块第一层
//    4PLACE2,L,X:0.45,Y:0.40,Z:0.04 -> 4DOF 左臂放块第二层
//    4PUTBACK,L        -> 4DOF 左臂放块到背部
//    4GETBACK,R        -> 4DOF 右臂从背部取块
//    4PICKALL,0.45,0.42,-0.21,0.45,-0.42,-0.21 -> 4DOF 双臂取块
//    4PUTBACKALL       -> 4DOF 双臂放块到背部
//    START,0,0,0        -> 4DOF 带 X/Y/Z 初始偏移启动（单位 mm）
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
    // ---- 串口管理 -----------------------------------------------
    /// @brief 打开指定路径串口并配置为 8N1 原始模式。
    /// @param port 设备路径，如 "/dev/ttyUSB0"
    /// @param baud_rate 波特率（9600 到 921600）
    /// @retval true 连接成功
    /// @retval false 连接失败（权限不足/设备不存在/波特率不支持）
    /// @note 内部先 close() 再 open()，避免重复打开导致 fd 泄漏。
    ///       失败时在 stderr 输出 errno 可读信息。
    bool open(const std::string& port, int baud_rate);

    /// @brief 通过 USB 硬件 ID 自动发现并连接串口设备。
    /// @param hw_id 格式 "VID:PID"（十六进制），例如 "0483:5740"
    /// @param baud_rate 波特率
    /// @param retry_ms 失败重试间隔（毫秒），默认 1000ms
    /// @retval true 连接成功（该函数只有成功时才返回）
    /// @retval 永不返回 false（设计为阻塞式常驻重试）
    /// @note 适用场景：设备可能晚于上位机插入，或启动脚本需要等待外设就绪。
    ///       内部调用 configure_auto_reconnect() 持久化参数，
    ///       后续断线时 receive_once() 可自动恢复。
    bool open_by_HWid(const std::string& hw_id, int baud_rate, int retry_ms = 1000);

    /// @brief 配置自动重连参数（不立即阻塞尝试连接）。
    /// @param hw_id 目标 USB 硬件 ID
    /// @param baud_rate 目标波特率
    /// @param retry_ms 重连尝试间隔（毫秒）
    /// @note 仅保存参数并启用自动重连标志；
    ///       实际重连由 receive_once()/try_reconnect_once() 触发。
    ///       适用于 ROS 主循环场景，避免构造函数阻塞 spin。
    void configure_auto_reconnect(const std::string& hw_id, int baud_rate, int retry_ms = 1000);

    /// @brief 关闭串口并释放文件描述符。
    /// @note 析构时自动调用；对已关闭的 fd（-1）重复调用安全。
    void close();

    /// @brief 查询串口连接状态。
    /// @retval true 串口已打开（fd_ >= 0）
    /// @retval false 串口未打开
    bool is_open() const;

    /// @brief 非阻塞单次重连尝试。
    /// @retval true 重连成功
    /// @retval false 设备未就绪或打开失败（立即返回，不等待）
    /// @note 按 retry_ms_ 控制尝试频率，避免 CPU 空转。
    ///       扫描 /dev/ttyACM* 和 /dev/ttyUSB*，按 HWID 匹配。
    bool try_reconnect_once();

    // ---- 协议模式 ------------------------------------------------
    /// @brief 校验协议名称是否与编译时锁定的协议一致。
    /// @param protocol_name "compiled"，或当前编译协议对应的 AA/BB 别名
    /// @retval true 参数与编译协议一致
    /// @retval false 参数非法或请求了另一种协议
    /// @note 该接口为源码兼容而保留，不再切换协议。
    bool set_protocol_from_string(const std::string& protocol_name);

    /// @brief 获取编译时锁定的协议枚举值。
    ArmProtocol protocol() const;

    /// @brief 获取编译时锁定的协议名称字符串（"aa" 或 "4dof"）。
    const char* protocol_name() const;

    // ---- 接收 & 解析 --------------------------------------------
    /// @brief 主接收循环入口：读串口 + 解析反馈帧 + 掉线检测。
    /// @retval true 本次调用成功解析出一帧有效反馈
    /// @retval false 无完整帧（数据不足/校验失败/串口未打开）
    /// @note 该函数集成三层逻辑：
    ///       1) libusb 掉线预检（仅 auto_reconnect 模式）
    ///       2) read() 读取字节追加到流式缓存
    ///       3) parse_feedback_frame() 帧同步与 CRC 校验
    ///       建议在 200Hz 以上的循环中调用以匹配 100Hz 下位机上报。
    bool receive_once();

    // ---- 状态读取（线程安全）-------------------------------------
    /// @brief 以 int16 定标视图读取机械臂末端坐标。
    /// @param arm_id 机械臂编号：0=LF 1=RF 2=LB 3=RB
    /// @retval ArmEndPos 坐标（float 内部值 / pos_scale_ 后取整）
    /// @note 内部状态始终保存为 float；int16 视图仅为兼容旧代码。
    ///       越界 arm_id 返回零值。
    ArmEndPos    get_arm_pos(int arm_id) const;

    /// @brief 以 int16 定标视图读取云台角度。
    GimbalAngle  get_gimbal()            const;

    /// @brief 读取电磁阀与微动开关的实时状态位。
    /// @retval SensorStatus 4 路各自独立的布尔状态
    SensorStatus get_sensor()            const;

    /// @brief 获取 float 精度机械臂坐标（未经定标换算）。
    ArmEndPosFloat   get_arm_pos_float(int arm_id) const;

    /// @brief 获取 float 精度云台角度。
    GimbalAngleFloat get_gimbal_float()             const;

    /// @brief 获取 4DOF 双臂位姿（仅 BB 协议有效）。
    /// @param arm_id 0=左臂, 1=右臂
    /// @retval Arm4DofPoseFloat 位姿（xyz + pitch），非 BB 协议或越界返回零值
    Arm4DofPoseFloat get_4dof_pose_float(int arm_id) const;

    /// @brief 取出并清零 4DOF 动作完成事件数量。
    /// @retval size_t 自上次调用以来收到的有效 BB CC 完成反馈帧数量
    /// @note BB CC 是一次性事件，不属于连续位姿状态；ROS 节点用它逐条发布 DONE。
    size_t consume_done_feedback_count();

    /// @brief 配置 int16 视图的换算比例。
    /// @param pos_scale 位置缩放因子（仅正值生效）
    /// @param angle_scale 角度缩放因子（仅正值生效）
    /// @note 不影响内部 float 缓存，仅影响 get_arm_pos/get_gimbal 的返回值。
    void set_decode_scale(float pos_scale, float angle_scale);

    // ---- 发送命令 ------------------------------------------------
    /// @brief 发送 AA 协议机械臂末端控制命令。
    /// @param arm_id 机械臂编号 [0,3]
    /// @param x 目标 X 坐标（float，单位由下位机定义）
    /// @param y 目标 Y 坐标（float，单位由下位机定义）
    /// @retval true 发送成功
    /// @retval false 发送失败（串口断开/协议不匹配）
    /// @note BB 协议下返回 false；4DOF 模式应使用 send_4dof_pose_cmd()。
    bool send_arm_cmd(int arm_id, float x, float y);

    /// @brief 发送 AA 协议云台角度控制命令。
    /// @param gimbal_id 云台编号（当前仅支持 0）
    /// @param yaw 偏航角（float）
    /// @param pitch 俯仰角（float）
    /// @retval true 发送成功
    /// @retval false BB 协议下返回 false（4DOF 无云台命令）
    bool send_gimbal_cmd(int gimbal_id, float yaw, float pitch);

    /// @brief 发送电磁阀控制命令（AA 和 BB 通用）。
    /// @param valve_id 电磁阀编号 [0,3]
    /// @param state true=开 / false=关
    /// @retval true 发送成功
    /// @retval false 发送失败
    bool send_valve_cmd(int valve_id, bool state);

    /// @brief 发送任务赛答案/语音控制命令。
    /// @param answer 答案编号 [0,255]，含义由下位机固件定义
    /// @retval true 发送成功
    bool send_answer_cmd(uint8_t answer);

    /// @brief 发送气泵控制命令（AA 和 BB 通用）。
    /// @param on true=开泵并设速度，false=关泵
    /// @param speed 泵速（int，单位由下位机定义，on=false 时忽略）
    /// @retval true 发送成功
    bool send_pump_cmd(bool on, int speed);

    /// @brief 发送 4DOF 单臂位姿控制命令（仅 BB 协议）。
    /// @param arm_id 0=左臂, 1=右臂
    /// @param x X 坐标（float）
    /// @param y Y 坐标（float）
    /// @param z Z 坐标（float）
    /// @param pitch 末端俯仰角（float）
    /// @retval true 发送成功
    /// @retval false 非 BB 协议或 arm_id 越界
    bool send_4dof_pose_cmd(int arm_id, float x, float y, float z, float pitch);

    /// @brief 发送 4DOF 预设动作触发命令（仅 BB 协议）。
    /// @param action_id 动作编号：0=中止当前动作，1..N=触发对应预设动作
    /// @retval true 发送成功
    /// @retval false 非 BB 协议
    bool send_4dof_action_cmd(uint8_t action_id);

    /// @brief 发送 4DOF 带初始偏移启动命令（仅 BB 协议）。
    /// @param offset_x X 方向初始偏移，单位 mm，按 float32 小端直接发送
    /// @param offset_y Y 方向初始偏移，单位 mm，按 float32 小端直接发送
    /// @param offset_z Z 方向初始偏移，单位 mm，按 float32 小端直接发送
    /// @retval true 发送成功
    /// @retval false 非 BB 协议或串口写入失败
    /// @note 帧格式：BB 99 offsetX offsetY offsetZ FF EE CRC8，CRC 覆盖 CRC 前所有字节。
    bool send_4dof_start_cmd(float offset_x, float offset_y, float offset_z);

    /// @brief 发送 4DOF 单臂取块动作命令（仅 BB 协议）。
    /// @param arm_id 0=左臂, 1=右臂
    /// @param x 取块目标 X，单位 m，按 float32 小端直接发送
    /// @param y 取块目标 Y，单位 m，按 float32 小端直接发送
    /// @param z 取块目标 Z，单位 m，按 float32 小端直接发送
    /// @retval true 发送成功
    /// @retval false 非 BB 协议、arm_id 越界、坐标非法或串口写入失败
    /// @note 帧格式：BB 11 arm_id x y z FF EE CRC8，CRC 覆盖 CRC 前所有字节。
    bool send_4dof_pick_cmd(int arm_id, float x, float y, float z);

    /// @brief 发送 4DOF 单臂放块第一层动作命令（仅 BB 协议）。
    /// @note 帧格式：BB 12 arm_id x y z FF EE CRC8，xyz 单位 m，不做 mm/m 换算。
    bool send_4dof_place_1f_cmd(int arm_id, float x, float y, float z);

    /// @brief 发送 4DOF 单臂放块第二层动作命令（仅 BB 协议）。
    /// @note 帧格式：BB 13 arm_id x y z FF EE CRC8，xyz 单位 m，不做 mm/m 换算。
    bool send_4dof_place_2f_cmd(int arm_id, float x, float y, float z);

    /// @brief 发送 4DOF 单臂放块到背部固定动作命令（仅 BB 协议）。
    /// @param arm_id 0=左臂, 1=右臂
    /// @note 帧格式：BB 14 arm_id FF EE CRC8。该动作不带 xyz，目标由下位机模板决定。
    bool send_4dof_put_block_back_cmd(int arm_id);

    /// @brief 发送 4DOF 单臂从背部取块固定动作命令（仅 BB 协议）。
    /// @param arm_id 0=左臂, 1=右臂
    /// @note 帧格式：BB 15 arm_id FF EE CRC8。该动作不带 xyz，目标由下位机模板决定。
    bool send_4dof_get_block_back_cmd(int arm_id);

    /// @brief 发送 4DOF 双臂取块动作命令（仅 BB 协议）。
    /// @param lx 左臂目标 X，单位 m
    /// @param ly 左臂目标 Y，单位 m
    /// @param lz 左臂目标 Z，单位 m
    /// @param rx 右臂目标 X，单位 m
    /// @param ry 右臂目标 Y，单位 m
    /// @param rz 右臂目标 Z，单位 m
    /// @note 帧格式：BB 21 left_x left_y left_z right_x right_y right_z FF EE CRC8。
    bool send_4dof_pick_all_cmd(float lx, float ly, float lz, float rx, float ry, float rz);

    /// @brief 发送 4DOF 双臂放块到背部固定动作命令（仅 BB 协议）。
    /// @note 帧格式：BB 22 FF EE CRC8。无 DATA 段，双臂目标由下位机动作模板决定。
    bool send_4dof_put_block_back_all_cmd();

    /// @brief 解析并执行文本命令（统一命令入口）。
    /// @param command_text 原始命令字符串，支持中英文标点容错
    /// @retval true 命令解析成功并已发送
    /// @retval false 格式错误或发送失败
    /// @note 命令格式：
    ///   - AA 机械臂: "LF,X:10,Y:20" / "RF,10,20"
    ///   - BB 4DOF:   "4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4"
    ///   - BB 动作:   "4ACT,1" / "4ACT,0"（中止）
    ///   - BB 新动作: "4PICK,L,0.45,0.42,-0.21" / "4PUTBACKALL"
    ///   - BB 启动:   "START,0,0,0" / "START,X:0,Y:0,Z:0"（偏移单位 mm）
    ///   - 云台:      "G,0,0"
    ///   - 电磁阀:    "V,1"（翻转）/ "V,1,ON"（显式）/ "V,ALL,ON"
    ///   - 气泵:      "P,ON,2500" / "P,OFF"
    ///   - 答案:      "A,0"
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

    static constexpr uint8_t kCmd4DofPICK = 0x11u;                ///< BB 命令字：4DOF 单臂取块位姿控制，可控制末端点
    static constexpr uint8_t kCmd4DofPLACE_1F = 0x12u;            ///< BB 命令字：4DOF 单臂放块第一层位姿控制，可控制末端点
    static constexpr uint8_t kCmd4DofPLACE_2F = 0x13u;            ///< BB 命令字：4DOF 单臂放块第二层位姿控制，可控制末端点
    static constexpr uint8_t kCmd4DofPUT_BLOCK_BACK = 0x14u;      ///< BB 命令字：4DOF 单臂放块到背部，不可控末端点
    static constexpr uint8_t kCmd4DofGET_BLOCK_BACK = 0x15u;      ///< BB 命令字：4DOF 单臂背部取块，不可控末端点

    static constexpr uint8_t kCmd4DofPICK_ALL = 0x21u;            ///< BB 命令字：4DOF 双臂取块位姿控制，可控制左右臂末端点
    static constexpr uint8_t kCmd4DofPUT_BLOCK_BACK_ALL = 0x22u;  ///< BB 命令字：4DOF 双臂放块到背部，不可控制左右臂末端点

    static constexpr uint8_t kCmd4DofStart = 0x99u; ///< BB 命令字：带初始偏移启动
    static constexpr uint8_t kCmd4DofDone  = 0xCCu; ///< BB 事件字：当前动作完成反馈
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
    /// BB CC 完成事件帧总字节数：BB + CC + FF + EE + CRC = 5
    static constexpr int k4DofDoneFrameLen = 5;

    // ---- CRC-8/SMBUS（多项式 0x07）------------------------------
    /// 对 data[0..len-1] 计算 CRC-8。
    static uint8_t calc_crc8(const uint8_t* data, size_t len);

    // ---- 帧解析 & 写 --------------------------------------------
    /// @brief 按编译时锁定的协议分派反馈帧解析。
    /// @retval true 成功解析一帧并更新状态缓存
    /// @retval false 未找到有效帧（等待更多数据）
    /// @note 内部可能消耗部分 rx_buf_ 字节（找到帧头但数据不足时保留帧头）。
    bool parse_feedback_frame();

    /// @brief AA 协议反馈帧解析状态机。
    /// @details 核心同步策略：
    ///   1) 滑动窗口搜索 0xAA 0x01 帧头
    ///   2) 对候选位置同时校验 49B（V1）和 53B（V2）两种帧长
    ///   3) 验证 tail（0xFF 0xEE）+ CRC8 完整性
    ///   4) 数据不足时保留帧头等待拼接；校验失败时丢弃一字节重同步
    /// @retval true 成功解析并更新 arm_pos_float_/gimbal_float_/sensor_
    /// @retval false 无完整有效帧
    bool parse_plane_feedback_frame();

    /// @brief BB 协议反馈帧解析。
    /// @details 固定 46 字节帧，同步策略与 AA 类似但帧长单一。
    ///   额外将 4DOF 左/右臂 x/y 投影到 AA 兼容字段 arm_pos_float_[0]/[1]。
    /// @retval true 成功解析并更新 dof4_pose_float_/arm_pos_float_/sensor_
    /// @retval false 无完整有效帧
    bool parse_4dof_feedback_frame();

    /// @brief 完整写入指定字节到串口。
    /// @param data 待发送数据指针
    /// @param len 数据长度（字节）
    /// @retval true 全部写入成功
    /// @retval false 写入失败（部分写入或串口错误）
    /// @note 内部循环 write() 处理部分写入；遇到断线错误时级联触发 reconnect_once()。
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

    /// @brief 断开后自动重连（内部调用，由 receive_once/write_bytes 级联触发）。
    /// @retval true 重连成功
    /// @retval false 条件不满足（auto_reconnect 未启用 / 间隔不足 / 无候选设备）
    /// @note 频率由 reconnect_retry_ms_ 控制；加 reconnect_mutex_ 防止并发重连。
    bool reconnect_once();

    /// @brief 使用 libusb 检查当前绑定的 USB 设备是否在线。
    /// @retval true 设备在线或未配置 HWID（保守策略：不触发掉线）
    /// @retval false HWID 配置异常或设备确实不在 USB 总线
    /// @note 设计意图：Linux 串口驱动掉线检测延迟可达数十秒，
    ///       libusb 直接查询 USB 总线可在亚秒级感知设备移除。
    bool is_bound_hwid_online_libusb() const;

    /// @brief 断线期间清空所有内部上报缓存。
    /// @note 将 arm_pos_float_/dof4_pose_float_/gimbal_float_/sensor_ 全部归零，
    ///       同时清空接收缓冲区 rx_len_，避免上层读取陈旧/无效数据。
    void clear_report_state();

    // ---- 串口文件描述符 -----------------------------------------
    int fd_ = -1;  ///< 串口文件描述符，-1 表示未打开（DISCONNECTED 状态）

    // ---- 接收缓冲区 ---------------------------------------------
    /// @name 流式接收缓冲区
    /// @details 串口 read() 不保证每次返回完整帧，采用循环缓冲区累积字节。
    ///          parse_feedback_frame() 消费前方完整帧后 memmove 压缩剩余字节。
    /// @{
    static constexpr size_t kBufSize = 512;     ///< 缓冲区容量（字节）
    uint8_t rx_buf_[kBufSize] = {};             ///< 字节缓冲区
    size_t  rx_len_           = 0;              ///< 缓冲区有效字节数
    /// @}

    // ---- 状态缓存（受 state_mutex_ 保护）-------------------------
    /// @name 上报状态缓存（线程安全）
    /// @details 所有 get_* 方法通过 state_mutex_ 串行化访问。
    ///          写入发生在 parse_*_feedback_frame() 校验通过后。
    /// @{
    mutable std::mutex state_mutex_;             ///< 状态读写互斥锁
    ArmEndPosFloat     arm_pos_float_[4] = {};   ///< AA 协议：4 臂末端 float 坐标
    Arm4DofPoseFloat   dof4_pose_float_[2] = {}; ///< BB 协议：左/右臂 4DOF 位姿
    GimbalAngleFloat   gimbal_float_     = {};   ///< AA 协议：云台角度
    SensorStatus       sensor_     = {};         ///< 电磁阀 + 微动开关状态
    size_t             pending_done_feedback_count_ = 0; ///< 待 ROS 节点发布的 BB CC 完成事件数量
    /// @}

    // ---- 解码比例（raw int16 -> float）--------------------------
    /// @name int16 视图换算比例
    /// @details 仅影响 get_arm_pos/get_gimbal 的 int16 返回值，
    ///          内部 float 缓存始终按原始编码存储。
    /// @{
    float pos_scale_ = 0.01f;    ///< 位置缩放（如 0.01 表示厘米单位）
    float angle_scale_ = 0.01f;  ///< 角度缩放（如 0.01 表示百分之一度）
    /// @}

    // ---- 发送互斥锁（防止多线程同时写串口）----------------------
    std::mutex send_mutex_;  ///< 保证 send_* 系列函数原子写入，避免字节交叉

    // ---- 电磁阀命令缓存（仅用于 V,ID 无状态参数时的翻转）---------
    /** @name 电磁阀命令态缓存
     * @details 当文本命令 "V,id" 未提供 ON/OFF 参数时，
     *          读取当前缓存态取反实现"一键翻转"。
     *          显式指定状态时直接覆盖缓存。
     */
    mutable std::mutex valve_cmd_mutex_;              ///< 电磁阀缓存保护锁
    std::array<bool, 4> valve_cmd_state_ = {false, false, false, false};  ///< 4 路电磁阀当前命令态

    // ---- 自动重连配置（由 open_by_HWid / configure_auto_reconnect 设置）----
    /** @name 自动重连状态
     * @details 当 auto_reconnect_enabled_ 为 true 时：
     *          - receive_once() 周期性用 libusb 检测 USB 设备存在性
     *          - write_bytes() 写入失败时立即触发 reconnect_once()
     *          - try_reconnect_once() 供 ROS 主循环非阻塞轮询
     */
    std::string reconnect_hw_id_;                     ///< 目标 USB 硬件 ID（"VID:PID"）
    int reconnect_baud_rate_ = 115200;                ///< 目标波特率
    int reconnect_retry_ms_ = 1000;                   ///< 重连尝试间隔（毫秒）
    std::atomic<bool> auto_reconnect_enabled_{false}; ///< 自动重连总开关

    /// @brief libusb 掉线检测相关
    /// @details 为避免过度调用 libusb 影响性能，
    ///          仅在距上次检查超过 usb_check_interval_ms_ 时才执行 USB 总线扫描。
    static constexpr int usb_check_interval_ms_ = 500;   ///< USB 状态轮询间隔（毫秒）
    std::chrono::steady_clock::time_point last_usb_check_tp_ = {};      ///< 上次 libusb 检查时间点
    std::chrono::steady_clock::time_point last_reconnect_attempt_tp_ = {}; ///< 上次重连尝试时间点
    std::mutex reconnect_mutex_;
};
