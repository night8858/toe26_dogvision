# dogvision_arm

ROS2 上位机机械臂通信包，支持 BB/4DOF 双臂协议与 CC 云台协议。上位机负责文本命令解析、串口打帧、状态发布和高层任务等待；STM32 执行动作完成后返回 `BB CC FF EE CRC8`，节点再发布完成反馈。

---

## 目录

- [系统架构](#系统架构)
- [BB 帧协议详解（下位机参考）](#bb-帧协议详解下位机参考)
  - [CRC8 算法](#crc8-算法)
  - [上行帧（STM32 → 上位机）](#上行帧stm32--上位机)
  - [下行帧（上位机 → STM32）](#下行帧上位机--stm32)
- [CC 云台协议](#cc-云台协议)
- [节点与话题](#节点与话题)
- [测试接口](#测试接口)
  - [终端命令测试](#终端命令测试)
  - [C++ 单元测试](#c-单元测试)
  - [手动发送话题测试](#手动发送话题测试)
- [启动与参数](#启动与参数)
- [下位机对接要点](#下位机对接要点)

---

## 系统架构

```mermaid
graph TD
    A[arm_cmd_terminal_node] -->|高层任务| B[/arm/mission_cmd]
    A -->|$低层命令| C[/arm_internation/cmd]
    B --> D[arm_mission_node]
    D -->|转发低层命令| C
    C --> E[arm_internation_node]
    E -->|串口 BB/CC 帧| F[STM32 下位机]
    F -->|BB 01 反馈 / BB CC 完成| E
    E -->|状态数据| G[/arm_internation/data]
    E -->|完成事件 DONE| H[/arm_internation/state]
    H -->|监听 DONE| D
    D -->|FEEDBACK:DONE/TIMEOUT| B
```

> **代码结构**：`arm_internation.cpp` 负责串口连接/CRC/重连等公共逻辑；`arm_internation_bb.cpp` 实现 BB/4DOF 双臂协议；`arm_internation_cc.cpp` 实现 CC 云台协议。文本命令由 `handle_text_command()` 统一入口按协议分派。

---

## BB 帧协议详解（下位机参考）

> **关键约定**：所有多字节数值均为 **小端字节序（Little-Endian）**，float 类型为 IEEE 754 单精度（4 字节），帧尾固定为 `FF EE`，最后一字节为 CRC8 校验值。

### CRC8 算法

- **多项式**：`0x07`（CRC-8/SMBus）
- **初始值**：`0x00`
- **不反转、不异或输出**
- **C 参考实现**：

```c
uint8_t calc_crc8(const uint8_t *data, size_t len) {
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++)
            crc = (crc & 0x80) ? (crc << 1) ^ 0x07 : (crc << 1);
    }
    return crc;
}
```

---

### 上行帧（STM32 → 上位机）

#### BB 01 — 周期位姿反馈（固定 46 字节）

STM32 应以约 100Hz 周期发送此帧，CRC8 覆盖字节 `[0]~[44]`。

| 字节偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | uint8 | 帧头 |
| 1 | 1 | `0x01` | uint8 | 命令字：反馈 |
| 2-5 | 4 | 左臂 X | float32 LE | 末端 X 坐标 |
| 6-9 | 4 | 左臂 Y | float32 LE | 末端 Y 坐标 |
| 10-13 | 4 | 左臂 Z | float32 LE | 末端 Z 坐标 |
| 14-17 | 4 | 左臂 Pitch | float32 LE | 末端俯仰角 |
| 18-21 | 4 | 右臂 X | float32 LE | 末端 X 坐标 |
| 22-25 | 4 | 右臂 Y | float32 LE | 末端 Y 坐标 |
| 26-29 | 4 | 右臂 Z | float32 LE | 末端 Z 坐标 |
| 30-33 | 4 | 右臂 Pitch | float32 LE | 末端俯仰角 |
| 34 | 1 | valve0 | uint8 | bit0=1 表示开 |
| 35 | 1 | valve1 | uint8 | bit0=1 表示开 |
| 36 | 1 | valve2 | uint8 | bit0=1 表示开 |
| 37 | 1 | valve3 | uint8 | bit0=1 表示开 |
| 38 | 1 | microswitch0 | uint8 | bit0=1 表示触发（预留） |
| 39 | 1 | microswitch1 | uint8 | bit0=1 表示触发（预留） |
| 40 | 1 | microswitch2 | uint8 | bit0=1 表示触发（预留） |
| 41 | 1 | microswitch3 | uint8 | bit0=1 表示触发（预留） |
| 42 | 1 | reserved | uint8 | 预留，填 0 |
| 43 | 1 | `0xFF` | uint8 | 帧尾 A |
| 44 | 1 | `0xEE` | uint8 | 帧尾 B |
| 45 | 1 | CRC8 | uint8 | 覆盖 [0]~[44] |

#### BB CC — 动作完成事件（固定 5 字节）

当 STM32 完成一个动作（PICK/PLACE/PUTBACK/GETBACK 及其变体）后，发送此帧。上位机收到后发布 `/arm_internation/state = "DONE"`。

| 字节偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | uint8 | 帧头 |
| 1 | 1 | `0xCC` | uint8 | 命令字：动作完成 |
| 2 | 1 | `0xFF` | uint8 | 帧尾 A |
| 3 | 1 | `0xEE` | uint8 | 帧尾 B |
| 4 | 1 | CRC8 | uint8 | 覆盖 [0]~[3] |

---

### 下行帧（上位机 → STM32）

> **通用帧结构**：`BB <CMD> [DATA...] FF EE CRC8`，CRC8 覆盖 CRC 字节之前的所有字节。

#### 命令字速查表

| 命令字 | 帧长 | 用途 | 可控末端 |
|:---:|:---:|------|:---:|
| `0x02` | 22 | 4DOF 位姿控制 | ✅ |
| `0x03` | 6 | 预设动作触发 | ❌ |
| `0x04` | 7 | 电磁阀控制 | — |
| `0x05` | 8 | 答案/语音 | — |
| `0x06` | 10 | 气泵控制 | — |
| `0x11` | 18 | 单臂取块 | ✅ |
| `0x12` | 18 | 单臂放块 | ✅ |
| `0x14` | 6 | 单臂放回背部 | ❌ |
| `0x15` | 6 | 单臂从背部取块 | ❌ |
| `0x21` | 29 | 双臂取块 | ✅ |
| `0x22` | 5 | 双臂放回背部 | ❌ |
| `0x23` | 29 | 双臂放块 | ✅ |
| `0x24` | 5 | 双臂从背部取块 | ❌ |
| `0x99` | 17 | 带偏移启动 | — |

---

#### BB 02 — 4DOF 单臂位姿控制（22 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | | |
| 1 | 1 | `0x02` | | |
| 2 | 1 | arm_id | uint8 | 0=左臂, 1=右臂 |
| 3-6 | 4 | X | float32 LE | |
| 7-10 | 4 | Y | float32 LE | |
| 11-14 | 4 | Z | float32 LE | |
| 15-18 | 4 | Pitch | float32 LE | |
| 19 | 1 | `0xFF` | | |
| 20 | 1 | `0xEE` | | |
| 21 | 1 | CRC8 | | 覆盖 [0]~[20] |

---

#### BB 03 — 预设动作触发（6 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | | |
| 1 | 1 | `0x03` | | |
| 2 | 1 | action_id | uint8 | 0=中止, 1..N=预设动作 |
| 3 | 1 | `0xFF` | | |
| 4 | 1 | `0xEE` | | |
| 5 | 1 | CRC8 | | 覆盖 [0]~[4] |

---

#### BB 04 — 电磁阀控制（7 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | | |
| 1 | 1 | `0x04` | | |
| 2 | 1 | valve_id | uint8 | 0~3 |
| 3 | 1 | state | uint8 | 0=关, 1=开 |
| 4 | 1 | `0xFF` | | |
| 5 | 1 | `0xEE` | | |
| 6 | 1 | CRC8 | | 覆盖 [0]~[5] |

---

#### BB 05 — 答案/语音控制（8 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | | |
| 1 | 1 | `0x05` | | |
| 2 | 1 | answer | uint8 | 0~255 |
| 3 | 1 | 0x00 | | 预留 |
| 4 | 1 | 0x00 | | 预留 |
| 5 | 1 | `0xFF` | | |
| 6 | 1 | `0xEE` | | |
| 7 | 1 | CRC8 | | 覆盖 [0]~[6] |

---

#### BB 06 — 气泵控制（10 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | | |
| 1 | 1 | `0x06` | | |
| 2 | 1 | on_off | uint8 | 0=关泵, 1=开泵 |
| 3-6 | 4 | speed | float32 LE | 泵速 |
| 7 | 1 | `0xFF` | | |
| 8 | 1 | `0xEE` | | |
| 9 | 1 | CRC8 | | 覆盖 [0]~[8] |

---

#### BB 11 — 单臂取块（18 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | | |
| 1 | 1 | `0x11` | | |
| 2 | 1 | arm_id | uint8 | 0=左臂, 1=右臂 |
| 3-6 | 4 | X | float32 LE | **单位：米（m）** |
| 7-10 | 4 | Y | float32 LE | **单位：米（m）** |
| 11-14 | 4 | Z | float32 LE | **单位：米（m）** |
| 15 | 1 | `0xFF` | | |
| 16 | 1 | `0xEE` | | |
| 17 | 1 | CRC8 | | 覆盖 [0]~[16] |

---

#### BB 12 — 单臂放块（18 字节）

格式同 BB 11，命令字为 `0x12`。XYZ 单位 **米**。

---

#### BB 14 — 单臂放回背部（6 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | | |
| 1 | 1 | `0x14` | | |
| 2 | 1 | arm_id | uint8 | 0=左臂, 1=右臂 |
| 3 | 1 | `0xFF` | | |
| 4 | 1 | `0xEE` | | |
| 5 | 1 | CRC8 | | 覆盖 [0]~[4] |

---

#### BB 15 — 单臂从背部取块（6 字节）

格式同 BB 14，命令字为 `0x15`。

---

#### BB 21 — 双臂取块（29 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | | |
| 1 | 1 | `0x21` | | |
| 2-5 | 4 | 左臂 X | float32 LE | **单位：米（m）** |
| 6-9 | 4 | 左臂 Y | float32 LE | |
| 10-13 | 4 | 左臂 Z | float32 LE | |
| 14-17 | 4 | 右臂 X | float32 LE | |
| 18-21 | 4 | 右臂 Y | float32 LE | |
| 22-25 | 4 | 右臂 Z | float32 LE | |
| 26 | 1 | `0xFF` | | |
| 27 | 1 | `0xEE` | | |
| 28 | 1 | CRC8 | | 覆盖 [0]~[27] |

---

#### BB 22 — 双臂放回背部（5 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | | |
| 1 | 1 | `0x22` | | |
| 2 | 1 | `0xFF` | | |
| 3 | 1 | `0xEE` | | |
| 4 | 1 | CRC8 | | 覆盖 [0]~[3] |

---

#### BB 23 — 双臂放块（29 字节）

格式同 BB 21，命令字为 `0x23`。XYZ 单位 **米**。

---

#### BB 24 — 双臂从背部取块（5 字节）

格式同 BB 22，命令字为 `0x24`。

---

#### BB 99 — 带初始偏移启动（17 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xBB` | | |
| 1 | 1 | `0x99` | | |
| 2-5 | 4 | offsetX | float32 LE | **单位：毫米（mm）** |
| 6-9 | 4 | offsetY | float32 LE | **单位：毫米（mm）** |
| 10-13 | 4 | offsetZ | float32 LE | **单位：毫米（mm）** |
| 14 | 1 | `0xFF` | | |
| 15 | 1 | `0xEE` | | |
| 16 | 1 | CRC8 | | 覆盖 [0]~[15] |

> ⚠️ **注意单位差异**：BB 11/12/21/23 的 XYZ 为 **米**，BB 99 的偏移为 **毫米**。

---

## CC 云台协议

> **帧头**：`0xCC`，帧尾 `FF EE`，CRC8 同 BB 协议（多项式 `0x07`）。多字节数值均为 float32 小端。

### 下行帧（上位机 → STM32）

#### CC 99 — 启动相机云台（5 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xCC` | uint8 | 帧头 |
| 1 | 1 | `0x99` | uint8 | 命令字：启动云台 |
| 2 | 1 | `0xFF` | uint8 | 帧尾 A |
| 3 | 1 | `0xEE` | uint8 | 帧尾 B |
| 4 | 1 | CRC8 | uint8 | 覆盖 [0]~[3] |

#### CC 01 — 运动云台到目标位置（17 字节）

| 偏移 | 长度 | 字段 | 类型 | 说明 |
|:---:|:---:|------|------|------|
| 0 | 1 | `0xCC` | uint8 | 帧头 |
| 1 | 1 | `0x01` | uint8 | 命令字：运动到目标 |
| 2-5 | 4 | J1 | float32 LE | **单位：度** |
| 6-9 | 4 | PITCH | float32 LE | **单位：度** |
| 10-13 | 4 | YAW | float32 LE | **单位：度** |
| 14 | 1 | `0xFF` | uint8 | 帧尾 A |
| 15 | 1 | `0xEE` | uint8 | 帧尾 B |
| 16 | 1 | CRC8 | uint8 | 覆盖 [0]~[15] |

### 文本命令

| 命令 | 对应帧 | 说明 |
| --- | --- | --- |
| `CAM_START` / `GIMBAL_START` / `云台启动` | `CC 99 FF EE CRC8` | 启动相机云台 |
| `CAM_MOVE,j1,pitch,yaw` / `GIMBAL_MOVE,...` / `云台运动,...` | `CC 01 j1 pitch yaw FF EE CRC8` | 运动到目标角度 |

---

## 节点与话题

| 节点 | 作用 |
| --- | --- |
| `arm_internation_node` | 连接 STM32 串口，解析 BB/CC 帧，发布状态与完成事件 |
| `arm_mission_node` | 单任务等待模型：接收高层任务，转为低层命令，等待 DONE 后反馈 |
| `arm_cmd_terminal_node` | 终端输入路由。无前缀发高层任务，`$` 前缀直发低层命令 |

| 话题 | 类型 | 方向 | 说明 |
| --- | --- | --- | --- |
| `/arm_internation/cmd` | `std_msgs/msg/String` | 输入 | 低层文本命令 |
| `/arm_internation/data` | `std_msgs/msg/String` | 输出 | 格式 `MODE:4DOF;L4:x,y,z,pitch;R4:x,y,z,pitch;VALVE_BITS:n;MICRO_BITS:n` |
| `/arm_internation/state` | `std_msgs/msg/String` | 输出 | 收到 BB CC 时发布 `DONE` |
| `/arm/mission_cmd` | `std_msgs/msg/String` | 输入/反馈 | 高层任务入口；完成后同话题发布 `FEEDBACK:DONE`，超时发布 `FEEDBACK:TIMEOUT`，忙时 `FEEDBACK:BUSY` |
| `/ocr/answer` | `std_msgs/msg/UInt8` | 输入 | 发送 `BB 05` 答案字段 |

---

## 测试接口

### 终端命令测试

使用 `arm_cmd_terminal_node` 可在终端直接测试所有命令。启动方式：

```bash
ros2 launch dogvision_arm arm_test.launch
```

#### 高层任务命令（无前缀，发往 `/arm/mission_cmd`）

这些命令经过 `arm_mission_node` 的任务队列，单任务串行执行，完成后反馈 `FEEDBACK:DONE`。

| 命令 | 对应 BB 帧 | 说明 |
| --- | --- | --- |
| `PICK,ID,x,y,z` | `BB 11 arm_id x y z FF EE CRC8` | 单臂到目标取块 |
| `PLACE,ID,x,y,z` | `BB 12 arm_id x y z FF EE CRC8` | 单臂到目标放块 |
| `PUTBACK,ID` | `BB 14 arm_id FF EE CRC8` | 单臂放回背部 |
| `GETBACK,ID` | `BB 15 arm_id FF EE CRC8` | 单臂从背部取块 |
| `PICKALL,lx,ly,lz,rx,ry,rz` | `BB 21 Lx Ly Lz Rx Ry Rz FF EE CRC8` | 双臂取块 |
| `PLACEALL,lx,ly,lz,rx,ry,rz` | `BB 23 Lx Ly Lz Rx Ry Rz FF EE CRC8` | 双臂放块 |
| `PUTBACKALL` | `BB 22 FF EE CRC8` | 双臂放回背部 |
| `GETBACKALL` | `BB 24 FF EE CRC8` | 双臂从背部取块 |

`ID` 支持：`0` / `L` / `LEFT` / `左` 表示左臂；`1` / `R` / `RIGHT` / `右` 表示右臂。坐标单位为米。

`arm_mission_node` 是单任务等待模型：执行中收到新任务会拒绝并反馈 `FEEDBACK:BUSY`。默认超时 15 秒后若仍未收到 STM32 的 `BB CC` 完成帧，自动解除 busy 并反馈 `FEEDBACK:TIMEOUT`（日志标为 TIMEOUT）；若下一条新任务开始前迟到 `DONE`，会补发 `FEEDBACK:DONE`。

#### 低层调试命令（`$` 前缀，直发 `/arm_internation/cmd`）

这些命令直接打包为 BB 帧发送，不经过任务队列，适合调试和手动控制。

```bash
# === 位姿控制（BB 02）===
$4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4
$4POSE,R,0.1,0.2,0.3,0.4

# === 预设动作（BB 03）===
$4ACT,1          # 触发预设动作 1
$4ACT,0          # 中止当前动作

# === 单臂取放（BB 11/12）===
$PICK,L,0.45,0.42,-0.21
$PLACE,R,0.45,-0.40,-0.21

# === 单臂背部动作（BB 14/15）===
$PUTBACK,L
$GETBACK,R

# === 双臂取放（BB 21/23）===
$PICKALL,0.45,0.42,-0.21,0.45,-0.42,-0.21
$PLACEALL,0.45,0.42,-0.21,0.45,-0.42,-0.21

# === 双臂背部动作（BB 22/24）===
$PUTBACKALL
$GETBACKALL

# === 启动（BB 99，偏移单位 mm）===
$START,0,0,0

# === 电磁阀（BB 04）===
$V,1              # 翻转电磁阀 1
$V,1,ON           # 打开电磁阀 1
$V,ALL,ON         # 全部打开

# === 气泵（BB 06）===
$P,ON,2500        # 开泵，速度 2500
$P,OFF            # 关泵

# === 云台（CC 99/01）===
$CAM_START        # 启动相机云台
$CAM_MOVE,90,-30,45  # 运动到 J1=90° PITCH=-30° YAW=45°
```

---

### C++ 单元测试

```bash
colcon build --packages-select dogvision_arm
ctest --test-dir build/dogvision_arm --output-on-failure
```

测试文件及覆盖范围：

| 测试文件 | 覆盖内容 |
| --- | --- |
| `test/answer_frame_test.cpp` | 使用伪终端（pty）验证 BB 11/12/14/15/21/22/23/24 帧打包正确性，逐字节校验帧头、命令字、arm_id、坐标值、尾部和 CRC |
| `test/mission_command_test.cpp` | 验证 `ArmMissionController` 状态机：命令解析、busy 拒绝、DONE 完成流转、中文别名 |

---

### 手动发送话题测试

在不启动 `arm_cmd_terminal_node` 的情况下，可直接用 `ros2 topic pub` 向话题发送命令进行调试。

#### 高层任务（`/arm/mission_cmd`）

```bash
# 单臂取块
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PICK,L,0.45,0.42,-0.21'"

# 单臂放块
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PLACE,R,0.45,-0.40,-0.21'"

# 单臂放回背部
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PUTBACK,L'"

# 单臂从背部取块
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'GETBACK,R'"

# 双臂取块
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PICKALL,0.45,0.42,-0.21,0.45,-0.42,-0.21'"

# 双臂放块
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PLACEALL,0.45,0.42,-0.21,0.45,-0.42,-0.21'"

# 双臂放回背部
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PUTBACKALL'"

# 双臂从背部取块
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'GETBACKALL'"
```

> **注意**：`arm_mission_node` 是单任务模型，上一个任务未完成（未收到 BB CC）时发送新任务会收到 `FEEDBACK:BUSY`。

#### 低层命令（`/arm_internation/cmd`）

```bash
# 位姿控制（BB 02）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: '4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4'"

# 预设动作（BB 03）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: '4ACT,1'"

# 单臂取块（BB 11）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'PICK,L,0.45,0.42,-0.21'"

# 单臂放块（BB 12）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'PLACE,R,0.45,-0.40,-0.21'"

# 单臂背部动作（BB 14/15）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'PUTBACK,L'"
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'GETBACK,R'"

# 双臂取放（BB 21/23）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'PICKALL,0.45,0.42,-0.21,0.45,-0.42,-0.21'"
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'PLACEALL,0.45,0.42,-0.21,0.45,-0.42,-0.21'"

# 双臂背部动作（BB 22/24）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'PUTBACKALL'"
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'GETBACKALL'"

# 启动（BB 99，偏移单位 mm）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'START,0,0,0'"

# 电磁阀（BB 04）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'V,1,ON'"
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'V,ALL,OFF'"

# 气泵（BB 06）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'P,ON,2500'"
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'P,OFF'"

# 云台启动（CC 99）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'CAM_START'"

# 云台运动（CC 01）
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'CAM_MOVE,90,-30,45'"
```

#### 模拟 STM32 完成反馈

如果需要在没有真实 STM32 的情况下测试 `arm_mission_node` 的状态机流转，可以手动发布 DONE 事件：

```bash
# 模拟 STM32 动作完成，使 arm_mission_node 解除 busy
ros2 topic pub --once /arm_internation/state std_msgs/msg/String "data: 'DONE'"
```

#### 监听状态

```bash
# 监听机械臂实时位姿数据
ros2 topic echo /arm_internation/data

# 监听完成事件
ros2 topic echo /arm_internation/state

# 监听任务反馈
ros2 topic echo /arm/mission_cmd
```

---

## 启动与参数

```bash
colcon build --packages-select dogvision_arm
source install/setup.bash
ros2 launch dogvision_arm arm_control.launch
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `hw_id` | `0483:5740` | 自动连接的 USB VID:PID |
| `baud_rate` | `115200` | 串口波特率 |
| `port` | 空 | 指定串口路径；留空时按 `hw_id` 自动重连 |
| `pos_scale` | `0.01` | 兼容 int16 位置视图缩放 |
| `ocr_answer_topic` | `/ocr/answer` | OCR 答案输入话题 |
| `timeout_ms` | `15000` | 任务超时（ms），超时后自动解除 busy 并反馈 `FEEDBACK:TIMEOUT`；设 0 禁用 |

测试启动（带终端）：

```bash
ros2 launch dogvision_arm arm_test.launch
```

---

## 下位机对接要点

以下是对接 STM32 固件时需要关注的协议细节：

### 1. 帧同步策略

上位机使用 **滑动窗口 + 帧头搜索** 方式解析字节流：
- 在缓冲区中搜索 `0xBB` 帧头
- 检查第二字节确定帧类型和长度
- 验证 `FF EE` 尾部和 CRC8
- 坏帧丢弃当前 `0xBB`，继续搜索下一个

因此 STM32 发送时务必保证：
- 帧间不加额外填充字节
- 每帧独立完整，CRC 正确
- BB 01 周期建议 ≤100Hz，避免上位机缓冲区溢出（缓冲区 512 字节）

### 2. 单位关键差异

| 帧类型 | 坐标单位 |
| --- | --- |
| BB 02 (4POSE) | 未限定，由双方约定 |
| BB 11/12/21/23 (PICK/PLACE) | **米（m）** |
| BB 99 (START) | **毫米（mm）** |

下位机解析 BB 11/12/21/23 时，需将米转为内部单位（如毫米：`×1000`）。

### 3. arm_id 约定

全部命令中 `arm_id` 字段：**0 = 左臂，1 = 右臂**。

### 4. 完成反馈必须发送 BB CC

所有动作命令（BB 11/12/14/15/21/22/23/24）执行完成后，STM32 **必须** 发送 `BB CC FF EE CRC8`，否则上位机的 `arm_mission_node` 将永远处于 busy 状态，拒绝后续任务。

### 5. BB 99 启动时序

上位机期望的启动流程：
1. STM32 上电后开始周期性发送 BB 01 反馈帧
2. 上位机连接串口后发送 `BB 99 offsetX offsetY offsetZ FF EE CRC8`
3. STM32 收到 BB 99 后初始化机械臂并进入就绪状态
4. 后续正常收发动作命令

### 6. 电磁阀状态一致性

`V,id`（无状态参数）依赖上位机内部的翻转缓存，如果 STM32 端电磁阀状态因外部原因改变（如掉电复位），上位机缓存不会自动同步。建议测试时使用 `V,id,ON` / `V,id,OFF` 显式设置。

### 7. 串口参数

上位机固定配置为 **8N1 原始模式**（8 数据位、无校验、1 停止位、无流控），波特率默认 115200。STM32 端需保持一致。
