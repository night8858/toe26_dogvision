# dogvision_arm

ROS2 Jazzy 机械臂控制包，支持 **AA（平面4臂）** 和 **BB（4DOF双臂）** 两种协议。协议通过 CMake 宏在编译时锁定，一次构建只支持其中一种。

## 架构概览

本包采用三层职责分离架构：

```
┌─────────────────────────────────────────────────────────────────┐
│ arm_cmd_terminal_node        arm_mission_node        arm_internation_node │
│ (用户终端 stdin)       ──▶   (任务拆解)        ──▶       (串口收发)      │
│                              │                              │              │
│ 发布: /arm/mission_cmd       │ 订阅: /arm/mission_cmd       │ 订阅: /arm_internation/cmd │
│       /arm_internation/cmd   │ 发布: /arm_internation/cmd   │ 发布: /arm_internation/data │
│                              │       /arm/mission_cmd(反馈)  │       /arm_internation/state│
└─────────────────────────────────────────────────────────────────┘
```

| 层 | 节点 | 职责 |
|----|------|------|
| 人机交互 | `arm_cmd_terminal_node` | stdin 输入，按 `$` 前缀路由到高层或低层话题 |
| 任务编排 | `arm_mission_node` | 理解"收起/吸取/放置"等高层语义，拆解为低层指令序列 |
| 串口通信 | `arm_internation_node` | 协议帧打包/解析、串口连接/断线自动重连、状态发布 |

核心库 `dogvision_arm_lib`（`arm_internation` 类）封装了串口连接管理、协议帧编解码、自动重连状态机、CRC-8 校验等底层细节。

---

## 输入接口总览

机械臂包当前没有 ROS Service 或 Action，所有控制输入来自 **终端 stdin、ROS 话题、启动参数、STM32 串口反馈** 或底层 C++ API。

| 输入接口 | 类型/格式 | 接收方 | 控制用途 | 协议 |
|---|---|---|---|---|
| 终端标准输入 | 文本行 | `arm_cmd_terminal_node` | 无 `$` 发布高层任务；有 `$` 发布低层命令 | 取决于目标话题 |
| `/arm/mission_cmd` | `std_msgs/msg/String` | `arm_mission_node` | 收起、吸取、启动、放置、阀和泵等任务编排 | 位置任务仅 AA；阀/泵 AA、BB 通用 |
| `/arm_internation/cmd` | `std_msgs/msg/String` | `arm_internation_node` | 直接控制协议帧：AA 四臂/云台，BB 双臂动作，阀、泵、答案 | AA 或 BB |
| `/ocr/answer` | `std_msgs/msg/UInt8` | `arm_internation_node` | 将 OCR 稳定答案 0-3 封装为 BB 05 发送 STM32 | 仅 BB/4DOF |
| STM32 串口 | AA/BB 二进制帧 | `arm_internation_node` | 输入位姿、传感器状态和动作完成事件 | AA 01、BB 01、BB CC |
| ROS 参数/YAML | 节点参数 | 三个节点 | 配置串口、协议一致性校验、话题和 AA 预设位置 | 启动时生效 |
| `arm_internation` C++ API | 函数调用 | 直接使用核心库的程序 | 绕过文本话题直接发送协议命令 | AA 或 BB |

### 节点分配与数据流

```text
stdin
  │
  ▼
arm_cmd_terminal_node
  ├─ 普通文本 ──▶ /arm/mission_cmd ──▶ arm_mission_node
  │                                      └─▶ /arm_internation/cmd
  └─ $低层命令 ─────────────────────────────▶ arm_internation_node ──▶ STM32

/ocr/answer ─────────────────────────────────▶ arm_internation_node ──▶ STM32
STM32 AA01/BB01/BBCC ─────────────────────────▶ arm_internation_node
                                                ├─▶ /arm_internation/data
                                                └─▶ /arm_internation/state
```

### 协议兼容性

| 功能 | AA 平面4臂 | BB 4DOF双臂 |
|---|---:|---:|
| `/arm/mission_cmd` 的 `STOW/PICK/START/PLACE` | 支持 | 不支持，生成的 LF/RF/LB/RB 命令会被 BB 模式拒绝 |
| `/arm/mission_cmd` 的 `VALVE/PUMP/PLACE_END` | 支持 | 支持 |
| 低层 `LF/RF/LB/RB`、`G` | 支持 | 不支持 |
| 低层 `4POSE/4ACT/4PICK/.../START,x,y,z` | 不支持 | 支持 |
| 低层 `V/P/A` | 支持 | 支持 |
| `/ocr/answer` 自动答案 | 不支持 | 支持，且仅接受 0-3 |

> 协议不能通过 launch 动态切换。默认构建为 BB/4DOF；需要 AA 平面臂时，必须以 `DOGVISION_ARM_USE_4DOF=OFF` 重新编译。`protocol` 参数仅校验启动配置是否与编译结果一致。

---

## 快速启动

### 标准启动

```bash
ros2 launch dogvision_arm arm_control.launch
```

### 编译协议选择

```bash
# BB/4DOF（默认）
colcon build --packages-select dogvision_arm \
  --cmake-args -DDOGVISION_ARM_USE_4DOF=ON

# AA 平面机械臂
colcon build --packages-select dogvision_arm \
  --cmake-args -DDOGVISION_ARM_USE_4DOF=OFF
```

切换协议时应清理该包的 CMake 缓存，或为两种配置使用不同的 `build`、`install` 目录。

### 启动完整调试模式（含终端节点）

```bash
ros2 launch dogvision_arm arm_test.launch
```

`arm_test.launch` 相比 `arm_control.launch` 额外启动了 `arm_cmd_terminal_node`，适合开发和调试。

### 单独运行终端命令节点

```bash
ros2 run dogvision_arm arm_cmd_terminal_node
```

在终端输入 `help` 查看支持的命令。

---

## 三节点详解

### 1. `arm_internation_node` — 串口通信节点

**职责**：串口连接管理、协议帧收发、断线自动重连、状态发布。

#### 订阅话题

| 话题 | 类型 | QoS | 说明 |
|------|------|-----|------|
| `/arm_internation/cmd` | `std_msgs/String` | 20 | 低层协议命令（见下方命令参考） |
| `/ocr/answer` | `std_msgs/UInt8` | reliable/volatile | OCR 稳定答案 `mod4`（0-3），4DOF 模式下转发为 BB 05 |

#### 发布话题

| 话题 | 类型 | QoS | 频率 | 说明 |
|------|------|-----|------|------|
| `/arm_internation/data` | `std_msgs/String` | 20 | 20Hz (50ms) | 机械臂实时状态（协议相关格式见下方） |
| `/arm_internation/state` | `std_msgs/String` | 20 | 事件触发 | 下位机动作完成事件，收到 `BB CC FF EE CRC8` 后发布 `DONE` |

#### 参数

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `hw_id` | `"0483:5740"` | string | USB 硬件 ID (VID:PID)，自动扫描匹配 |
| `baud_rate` | `115200` | int | 串口波特率 |
| `port` | `""` | string | 串口设备路径（如 `/dev/ttyUSB0`），留空则按 `hw_id` 自动查找 |
| `protocol` | `"compiled"` | string | 编译协议校验；`compiled` 自动匹配，也可填写与编译结果一致的 AA/BB 别名 |
| `pos_scale` | `0.01` | double | 位置解码缩放因子（仅影响 `get_arm_pos()` int16 视图，默认 1cm） |
| `angle_scale` | `0.01` | double | 角度解码缩放因子（仅影响 `get_gimbal()` int16 视图） |
| `cmd_topic` | `"/arm_internation/cmd"` | string | 命令订阅话题名 |
| `data_topic` | `"/arm_internation/data"` | string | 状态发布话题名 |
| `state_topic` | `"/arm_internation/state"` | string | 一次性事件发布话题名 |
| `ocr_answer_topic` | `"/ocr/answer"` | string | OCR 稳定答案订阅话题名 |

#### 自动重连机制

- 支持通过 `hw_id` 自动扫描 `/dev/ttyACM*` 和 `/dev/ttyUSB*` 并匹配 USB VID:PID
- 使用 **libusb** 辅助掉线检测（亚秒级感知 USB 拔出），不依赖串口驱动超时
- 断线后自动清空上报缓存（避免读出陈旧数据），按 1 秒间隔重试
- 支持指定 `port` 直连（跳过 HWID 扫描）

---

### 2. `arm_mission_node` — 任务编排节点

**职责**：接收高层语义命令，拆解为多步低层串口指令序列，每条命令完成后发布 `FEEDBACK:DONE`。

> 该节点的 `STOW/PICK/START/PLACE` 使用 `LF/RF/LB/RB,X,Y` 格式，属于 AA 平面4臂控制。BB/4DOF 模式下这些运动命令会被 `arm_internation_node` 拒绝。`VALVE`、`PUMP` 和 `PLACE_END` 产生的阀/泵命令可在两种协议下使用。

#### 订阅话题

| 话题 | 类型 | QoS | 说明 |
|------|------|-----|------|
| `/arm/mission_cmd` | `std_msgs/String` | 10 | 高层任务命令 |

#### 发布话题

| 话题 | 类型 | QoS | 说明 |
|------|------|-----|------|
| `/arm_internation/cmd` | `std_msgs/String` | 10 | 拆解后的低层指令序列 |
| `/arm/mission_cmd` | `std_msgs/String` | 10 | 任务完成反馈 `FEEDBACK:DONE` |

#### 参数

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `mission_topic` | `"/arm/mission_cmd"` | string | 任务命令订阅话题 |
| `cmd_topic` | `"/arm_internation/cmd"` | string | 低层命令发布话题 |
| `start_pos.*` | 见 YAML | double | 各臂启动位置 (x, y) |
| `stow_pos.*` | 见 YAML | double | 各臂收起位置 (x, y) |
| `pick_pos.*` | 见 YAML | double | 各臂吸取位置 (x, y) |
| `place_pos.*` | 见 YAML | double | 各臂放置位置 (x, y) |

位置参数通过 `pos_set.yaml` 配置，支持 `LF`/`RF`/`LB`/`RB` 四个臂别名。

> 当前 `mission_topic` 只改变任务命令的订阅话题；`FEEDBACK:DONE` 仍固定发布到 `/arm/mission_cmd`。若重映射任务入口，反馈监听端仍需订阅默认话题。

---

### 3. `arm_cmd_terminal_node` — 终端命令节点

**职责**：提供 stdin 终端交互，根据输入前缀路由到不同话题。

#### 路由规则

| 输入前缀 | 发布话题 | 示例 |
|----------|----------|------|
| 无前缀 | `/arm/mission_cmd` | `STOW,ALL` |
| `$` 开头 | `/arm_internation/cmd` | `$LF,X:10,Y:20` |

#### 发布话题

| 话题 | 类型 | QoS | 说明 |
|------|------|-----|------|
| `/arm/mission_cmd` | `std_msgs/String` | 10 | 高层任务命令（无 $ 前缀） |
| `/arm_internation/cmd` | `std_msgs/String` | 10 | 低层协议命令（带 $ 前缀） |

#### 参数

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `cmd_topic` | `"/arm_internation/cmd"` | string | 低层命令发布话题 |
| `mission_topic` | `"/arm/mission_cmd"` | string | 高层任务发布话题 |

#### 内置命令

在终端输入：
- `help` / `h` — 显示帮助信息
- `quit` / `exit` — 退出终端节点

---

## 命令参考

### 低层协议命令（发布到 `/arm_internation/cmd`）

这些命令由 `arm_internation` 类中的 `handle_text_command()` 解析，直接打包为协议帧通过串口发送。

#### AA 平面协议命令

| 命令格式 | 说明 | 示例 |
|----------|------|------|
| `<alias>,X:<x>,Y:<y>` | 控制机械臂末端坐标 | `LF,X:10,Y:20` |
| `<alias>,<x>,<y>` | 简写格式 | `RF,10,20` |
| `G,<yaw>,<pitch>` | 控制云台角度 | `G,0,0` |

**机械臂别名**：`LF`/`FL`(0), `RF`/`FR`(1), `LB`/`BL`(2), `RB`/`BR`(3)

#### BB 4DOF 协议命令

| 命令格式 | 说明 | 示例 |
|----------|------|------|
| `4POSE,<arm>,X:<x>,Y:<y>,Z:<z>,PITCH:<pitch>` | 控制 4DOF 臂位姿（带前缀） | `4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4` |
| `4POSE,<arm>,<x>,<y>,<z>,<pitch>` | 简写格式 | `4POSE,R,0.1,0.2,0.3,0.4` |
| `4ACT,<id>` | 触发预设动作（0=中止, 1-N=动作） | `4ACT,1` |
| `4PICK,  <arm>,<x>,<y>,<z>`   | 发送 `BB 11`，单臂按 PC 目标点取块，xyz 单位 m | `4PICK,L,0.45,0.42,-0.21` |
| `4PLACE,<arm>,<x>,<y>,<z>`    | 发送 `BB 12`，单臂放块，xyz 单位 m | `4PLACE1,R,0.45,-0.40,-0.21` |
| `4PUTBACK,<arm>`              | 发送 `BB 14`，单臂放块到背部固定动作 | `4PUTBACK,L` |
| `4GETBACK,<arm>`              | 发送 `BB 15`，单臂从背部取块固定动作 | `4GETBACK,R` |
| `4PICKALL,<lx>,<ly>,<lz>,<rx>,<ry>,<rz>` | 发送 `BB 21`，双臂按 PC 目标点取块，xyz 单位 m | `4PICKALL,0.45,0.42,-0.21,0.45,-0.42,-0.21` |
| `4PUTBACKALL` | 发送 `BB 22`，双臂放块到背部固定动作 | `4PUTBACKALL` |
| `START,<x>,<y>,<z>` | 发送 `BB 99` 带初始偏移启动，偏移单位 mm | `START,0,0,0` |
| `START,X:<x>,Y:<y>,Z:<z>` | 带前缀写法，等价于上方简写 | `START,X:0,Y:0,Z:0` |

**4DOF 臂别名**：`L`/`LEFT`/`左`/`0`(左臂), `R`/`RIGHT`/`右`/`1`(右臂)

#### 通用命令（AA 和 BB 均支持）

| 命令格式 | 说明 | 示例 |
|----------|------|------|
| `V,<id>,<state>` | 电磁阀控制 | `V,1,ON` / `V,1,OFF` |
| `V,<id>` | 翻转电磁阀状态 | `V,1` |
| `P,ON,<speed>` | 开泵并设速度 | `P,ON,2500` |
| `P,OFF` | 关泵 | `P,OFF` |
| `A,<answer>` | 手动发送任务赛答案 (0-255)；BB 模式封装为 BB 05 | `A,0` |

**电磁阀 state**：`ON`/`1`/`OPEN`/`TRUE` 表示开，`OFF`/`0`/`CLOSE`/`FALSE` 表示关。

---

### 高层任务命令（发布到 `/arm/mission_cmd`）

这些命令由 `arm_mission_node` 接收，拆解为多步低层指令并等待间隔（默认 0.2s/步）。

| 命令 | 别名 | 参数说明 | 示例 |
|------|------|----------|------|
| `STOW` | `收起` | `ALL`(所有臂), 或臂别名 | `STOW,ALL` / `STOW,LF` |
| `PICK` | `吸取` | `ALL`(所有臂), 或臂别名 | `PICK,RF` / `PICK,ALL` |
| `START` | `启动` | `ALL`(所有臂), 或臂别名 | `START,ALL` / `START,LB` |
| `PLACE` | `放置` | `ALL`/臂别名, 或 `id,X,Y` | `PLACE,ALL` / `PLACE,0,100,200` |
| `VALVE` / `V` | `电磁阀` | `id,ON/OFF` 或 `ALL,ON/OFF` | `VALVE,1,ON` / `V,ALL,OFF` |
| `PUMP` / `P` | `气泵` | `ON[,speed]` 或 `OFF` | `PUMP,ON,2500` / `P,OFF` |
| `PLACE_END` / `PLACEEND` | `放置结束` | 无额外参数 | `PLACE_END` |

**执行顺序示例**（完整流程）：
```
START,ALL         → 各臂到启动位置
PICK,RF           → 右前臂到吸取位置
VALVE,1,ON        → 打开电磁阀 1 吸取物块
PLACE,ALL         → 各臂到放置位置
VALVE,1,OFF       → 关闭电磁阀 1 放下物块
STOW,ALL          → 各臂收起
PLACE_END         → 关闭所有电磁阀 + 气泵
```

每条高层命令执行完毕后，节点会向 `/arm/mission_cmd` 发布 `FEEDBACK:DONE`。

### 各输入接口的直接使用方法

#### 1. 高层任务话题

适合 AA 平面4臂的预设位置任务，或两种协议通用的阀/泵任务：

```bash
# AA：四臂移动到收起位置
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String \
  "{data: 'STOW,ALL'}"

# AA：RF 移动到吸取位置
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String \
  "{data: 'PICK,RF'}"

# AA/BB：打开全部电磁阀
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String \
  "{data: 'VALVE,ALL,ON'}"

# AA/BB：开启气泵
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String \
  "{data: 'PUMP,ON,2500'}"
```

任务执行完后会在同一话题发布 `FEEDBACK:DONE`。订阅该话题的其他节点应区分任务命令和反馈字符串。

#### 2. 低层命令话题

直接进入 `arm_internation::handle_text_command()`，不会经过任务拆解：

```bash
# AA：LF 末端坐标
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String \
  "{data: 'LF,X:10,Y:20'}"

# BB：左臂位姿
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String \
  "{data: '4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4'}"

# BB：左臂按目标点取块，xyz 单位 m
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String \
  "{data: '4PICK,L,0.45,0.42,-0.21'}"

# AA/BB：打开 1 号电磁阀
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String \
  "{data: 'V,1,ON'}"
```

命令解析失败、协议不匹配或串口发送失败时，`arm_internation_node` 会输出 `invalid cmd` 警告。

#### 3. OCR 自动答案话题

```bash
ros2 topic pub --once /ocr/answer std_msgs/msg/UInt8 "{data: 2}"
```

仅在以下条件全部满足时发送 BB 05：

- 当前包以 `DOGVISION_ARM_USE_4DOF=ON` 编译；
- 串口当前已连接；
- 数值范围为 0-3；
- 话题采用 reliable + volatile QoS，不重放旧答案。

手动发送 0-255 的答案字段也可使用低层命令：

```bash
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String \
  "{data: 'A,2'}"
```

`A,<answer>` 在 AA 和 BB 下均可发送；`/ocr/answer` 则额外限制为 BB 协议和 0-3。

#### 4. 终端输入

```bash
ros2 launch dogvision_arm arm_test.launch
```

启动后直接在终端输入：

```text
PUMP,ON,2500
$4POSE,L,0.1,0.2,0.3,0.4
$V,1,ON
help
```

- 普通文本发送到 `/arm/mission_cmd`；
- `$` 前缀会被去除后发送到 `/arm_internation/cmd`；
- `help`/`h` 显示帮助；
- `quit`/`exit` 关闭终端节点。

#### 5. STM32 串口输入

该输入由硬件自动产生，不通过 `ros2 topic pub` 控制：

| 串口帧 | 节点处理 | ROS 输出 |
|---|---|---|
| `AA 01 ... CRC8` | 更新四个平面臂、云台、阀和微动状态 | `/arm_internation/data` |
| `BB 01 ... CRC8` | 更新左右 4DOF 位姿、阀和微动状态 | `/arm_internation/data` |
| `BB CC FF EE CRC8` | 累计一次动作完成事件 | `/arm_internation/state` 发布 `DONE` |

#### 6. 启动参数输入

Launch 文件直接暴露：

```bash
ros2 launch dogvision_arm arm_control.launch \
  port:=/dev/ttyACM0 \
  baud_rate:=115200 \
  ocr_answer_topic:=/ocr/answer
```

`protocol` 参数默认是 `compiled`。显式传入 `aa` 或 `4dof` 只进行一致性校验；若与编译协议不一致，节点会在连接串口前以 FATAL 错误退出。

`arm_internation_node` 的 `cmd_topic`、`data_topic`、`state_topic` 等参数未作为当前 launch 参数暴露；单独运行节点时可覆盖：

```bash
ros2 run dogvision_arm arm_internation_node --ros-args \
  -p cmd_topic:=/custom/arm_cmd \
  -p data_topic:=/custom/arm_data
```

`arm_mission_node` 的预设位置由 `pos_set.yaml` 输入，可通过：

```bash
ros2 launch dogvision_arm arm_control.launch \
  mission_config:=/absolute/path/to/pos_set.yaml
```

#### 7. C++ 核心库接口

其他 C++ 节点也可直接链接 `dogvision_arm_lib`，调用：

```cpp
arm_internation arm;
arm.set_protocol_from_string("compiled");
arm.open("/dev/ttyACM0", 115200);
arm.send_4dof_pose_cmd(0, 0.1F, 0.2F, 0.3F, 0.4F);
arm.send_valve_cmd(1, true);
arm.send_pump_cmd(true, 2500);
arm.close();
```

主要发送接口包括：

- AA：`send_arm_cmd()`、`send_gimbal_cmd()`；
- BB：`send_4dof_pose_cmd()`、`send_4dof_action_cmd()`、`send_4dof_start_cmd()`、各取放块接口；
- 通用：`send_valve_cmd()`、`send_answer_cmd()`、`send_pump_cmd()`；
- 统一文本入口：`handle_text_command()`。

---

## 位置配置

### `pos_set.yaml`

配置文件安装在 `<pkg-share>/config/pos_set.yaml`，通过 launch 文件的 `mission_config` 参数传入。

```yaml
arm_mission_node:
  ros__parameters:
    start_pos:                          # 启动/初始位置
      LF: {x: 160.0, y: 220.0}
      RF: {x: 160.0, y: -220.0}
      LB: {x: -160.0, y: 220.0}
      RB: {x: -160.0, y: -220.0}
    stow_pos:                           # 收起位置（待机）
      LF: {x: 220.0, y: 380.0}
      RF: {x: 220.0, y: -380.0}
      LB: {x: -220.0, y: 380.0}
      RB: {x: -220.0, y: -380.0}
    pick_pos:                           # 吸取位置（对准物块）
      LF: {x: 450.0, y: 450.0}
      RF: {x: 450.0, y: -450.0}
      LB: {x: -450.0, y: 450.0}
      RB: {x: -450.0, y: -450.0}
    place_pos:                          # 放置位置（目标区域）
      LF: {x: 90.0, y: 900.0}
      RF: {x: 90.0, y: -900.0}
      LB: {x: -90.0, y: 900.0}
      RB: {x: -90.0, y: -900.0}
```

4 组位置 × 4 个臂别名 (`LF`/`RF`/`LB`/`RB`)，每组 2 个坐标 (`x`, `y`)。修改后重新启动 launch 即可生效。

**launch 传入自定义配置**：
```bash
ros2 launch dogvision_arm arm_control.launch mission_config:=/path/to/my_config.yaml
```

---

## 状态数据格式

节点以 20Hz 向 `/arm_internation/data` 发布状态字符串，格式按编译时协议不同。

### AA 协议

```
LF:x,y;RF:x,y;LB:x,y;RB:x,y;YAW:yaw;PITCH:pitch;VALVE_BITS:n;MICRO_BITS:n
```

示例：
```
LF:1.234,2.345;RF:3.456,4.567;LB:5.678,6.789;RB:7.890,8.901;YAW:10.5;PITCH:20.3;VALVE_BITS:5;MICRO_BITS:0
```

### BB 4DOF 协议

```
MODE:4DOF;L4:x,y,z,pitch;R4:x,y,z,pitch;VALVE_BITS:n;MICRO_BITS:n
```

示例：
```
MODE:4DOF;L4:0.100,0.200,0.300,0.400;R4:0.150,0.250,0.350,0.450;VALVE_BITS:5;MICRO_BITS:0
```

**字段说明**：

| 字段 | 含义 |
|------|------|
| `VALVE_BITS` | 4 位电磁阀状态位图（bit0=阀0, ..., bit3=阀3），1=开 |
| `MICRO_BITS` | 4 位微动开关状态位图，1=触发 |

---

## 测试与调试

OCR 自动答案链路：

```text
/ocr/answer (UInt8 mod4) → arm_internation_node → BB 05 answer 00 00 FF EE CRC8
```

仅当协议为 `4dof`、串口已连接且答案为 0-3 时发送。话题采用 volatile durability，不会在节点重启后重放旧答案。

### 1. 启动完整调试模式

```bash
ros2 launch dogvision_arm arm_test.launch
# 等同于 arm_control.launch + arm_cmd_terminal_node
```

### 2. 查看节点列表

```bash
ros2 node list
# 应看到: /arm_internation_node, /arm_mission_node, /arm_cmd_terminal_node
```

### 3. 查看话题列表

```bash
ros2 topic list
# 应看到: /arm_internation/cmd, /arm/mission_cmd, /arm_internation/data,
#         /arm_internation/state, /ocr/answer
```

### BB 05 字节测试

```bash
ctest --test-dir build/dogvision_arm -R answer_frame_test --output-on-failure
```

```bash
ros2 topic pub --once /ocr/answer std_msgs/msg/UInt8 "{data: 2}"
```

测试使用伪终端校验答案 0-3 的 8 字节帧、帧尾及 CRC8-0x07。

### 4. 监听状态数据

```bash
ros2 topic echo /arm_internation/data

# 动作完成事件会在这里看到 DONE
ros2 topic echo /arm_internation/state
```

### 5. 手动发布低层命令（无需终端节点）

```bash
# AA 协议：控制 LF 臂到 (10, 20)
ros2 topic pub /arm_internation/cmd std_msgs/String "data: 'LF,X:10,Y:20'" --once

# BB 4DOF 协议：左臂位姿
ros2 topic pub /arm_internation/cmd std_msgs/String "data: '4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4'" --once

# BB 4DOF 协议：带初始偏移启动，xyz 单位 mm
ros2 topic pub /arm_internation/cmd std_msgs/String "data: 'START,0,0,0'" --once

# 电磁阀打开
ros2 topic pub /arm_internation/cmd std_msgs/String "data: 'V,1,ON'" --once
```

### 6. 手动发布高层任务命令

```bash
ros2 topic pub /arm/mission_cmd std_msgs/String "data: 'STOW,ALL'" --once
ros2 topic pub /arm/mission_cmd std_msgs/String "data: 'PICK,RF'" --once
```

### 7. 单独运行各节点

```bash
# 只运行串口通信节点（需要先有串口设备）
ros2 run dogvision_arm arm_internation_node

# 只运行任务编排节点（配合其他节点使用）
ros2 run dogvision_arm arm_mission_node

# 只运行终端命令节点
ros2 run dogvision_arm arm_cmd_terminal_node
```

### 8. 查看节点参数

```bash
ros2 param dump /arm_internation_node
ros2 param dump /arm_mission_node
ros2 param dump /arm_cmd_terminal_node
```

### 9. 指定串口直连（跳过 HWID 扫描）

```bash
# 启动时指定串口路径
ros2 launch dogvision_arm arm_control.launch port:=/dev/ttyUSB0
```

### 10. 修改波特率

```bash
ros2 launch dogvision_arm arm_control.launch baud_rate:=9600
```

---

## 构建

```bash
cd <workspace>

# 默认：BB/4DOF
colcon build --packages-select dogvision_arm \
  --cmake-args -DDOGVISION_ARM_USE_4DOF=ON

# 或：AA 平面机械臂
colcon build --packages-select dogvision_arm \
  --cmake-args -DDOGVISION_ARM_USE_4DOF=OFF

source install/setup.bash
```

`DOGVISION_ARM_USE_4DOF` 会生成公开头文件 `dogvision_arm/protocol_config.hpp`，其中宏值为 `1`（BB）或 `0`（AA）。切换选项后必须重新构建；建议使用独立构建目录，或删除该包旧的 CMake 缓存。

依赖项：`rclcpp`, `std_msgs`, `libusb-1.0`（用于 USB 掉线检测）。
