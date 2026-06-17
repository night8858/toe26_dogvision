# dogvision_arm

ROS2 Jazzy 机械臂控制包，支持 **AA（平面4臂）** 和 **BB（4DOF双臂）** 两种协议。

## 架构概览

本包采用三层职责分离架构：

```
┌─────────────────────────────────────────────────────────────────┐
│ arm_cmd_terminal_node        arm_mission_node        arm_internation_node │
│ (用户终端 stdin)       ──▶   (任务拆解)        ──▶       (串口收发)      │
│                              │                              │              │
│ 发布: /arm/mission_cmd       │ 订阅: /arm/mission_cmd       │ 订阅: /arm/internation/cmd │
│       /arm/internation/cmd   │ 发布: /arm/internation/cmd   │ 发布: /arm/internation/data │
│                              │       /arm/mission_cmd(反馈)  │                              │
└─────────────────────────────────────────────────────────────────┘
```

| 层 | 节点 | 职责 |
|----|------|------|
| 人机交互 | `arm_cmd_terminal_node` | stdin 输入，按 `$` 前缀路由到高层或低层话题 |
| 任务编排 | `arm_mission_node` | 理解"收起/吸取/放置"等高层语义，拆解为低层指令序列 |
| 串口通信 | `arm_internation_node` | 协议帧打包/解析、串口连接/断线自动重连、状态发布 |

核心库 `dogvision_arm_lib`（`arm_internation` 类）封装了串口连接管理、协议帧编解码、自动重连状态机、CRC-8 校验等底层细节。

---

## 快速启动

### 标准启动（默认使用 4DOF 双臂协议）

```bash
ros2 launch dogvision_arm arm_control.launch
```

### 使用旧 AA 平面机械臂协议

```bash
ros2 launch dogvision_arm arm_control.launch protocol:=aa
```

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

#### 发布话题

| 话题 | 类型 | QoS | 频率 | 说明 |
|------|------|-----|------|------|
| `/arm_internation/data` | `std_msgs/String` | 20 | 20Hz (50ms) | 机械臂实时状态（协议相关格式见下方） |

#### 参数

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `hw_id` | `"0483:5740"` | string | USB 硬件 ID (VID:PID)，自动扫描匹配 |
| `baud_rate` | `115200` | int | 串口波特率 |
| `port` | `""` | string | 串口设备路径（如 `/dev/ttyUSB0`），留空则按 `hw_id` 自动查找 |
| `protocol` | `"aa"` | string | 协议：`"aa"`/`"plane"` 或 `"bb"`/`"4dof"`/`"dof4"` |
| `pos_scale` | `0.01` | double | 位置解码缩放因子（仅影响 `get_arm_pos()` int16 视图，默认 1cm） |
| `angle_scale` | `0.01` | double | 角度解码缩放因子（仅影响 `get_gimbal()` int16 视图） |
| `cmd_topic` | `"/arm_internation/cmd"` | string | 命令订阅话题名 |
| `data_topic` | `"/arm_internation/data"` | string | 状态发布话题名 |

#### 自动重连机制

- 支持通过 `hw_id` 自动扫描 `/dev/ttyACM*` 和 `/dev/ttyUSB*` 并匹配 USB VID:PID
- 使用 **libusb** 辅助掉线检测（亚秒级感知 USB 拔出），不依赖串口驱动超时
- 断线后自动清空上报缓存（避免读出陈旧数据），按 1 秒间隔重试
- 支持指定 `port` 直连（跳过 HWID 扫描）

---

### 2. `arm_mission_node` — 任务编排节点

**职责**：接收高层语义命令，拆解为多步低层串口指令序列，每条命令完成后发布 `FEEDBACK:DONE`。

#### 订阅话题

| 话题 | 类型 | QoS | 说明 |
|------|------|-----|------|
| `/arm/mission_cmd` | `std_msgs/String` | 10 | 高层任务命令 |

#### 发布话题

| 话题 | 类型 | QoS | 说明 |
|------|------|-----|------|
| `/arm/internation/cmd` | `std_msgs/String` | 10 | 拆解后的低层指令序列 |
| `/arm/mission_cmd` | `std_msgs/String` | 10 | 任务完成反馈 `FEEDBACK:DONE` |

#### 参数

| 参数 | 默认值 | 类型 | 说明 |
|------|--------|------|------|
| `mission_topic` | `"/arm/mission_cmd"` | string | 任务命令订阅话题 |
| `cmd_topic` | `"/arm/internation/cmd"` | string | 低层命令发布话题 |
| `start_pos.*` | 见 YAML | double | 各臂启动位置 (x, y) |
| `stow_pos.*` | 见 YAML | double | 各臂收起位置 (x, y) |
| `pick_pos.*` | 见 YAML | double | 各臂吸取位置 (x, y) |
| `place_pos.*` | 见 YAML | double | 各臂放置位置 (x, y) |

位置参数通过 `pos_set.yaml` 配置，支持 `LF`/`RF`/`LB`/`RB` 四个臂别名。

---

### 3. `arm_cmd_terminal_node` — 终端命令节点

**职责**：提供 stdin 终端交互，根据输入前缀路由到不同话题。

#### 路由规则

| 输入前缀 | 发布话题 | 示例 |
|----------|----------|------|
| 无前缀 | `/arm/mission_cmd` | `STOW,ALL` |
| `$` 开头 | `/arm/internation/cmd` | `$LF,X:10,Y:20` |

#### 发布话题

| 话题 | 类型 | QoS | 说明 |
|------|------|-----|------|
| `/arm/mission_cmd` | `std_msgs/String` | 10 | 高层任务命令（无 $ 前缀） |
| `/arm/internation/cmd` | `std_msgs/String` | 10 | 低层协议命令（带 $ 前缀） |

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

### 低层协议命令（发布到 `/arm/internation/cmd`）

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

**4DOF 臂别名**：`L`/`LEFT`/`左`/`0`(左臂), `R`/`RIGHT`/`右`/`1`(右臂)

#### 通用命令（AA 和 BB 均支持）

| 命令格式 | 说明 | 示例 |
|----------|------|------|
| `V,<id>,<state>` | 电磁阀控制 | `V,1,ON` / `V,1,OFF` |
| `V,<id>` | 翻转电磁阀状态 | `V,1` |
| `P,ON,<speed>` | 开泵并设速度 | `P,ON,2500` |
| `P,OFF` | 关泵 | `P,OFF` |
| `A,<answer>` | 任务赛答案 (0-255) | `A,0` |

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

节点以 20Hz 向 `/arm_internation/data` 发布状态字符串，格式按当前协议不同。

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
# 应看到: /arm/internation/cmd, /arm/mission_cmd, /arm_internation/data
```

### 4. 监听状态数据

```bash
ros2 topic echo /arm_internation/data
```

### 5. 手动发布低层命令（无需终端节点）

```bash
# AA 协议：控制 LF 臂到 (10, 20)
ros2 topic pub /arm/internation/cmd std_msgs/String "data: 'LF,X:10,Y:20'" --once

# BB 4DOF 协议：左臂位姿
ros2 topic pub /arm/internation/cmd std_msgs/String "data: '4POSE,L,X:0.1,Y:0.2,Z:0.3,PITCH:0.4'" --once

# 电磁阀打开
ros2 topic pub /arm/internation/cmd std_msgs/String "data: 'V,1,ON'" --once
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
colcon build --packages-select dogvision_arm
source install/setup.bash
```

依赖项：`rclcpp`, `std_msgs`, `libusb-1.0`（用于 USB 掉线检测）。
