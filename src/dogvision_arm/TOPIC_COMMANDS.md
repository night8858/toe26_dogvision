# dogvision_arm 话题 data 速查

本文只列协作方需要发布到 ROS2 话题的 `data` 内容，以及需要监听的反馈格式。

## 1. 推荐使用方式

普通任务协作方优先向 `/arm/mission_cmd` 发布高层任务。

```bash
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: '<命令>'"
```

低层调试才直接向 `/arm_internation/cmd` 发布命令。

```bash
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: '<命令>'"
```

## 2. 高层任务 `/arm/mission_cmd`

类型：`std_msgs/msg/String`

作用：任务入口。`arm_mission_node` 会串行执行任务，等待 STM32 返回 `DONE` 或 `DIAG` 后反馈。

### 2.1 单臂任务

| 作用 | data |
| --- | --- |
| 左臂取块 | `PICK,L,x,y,z` |
| 右臂取块 | `PICK,R,x,y,z` |
| 左臂放块 | `PLACE,L,x,y,z` |
| 右臂放块 | `PLACE,R,x,y,z` |
| 左臂放回背部 | `PUTBACK,L` |
| 右臂放回背部 | `PUTBACK,R` |
| 左臂从背部取块 | `GETBACK,L` |
| 右臂从背部取块 | `GETBACK,R` |

示例：

```bash
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PICK,L,0.30,0.40,-0.21'"
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PLACE,R,0.30,-0.40,-0.21'"
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PUTBACK,L'"
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'GETBACK,R'"
```

### 2.2 双臂任务

| 作用 | data |
| --- | --- |
| 双臂取块 | `PICKALL,lx,ly,lz,rx,ry,rz` |
| 双臂放块 | `PLACEALL,lx,ly,lz,rx,ry,rz` |
| 双臂放回背部 | `PUTBACKALL` |
| 双臂从背部取块 | `GETBACKALL` |

示例：

```bash
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PICKALL,0.30,0.40,-0.21,0.30,-0.40,-0.21'"
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PLACEALL,0.30,0.40,-0.21,0.30,-0.40,-0.21'"
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'PUTBACKALL'"
ros2 topic pub --once /arm/mission_cmd std_msgs/msg/String "data: 'GETBACKALL'"
```

### 2.3 参数说明

| 参数 | 说明 |
| --- | --- |
| `L` / `0` / `LEFT` / `左` / `左臂` | 左臂 |
| `R` / `1` / `RIGHT` / `右` / `右臂` | 右臂 |
| `x,y,z` | 目标点坐标，单位 m |
| `lx,ly,lz` | 左臂目标点坐标，单位 m |
| `rx,ry,rz` | 右臂目标点坐标，单位 m |

中文命令也可用：

| 中文 data | 等价英文 |
| --- | --- |
| `取块,L,x,y,z` | `PICK,L,x,y,z` |
| `放置,L,x,y,z` / `放块,L,x,y,z` | `PLACE,L,x,y,z` |
| `放回背部,L` | `PUTBACK,L` |
| `背部取块,L` | `GETBACK,L` |
| `双臂取块,lx,ly,lz,rx,ry,rz` | `PICKALL,lx,ly,lz,rx,ry,rz` |
| `双臂放置,lx,ly,lz,rx,ry,rz` / `双臂放块,lx,ly,lz,rx,ry,rz` | `PLACEALL,lx,ly,lz,rx,ry,rz` |
| `双臂放回背部` | `PUTBACKALL` |
| `双臂背部取块` | `GETBACKALL` |

## 3. 高层任务反馈

监听同一个话题 `/arm/mission_cmd`，过滤 `FEEDBACK:` 开头的消息。

```bash
ros2 topic echo /arm/mission_cmd
```

| 反馈 data | 含义 |
| --- | --- |
| `FEEDBACK:DONE` | 当前任务收到 STM32 完成事件；也可能是超时后、下一条新任务开始前迟到的 DONE 补发。 |
| `FEEDBACK:TIMEOUT` | 上位机等待 DONE/DIAG 超时，已自动释放 busy。 |
| `FEEDBACK:BUSY` | 上一个任务还在执行，新任务被拒绝。 |
| `FEEDBACK:REJECTED:<reason>` | STM32 返回诊断拒绝，`reason` 是下位机原因码。 |

注意：如果日志出现 `TIMEOUT for ...`，说明超时时刻没有收到 STM32 的 DONE 帧；此时 `FEEDBACK:TIMEOUT` 只表示上位机释放 busy，不等价于真实动作完成。若下一条新任务开始前迟到 `DONE`，会再发布 `FEEDBACK:DONE`。

## 4. 低层调试 `/arm_internation/cmd`

类型：`std_msgs/msg/String`

作用：直接打包串口帧发送给 STM32，不经过高层任务 busy 管理。

### 4.1 低层动作命令

| 作用 | data |
| --- | --- |
| 单臂取块 | `PICK,L,x,y,z` / `PICK,R,x,y,z` |
| 单臂放块 | `PLACE,L,x,y,z` / `PLACE,R,x,y,z` |
| 单臂放回背部 | `PUTBACK,L` / `PUTBACK,R` |
| 单臂从背部取块 | `GETBACK,L` / `GETBACK,R` |
| 双臂取块 | `PICKALL,lx,ly,lz,rx,ry,rz` |
| 双臂放块 | `PLACEALL,lx,ly,lz,rx,ry,rz` |
| 双臂放回背部 | `PUTBACKALL` |
| 双臂从背部取块 | `GETBACKALL` |

示例：

```bash
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'PICK,L,0.30,0.40,-0.21'"
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'PLACE,R,0.30,-0.40,-0.21'"
```

### 4.2 直接位姿控制

| 作用 | data |
| --- | --- |
| 左臂位姿 | `4POSE,L,x,y,z,pitch` |
| 右臂位姿 | `4POSE,R,x,y,z,pitch` |
| 命名参数写法 | `4POSE,L,X:x,Y:y,Z:z,PITCH:pitch` |

示例：

```bash
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: '4POSE,L,0.10,0.20,0.30,0.40'"
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: '4POSE,R,X:0.10,Y:-0.20,Z:0.30,PITCH:0.40'"
```

### 4.3 预设动作和启动

| 作用 | data | 说明 |
| --- | --- | --- |
| 触发预设动作 | `4ACT,action_id` | `action_id=0` 通常表示中止。 |
| 启动/初始偏移 | `START,ox,oy,oz` | 偏移单位 mm。 |
| 启动/初始偏移命名写法 | `START,X:ox,Y:oy,Z:oz` | 偏移单位 mm。 |

示例：

```bash
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: '4ACT,0'"
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'START,0,0,0'"
```

### 4.4 电磁阀

| 作用 | data |
| --- | --- |
| 翻转单个电磁阀 | `V,id` |
| 打开单个电磁阀 | `V,id,ON` |
| 关闭单个电磁阀 | `V,id,OFF` |
| 打开全部电磁阀 | `V,ALL,ON` |
| 关闭全部电磁阀 | `V,ALL,OFF` |

`id` 定义：

| id | 含义 |
| --- | --- |
| `0` | 左臂吸盘 |
| `1` | 右臂吸盘 |
| `2` | 左背部吸盘 |
| `3` | 右背部吸盘 |

示例：

```bash
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'V,0,ON'"
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'V,ALL,OFF'"
```

建议协作调试时使用 `V,id,ON/OFF`，少用 `V,id` 翻转，避免上位机缓存和实际阀门状态不一致。

### 4.5 气泵

| 作用 | data |
| --- | --- |
| 开泵并设置速度 | `P,ON,speed` |
| 关泵 | `P,OFF` |

示例：

```bash
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'P,ON,2500'"
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'P,OFF'"
```

### 4.6 答案/语音

| 作用 | data |
| --- | --- |
| 发送答案字段 | `A,answer` |

示例：

```bash
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'A,2'"
```

## 5. OCR 答案 `/ocr/answer`

类型：`std_msgs/msg/UInt8`

作用：视觉/OCR 节点发布稳定答案，`arm_internation_node` 转为 STM32 `BB 05` 答案帧。

有效范围：`0..3`

```bash
ros2 topic pub --once /ocr/answer std_msgs/msg/UInt8 "{data: 2}"
```

## 6. 云台命令 `/arm_internation/cmd`

类型：`std_msgs/msg/String`

| 作用 | data |
| --- | --- |
| 启动云台 | `CAM_START` |
| 启动云台 | `GIMBAL_START` |
| 云台运动 | `CAM_MOVE,j1,pitch,yaw` |
| 云台运动 | `GIMBAL_MOVE,j1,pitch,yaw` |

角度单位：度。

示例：

```bash
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'CAM_START'"
ros2 topic pub --once /arm_internation/cmd std_msgs/msg/String "data: 'CAM_MOVE,90,-30,45'"
```

## 7. 状态监听

### 7.1 机械臂状态 `/arm_internation/data`

类型：`std_msgs/msg/String`

```bash
ros2 topic echo /arm_internation/data
```

data 格式：

```text
MODE:4DOF;L4:x,y,z,pitch;R4:x,y,z,pitch;VALVE_BITS:n;MICRO_BITS:n
```

示例：

```text
MODE:4DOF;L4:0.301,0.402,-0.210,-1.500;R4:0.300,-0.398,-0.211,-1.500;VALVE_BITS:3;MICRO_BITS:0
```

### 7.2 完成/诊断事件 `/arm_internation/state`

类型：`std_msgs/msg/String`

```bash
ros2 topic echo /arm_internation/state
```

| data | 含义 |
| --- | --- |
| `DONE` | STM32 返回动作完成事件。 |
| `DIAG,arm=a,reason=r,mask=m,req=x/y/z/p,lim=x/y/z/p` | STM32 返回诊断/拒绝事件。 |

示例：

```text
DONE
DIAG,arm=1,reason=4,mask=5,req=0.300/-0.400/-0.210/-1.500,lim=0.300/-0.400/-0.210/-1.500
```

## 8. 常用启动

```bash
colcon build --packages-select dogvision_arm
source install/setup.bash

# 不带终端
ros2 launch dogvision_arm arm_control.launch

# 带终端调试
ros2 launch dogvision_arm arm_test.launch
```

常用 launch 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `port` | 空 | 指定串口，如 `/dev/ttyACM0`；为空时按 `hw_id` 自动连接。 |
| `hw_id` | `0483:5740` | 自动连接的 USB VID:PID。 |
| `baud_rate` | `115200` | 串口波特率。 |
| `timeout_ms` | `15000` | 高层任务等待 DONE/DIAG 的超时时间；`0` 或负数禁用。 |
| `ocr_answer_topic` | `/ocr/answer` | OCR 答案输入话题。 |

示例：

```bash
ros2 launch dogvision_arm arm_control.launch port:=/dev/ttyACM0
ros2 launch dogvision_arm arm_control.launch timeout_ms:=15000
```
