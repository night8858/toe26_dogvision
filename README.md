# toe26_dogvision

基于 **ROS1 Noetic + catkin** 的机器狗视觉与机械臂控制工作空间。

---

## 目录

1. [项目结构](#1-项目结构)
2. [功能概述](#2-功能概述)
3. [依赖安装](#3-依赖安装)
4. [构建](#4-构建)
5. [配置](#5-配置)
6. [快速使用](#6-快速使用)
7. [话题与消息](#7-话题与消息)
8. [节点参数](#8-节点参数)
9. [串口协议说明](#9-串口协议说明)

---

## 1. 项目结构

```
toe26_dogvision/                        # catkin 工作空间根
├── build/                              # 构建产物（自动生成）
├── devel/                              # 开发空间（自动生成）
├── launch/                             # 工作空间级 launch（测试用）
│   └── internation_test.launch
└── src/
    ├── CMakeLists.txt                  # catkin 顶层 CMake
    │
    ├── dogvision_msgs/                 # 自定义消息包
    │   └── msg/arm4_control.msg
    │
    ├── dogvision_arm/                  # 机械臂串口通信包
    │   ├── include/dogvision_arm/
    │   │   └── arm_internation.hpp     # 串口类（连接/收发/命令解析）
    │   ├── src/
    │   │   ├── arm_internation.cpp
    │   │   ├── Arm_internation_node.cpp   # 串口通信节点（200Hz）
    │   │   └── arm_cmd_terminal_node.cpp  # 终端命令输入节点
    │   └── launch/arm.launch
    │
    ├── dogvision_camera/               # 相机驱动库包
    │   ├── include/dogvision_camera/
    │   │   ├── hikvision.hpp           # 海康工业相机（MVS SDK）
    │   │   └── usbcam.hpp              # USB 摄像头（OpenCV）
    │   └── src/
    │       ├── hikvision.cpp
    │       └── usbcam.cpp
    │
    ├── dogvision_vision/               # 视觉推理包
    │   ├── include/dogvision_vision/
    │   │   ├── common_structs.h        # 全局数据结构
    │   │   ├── detector.hpp            # 检测器基类
    │   │   ├── nuc_detect.hpp          # YOLO OpenVINO 检测器
    │   │   ├── ocr_detect.hpp          # PPOCR 检测/识别器
    │   │   ├── yolo_utils.hpp          # YOLO 工具函数
    │   │   └── ocr_utils.hpp           # OCR 工具函数
    │   ├── src/
    │   │   ├── detector.cpp            # 基类：配置加载、鱼眼矫正
    │   │   ├── nuc_detect.cpp          # YOLO 推理实现
    │   │   ├── ocr_detect.cpp          # PPOCR 推理实现
    │   │   ├── yolo_utils.cpp          # NMS/网格分配/JSON序列化
    │   │   ├── ocr_utils.cpp           # 裁剪/算术解析/ROI定位
    │   │   ├── yolo_node.cpp           # YOLO 节点入口
    │   │   └── ppocr_node.cpp          # PPOCR 节点入口
    │   ├── config/settings.json        # 运行时配置（模型路径/相机参数）
    │   └── models/
    │       ├── yolo/yolo/              # YOLO IR 模型（.xml/.bin）
    │       └── ppocr/                  # PPOCRv4 Paddle 模型（.pdmodel）
    │
    └── dogvision_bringup/              # 启动文件与全局配置包
        ├── launch/
        │   ├── full_system.launch      # 一键启动全系统
        │   ├── internation_test.launch # 仅启动机械臂子系统
        │   └── vision.launch           # 仅启动视觉子系统
        └── config/
            ├── settings.json
            └── fisheye_params.yaml
```

---

## 2. 功能概述

| 包 | 节点 | 功能 |
|---|---|---|
| `dogvision_arm` | `Arm_internation_node` | 串口收发（53B 协议帧）、状态发布 20Hz、每秒断线检测重连 |
| `dogvision_arm` | `arm_cmd_terminal_node` | 终端输入命令并转发到 `/arm_internation/cmd` |
| `dogvision_vision` | `yolo_node` | 触发式抓帧→YOLO推理→跨帧NMS→2×4网格分配→JSON发布 |
| `dogvision_vision` | `ppocr_node` | 白色算术题区域定位→PPOCR文本检测+识别→算术计算→结果发布 |

---

## 3. 依赖安装

### 3.1 基础依赖（Ubuntu 20.04）

```bash
sudo apt-get update
sudo apt-get install -y \
    ros-noetic-desktop-full \
    python3-catkin-tools \
    libopencv-dev \
    libjsoncpp-dev \
    xterm
```

### 3.2 OpenVINO（≥ 2024.0）

```bash
# 方式一：官方脚本（自动处理依赖）
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu20 main" \
    | sudo tee /etc/apt/sources.list.d/intel-openvino-2024.list
sudo apt-get update
sudo apt-get install -y openvino-2024.0.0

# 激活（每次新终端需执行，或写入 ~/.bashrc）
source /opt/intel/openvino_2024/setupvars.sh
```

> 也可从 [https://docs.openvino.ai](https://docs.openvino.ai) 下载离线安装包。（我是如此安装的）

### 3.3 海康威视 MVS SDK（可选，使用工业相机时必须）

1. 从[海康机器人官网](https://www.hikrobotics.com/cn/machinevision/service/download)下载 MVS Linux 安装包
2. 解压后运行安装脚本：

```bash
tar -xzf MVS-x.x.x_x86_64_yyyymmdd.tar.gz
cd MVS-x.x.x_x86_64
sudo bash setup.sh
# 头文件安装至 /opt/MVS/include，库文件至 /opt/MVS/lib
```

### 3.4 PaddlePaddle 推理库（PPOCRv4 模型加载）

PPOCRv4 模型通过 **OpenVINO 前端**直接加载 `.pdmodel`，无需安装 PaddlePaddle。  
确保 OpenVINO 安装包含 Paddle 前端（`openvino-2024.0.0` 默认包含）。

### 3.5 Python 工具（可选，串口调试）

```bash
pip3 install pyserial
```

---

## 4. 构建

```bash
cd ~/toe26_dogvision

# 加载 ROS 环境
source /opt/ros/noetic/setup.bash

# 全量构建
catkin_make -j8

# 加载工作空间环境（写入 ~/.bashrc 避免每次手动执行）
source devel/setup.bash     
```
我们也可以选择zsh或fish，并将 `source devel/setup.bash` 替换为相应的 `setup.zsh` 或 `setup.fish`。

**分包构建**（调试单个包时更快）：

```bash
catkin_make --only-pkg-with-deps dogvision_arm
catkin_make --only-pkg-with-deps dogvision_vision
```

---

## 5. 配置

所有运行参数集中在：

```
src/dogvision_vision/config/settings.json
```

必须在首次运行前检查以下字段：

```jsonc
{
  "path": {
    // YOLO 模型（OpenVINO IR 格式）
    "openvino_xml_file_path": "models/yolo/yolo/m26325.xml",
    "openvino_bin_file_path": "models/yolo/yolo/m26325.bin",

    // PPOCRv4 模型（Paddle inference 格式）
    "ppocr_det_model_path": "models/ppocr/ch_PP-OCRv4_det_infer/inference.pdmodel",
    "ppocr_rec_model_path": "models/ppocr/ch_PP-OCRv4_rec_infer/inference.pdmodel",
    "ppocr_dict_path":      "models/ppocr/ppocr/Dict/ppocr_keys_v1.txt"
  },
  "hikcamera": {
    "device_id": 0,
    "width": 1440, "height": 1080,
    "exposure": 11000
  },
  "nums": {
    "classes": 4,
    "cls0": "food", "cls1": "tool", "cls2": "medicine", "cls3": "instrument"
  }
}
```

路径均为相对于 `dogvision_vision` 包根目录的相对路径，由 `ros::package::getPath("dogvision_vision")` 自动拼接。

---

## 6. 快速使用

> 所有操作均需先执行 `source devel/setup.bash`

### 6.1 启动全系统（机械臂 + 视觉）

```bash
cd ~/toe26_dogvision
roslaunch dogvision_bringup full_system.launch
```

### 6.2 仅启动机械臂子系统

```bash
roslaunch dogvision_bringup internation_test.launch
```

`arm_cmd_terminal_node` 会在 xterm 窗口中等待命令输入，支持如下格式：

```
RL,X:100,Y:50      # 控制左前臂移动至 (100, 50)
RF,X:0,Y:0         # 控制右前臂回零
G,30,10            # 云台 yaw=30 pitch=10（int16 角度）
V,1,ON             # 打开电磁阀 1
V,2                # 翻转电磁阀 2 状态
quit               # 退出节点
```

### 6.3 仅启动视觉子系统

```bash
roslaunch dogvision_bringup vision.launch
```

手动触发 YOLO 推理：

```bash
rostopic pub /yolo/trigger std_msgs/String "data: 'start_infer'" --once
```

查看 YOLO 结果：

```bash
rostopic echo /yolo/result          # JSON 检测结果
rostopic echo /yolo/block_grid      # 2×4 网格分配结果
```

查看机械臂状态：

```bash
rostopic echo /arm_internation/data
```

### 6.4 单独运行节点（调试）

```bash
rosrun dogvision_arm Arm_internation_node
rosrun dogvision_arm arm_cmd_terminal_node
rosrun dogvision_vision yolo_node
rosrun dogvision_vision ppocr_node
```

---

## 7. 话题与消息

| 话题 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/arm_internation/data` | `std_msgs/String` | 发布 | 机械臂 + 云台 + 传感器状态，20Hz |
| `/arm_internation/cmd` | `std_msgs/String` | 订阅 | 文本控制命令 |
| `/yolo/trigger` | `std_msgs/String` | 订阅 | 发布 `"start_infer"` 触发一次推理 |
| `/yolo/result` | `std_msgs/String` | 发布 | 检测结果 JSON（latched） |
| `/yolo/block_grid` | `std_msgs/String` | 发布 | 2×4 网格 JSON（latched） |

**`/arm_internation/data` 格式示例：**
```
LF:2.547,2.547;RF:-276.225,-337.042;LB:276.225,337.042;RB:276.225,-337.042;YAW:0.000;PITCH:0.000;VALVE_BITS:0;MICRO_BITS:0
```

**`/yolo/result` 格式示例：**
```json
{"detections":[{"pos_id":1,"class":"food","conf":0.8821,"bbox":[120.0,80.0,200.0,150.0]}]}
```

**`/yolo/block_grid` 格式示例：**
```json
{"block":[["food","tool","null","null"],["medicine","null","null","null"]]}
```

---

## 8. 节点参数

### `Arm_internation_node`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `~hw_id` | `"0483:5740"` | USB 设备 VID:PID（用于自动定位 ttyACM*） |
| `~baud_rate` | `115200` | 串口波特率 |
| `~cmd_topic` | `/arm_internation/cmd` | 订阅命令的话题 |
| `~data_topic` | `/arm_internation/data` | 发布状态的话题 |
| `~pos_scale` | `0.01` | float→int16 坐标换算比例 |
| `~angle_scale` | `0.01` | float→int16 角度换算比例 |

### `yolo_node`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `~config_path` | `<pkg>/config/settings.json` | 配置文件路径 |
| `~result_topic` | `/yolo/result` | 结果发布话题 |
| `~show_window` | `true` | 是否显示 OpenCV 可视化窗口 |
| `~enable_undistort` | `true` | 是否对图像做鱼眼矫正 |

### `ppocr_node`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `~config_path` | `<pkg>/config/settings.json` | 配置文件路径 |
| `~image_path` | `<pkg>/data/img/...` | 输入图像路径 |
| `~output_dir` | `<pkg>/data/ocr_output` | OCR 结果输出目录 |

---

## 9. 串口协议说明

设备通过 USB-CDC（`/dev/ttyACM*`，VID:PID = `0483:5740`）连接，波特率 115200。

**AA 01 反馈帧（53 字节，V2 格式）：**

| 字节范围 | 内容 |
|---|---|
| `[0]` | `0xAA`（帧头） |
| `[1]` | `0x01`（命令字） |
| `[2~9]` | LF 末端坐标 x, y（各 4 字节 float，小端） |
| `[10~17]` | RF 末端坐标 x, y |
| `[18~25]` | LB 末端坐标 x, y |
| `[26~33]` | RB 末端坐标 x, y |
| `[34~37]` | YAW（4 字节 float） |
| `[38~41]` | PITCH（4 字节 float） |
| `[42~45]` | 电磁阀 / 微动开关状态位 |
| `[46~49]` | 保留（4 字节，V2 扩展） |
| `[50~51]` | `0xFF 0xEE`（帧尾） |
| `[52]` | CRC-8/SMBUS（覆盖 `[0]~[51]`） |

