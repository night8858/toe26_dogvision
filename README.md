# toe26_dogvision

基于 **ROS2 Jazzy + colcon** 的机器狗视觉、机械臂控制与算术题识别工作空间。

## 1. 项目结构

```text
toe26_dogvision/
├── src/
│   ├── dogvision_arm/               # 串口通信、任务编排、终端控制
│   ├── dogvision_vision/            # 相机、YOLO、PPOCR、数学题与视觉测试
│   └── dogvision_bringup/           # 系统 launch 与共享配置
├── fisheye_params.yaml
└── README.md
```

核心代码按“纯 C++ 库 + ROS2 节点入口”组织，视觉推理、OCR 工具、机械臂协议和数学题生成逻辑都尽量保持可独立调用。

## 2. 功能概述

| 包 | 可执行文件 | 功能 |
|---|---|---|
| `dogvision_arm` | `arm_internation_node` | 串口收发、状态发布、断线重连 |
| `dogvision_arm` | `arm_mission_node` | 高层任务指令拆解为低层机械臂命令 |
| `dogvision_arm` | `arm_cmd_terminal_node` | 终端输入命令并发布到控制话题 |
| `dogvision_vision` | `yolo_node` | 触发式单帧抓帧、YOLO 推理、2x4 网格 JSON 发布 |
| `dogvision_vision` | `yolo_accuracy_test_node` | 连续实时 YOLO 测试、窗口可视化、标注视频保存 |
| `dogvision_vision` | `ppocr_node` | 算术题 ROI、PPOCR 检测识别、表达式计算、JSON 发布 |
| `dogvision_vision` | `math_generator_node` | 生成算术题、全屏显示、写入 YAML |

## 3. 依赖安装

目标环境为 Ubuntu 24.04 + ROS2 Jazzy。
所需的库有：OpenVINO 2025 Archive、Hikvision MVS SDK、OpenCV、jsoncpp 和 libusb。

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  python3-colcon-common-extensions \
  libopencv-dev \
  libjsoncpp-dev \
  libusb-1.0-0-dev
```

安装 OpenVINO Archive 后，在构建和运行前加载环境。本项目当前验证使用的路径为：

```bash
curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.6/linux/l_openvino_toolkit_ubuntu24_2024.6.0.17404.4c0f47d2335_x86_64.tgz --output openvino_2024.6.0.tgz
tar -xf openvino_2024.6.0.tgz
sudo mv l_openvino_toolkit_ubuntu24_2024.6.0.17404.4c0f47d2335_x86_64 /opt/intel/openvino_2024.6.0
source /opt/intel/openvino_2024.6.0/setupvars.sh
```

Hikvision MVS SDK 是强制依赖。请安装到默认路径，使以下文件存在：

```text
/opt/MVS/include/MvCameraControl.h
/opt/MVS/lib/64/libMVFGControl.so
```

若缺少 MVS SDK，`dogvision_vision` 会在 CMake 配置阶段直接失败。

## 4. 构建

```bash
cd ~/toe26_dogvision
source /opt/ros/jazzy/setup.bash
source /opt/intel/openvino_2024.6.0/setupvars.sh

colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

如果终端默认 Python 来自 Conda，ROS2 消息生成可能因为缺少 NumPy 失败。构建时固定 `-DPython3_EXECUTABLE=/usr/bin/python3`，可以避免误用 Conda Python。

分包构建示例：

```bash
colcon build --packages-up-to dogvision_vision --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
colcon build --packages-up-to dogvision_arm --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

```
sudo nano /etc/ld.so.conf.d/hikvision_mvs.conf
将以下行粘贴到文件中并保存:
/opt/MVS/lib/64
更新系统的链接器缓存以应用更改:
sudo ldconfig
```
## 5. 配置

视觉配置位于：

```text
src/dogvision_vision/config/settings.json
```

模型路径相对于 `dogvision_vision` 的安装 share 目录解析。例如：

```jsonc
{
  "path": {
    "openvino_xml_file_path": "models/yolo/yolo/m26325.xml",
    "openvino_bin_file_path": "models/yolo/yolo/m26325.bin",
    "ppocr_det_model_path": "models/ppocr/ch_PP-OCRv4_det_infer/inference.pdmodel",
    "ppocr_rec_model_path": "models/ppocr/ch_PP-OCRv4_rec_infer/inference.pdmodel",
    "ppocr_dict_path": "models/ppocr/Dict/ppocr_keys_v1.txt"
  }
}
```

机械臂任务位置参数位于：

```text
src/dogvision_arm/pos_set.yaml
```

该文件已使用 ROS2 参数格式，可由 `arm_test.launch` 和 `arm_control.launch` 直接加载。

## 6. 快速使用

所有命令需先加载工作空间：

```bash
source /opt/ros/jazzy/setup.bash
source /home/waterking/openvino_toolkit_ubuntu24_2025.4.1.20426.82bbf0292c5_x86_64/setupvars.sh
source install/setup.bash
```

启动全系统。默认是生产模式，包含机械臂控制、任务节点、YOLO 和 PPOCR，不启动终端输入节点：

```bash
ros2 launch dogvision_bringup full_system.launch
```

仅启动视觉：

```bash
ros2 launch dogvision_bringup vision.launch
```

YOLO 准确性测试，会打开可视化窗口并在退出后保存 AVI/MJPG 标注视频：

```bash
ros2 launch dogvision_vision yolo_accuracy_test.launch
```

PPOCR 连续测试模式：

```bash
ros2 launch dogvision_vision ppocr_test.launch
```

数学题生成器：

```bash
ros2 launch dogvision_vision math_generator.launch
```

触发 YOLO：

```bash
ros2 topic pub --once /yolo/trigger std_msgs/msg/String "{data: start_infer}"
```

查看结果：

```bash
ros2 topic echo /yolo/result
ros2 topic echo /yolo/block_grid
ros2 topic echo /ocr/result
```

YOLO 每次触发后只执行一次单帧推理，并默认保存带检测框、类别和置信度的结果图到：

```text
install/dogvision_vision/share/dogvision_vision/data/yolorun
```

仅启动机械臂测试：

```bash
ros2 launch dogvision_arm arm_test.launch
```

仅启动机械臂生产控制：

```bash
ros2 launch dogvision_arm arm_control.launch
```

单独运行节点：

```bash
ros2 run dogvision_arm arm_internation_node
ros2 run dogvision_arm arm_cmd_terminal_node
ros2 run dogvision_arm arm_mission_node
ros2 run dogvision_vision yolo_node
ros2 run dogvision_vision yolo_accuracy_test_node
ros2 run dogvision_vision ppocr_node --ros-args -p mode:=production
ros2 run dogvision_vision math_generator_node
```

## 7. Launch 参数

### `dogvision_bringup full_system.launch`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `hw_id` | `0483:5740` | 机械臂 USB 设备 VID:PID |
| `baud_rate` | `115200` | 机械臂串口波特率 |
| `port` | 空字符串 | 指定串口路径，非空时跳过 HWID 扫描 |
| `pos_scale` | `0.01` | 机械臂坐标换算比例 |
| `angle_scale` | `0.01` | 云台角度换算比例 |
| `mission_config` | `<share>/dogvision_arm/config/pos_set.yaml` | 机械臂任务位置配置 |
| `config_path` | `<share>/dogvision_vision/config/settings.json` | 视觉配置文件 |
| `show_window` | `false` | YOLO 是否显示 OpenCV 窗口 |
| `enable_undistort` | `true` | YOLO 是否启用去畸变 |
| `save_images` | `true` | 是否保存每次触发后的 YOLO 结果图 |
| `save_dir` | `<share>/dogvision_vision/data/yolorun` | YOLO 结果图保存目录 |
| `ppocr_mode` | `production` | PPOCR 模式，支持 `production` 或 `test` |
| `ppocr_show_visual` | `true` | PPOCR 是否显示可视化窗口 |
| `ocr_yaml_path` | `<share>/dogvision_vision/data/ocr_output/ocr_results.yaml` | PPOCR test 模式输出 YAML |

### `dogvision_vision yolo_accuracy_test.launch`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `config_path` | `<share>/dogvision_vision/config/settings.json` | 视觉配置文件 |
| `enable_undistort` | `true` | 是否启用去畸变 |
| `output_dir` | `<share>/dogvision_vision/data/yolotest` | 测试视频输出目录 |
| `video_fps` | `20.0` | 保存视频的帧率 |
| `visual_nms_thresh` | `0.7` | 测试可视化 NMS 阈值，较高时更容易保留同帧多个目标 |

### `dogvision_arm arm_control.launch` 与 `arm_test.launch`

两者参数相同，`arm_control.launch` 面向生产运行，`arm_test.launch` 额外启动 `arm_cmd_terminal_node` 便于手动输入命令。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `hw_id` | `0483:5740` | USB 设备 VID:PID |
| `baud_rate` | `115200` | 串口波特率 |
| `port` | 空字符串 | 指定串口路径 |
| `pos_scale` | `0.01` | 坐标换算比例 |
| `angle_scale` | `0.01` | 角度换算比例 |
| `mission_config` | `<share>/dogvision_arm/config/pos_set.yaml` | 任务位置配置 |

### `dogvision_bringup vision.launch`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `config_path` | `<share>/dogvision_vision/config/settings.json` | 视觉配置文件 |
| `show_window` | `false` | YOLO 是否显示 OpenCV 窗口 |
| `enable_undistort` | `true` | YOLO 是否启用去畸变 |
| `save_images` | `true` | 是否保存每次触发后的 YOLO 结果图 |
| `save_dir` | `<share>/dogvision_vision/data/yolorun` | YOLO 结果图保存目录 |
| `ppocr_mode` | `production` | PPOCR 模式 |
| `ppocr_show_visual` | `true` | PPOCR 是否显示可视化窗口 |
| `ocr_yaml_path` | `<share>/dogvision_vision/data/ocr_output/ocr_results.yaml` | PPOCR test 模式输出 YAML |

## 8. 话题与消息

| 话题 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/arm_internation/data` | `std_msgs/msg/String` | 发布 | 机械臂、云台、传感器状态，20Hz |
| `/arm_internation/cmd` | `std_msgs/msg/String` | 订阅 | 低层文本控制命令 |
| `/arm/mission_cmd` | `std_msgs/msg/String` | 订阅/反馈 | 高层任务命令与完成反馈 |
| `/yolo/trigger` | `std_msgs/msg/String` | 订阅 | 发布 `start_infer` 触发一次单帧推理 |
| `/yolo/result` | `std_msgs/msg/String` | 发布 | transient_local YOLO JSON 结果 |
| `/yolo/block_grid` | `std_msgs/msg/String` | 发布 | transient_local 2x4 网格 JSON |
| `/ocr/trigger` | `std_msgs/msg/String` | 订阅 | 启动或重置生产模式 OCR 持续跟踪 |
| `/ocr/result` | `std_msgs/msg/String` | 发布 | transient_local 稳定 OCR JSON 结果 |

`/arm_internation/data` 示例：

```text
LF:2.547,2.547;RF:-276.225,-337.042;LB:276.225,337.042;RB:276.225,-337.042;YAW:0.000;PITCH:0.000;VALVE_BITS:0;MICRO_BITS:0
```

`/yolo/result` 示例：

```json
{"detections":[{"pos_id":1,"class":"food","conf":0.8821,"bbox":[120.0,80.0,200.0,150.0]}]}
```

`/yolo/block_grid` 示例：

```json
{"block":[["food","tool","null","null"],["medicine","null","null","null"]]}
```

`/ocr/result` 示例：

```json
{"expr":"12+3*4","result":24,"mod4":0}
```

PPOCR 使用最近 10 个处理帧进行投票。某个归一化算式至少出现 6 次，
并且占窗口内有效识别结果的 60% 以上时，才会成为稳定结果并发布。
稳定结果会保持到另一个算式满足相同条件；连续 10 帧没有有效算式时
只清除本地可视化，不发布错误消息。生产模式收到新的 `/ocr/trigger`
后会清空历史并开始新一轮持续跟踪。

## 9. 节点参数

### `arm_internation_node`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `hw_id` | `0483:5740` | USB 设备 VID:PID |
| `baud_rate` | `115200` | 串口波特率 |
| `port` | 空字符串 | 指定串口路径，非空时跳过 HWID 扫描 |
| `cmd_topic` | `/arm_internation/cmd` | 低层命令订阅话题 |
| `data_topic` | `/arm_internation/data` | 状态发布话题 |
| `pos_scale` | `0.01` | 坐标换算比例 |
| `angle_scale` | `0.01` | 角度换算比例 |

### `arm_mission_node`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `mission_topic` | `/arm/mission_cmd` | 高层任务命令订阅话题 |
| `cmd_topic` | `/arm_internation/cmd` | 低层命令发布话题 |
| `stow_pos.*` | 见 `pos_set.yaml` | 收起位置 |
| `pick_pos.*` | 见 `pos_set.yaml` | 吸取位置 |
| `place_pos.*` | 见 `pos_set.yaml` | 放置位置 |
| `start_pos.*` | 见 `pos_set.yaml` | 启动位置 |

### `yolo_node`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `config_path` | `<share>/dogvision_vision/config/settings.json` | 视觉配置文件 |
| `result_topic` | `/yolo/result` | 检测结果发布话题 |
| `show_window` | `false` | 是否显示本地 OpenCV 窗口 |
| `enable_undistort` | `true` | 是否进行鱼眼去畸变 |
| `save_images` | `true` | 是否保存每次触发后的 YOLO 结果图 |
| `save_dir` | `<share>/dogvision_vision/data/yolorun` | YOLO 结果图保存目录 |

### `yolo_accuracy_test_node`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `config_path` | `<share>/dogvision_vision/config/settings.json` | 视觉配置文件 |
| `enable_undistort` | `true` | 是否进行鱼眼去畸变 |
| `output_dir` | `<share>/dogvision_vision/data/yolotest` | 标注测试视频输出目录 |
| `video_fps` | `20.0` | AVI/MJPG 视频写入帧率 |
| `visual_nms_thresh` | `0.7` | 测试可视化 NMS 阈值，避免相近目标被过度合并 |

### `ppocr_node`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `config_path` | `<share>/dogvision_vision/config/settings.json` | 视觉配置文件 |
| `mode` | `production` | `production` 触发后持续跟踪，`test` 连续投票并将稳定变化写入 YAML |
| `show_visual` | `true` | 是否显示本地 OpenCV 窗口 |
| `yaml_path` | `<share>/dogvision_vision/data/ocr_output/ocr_results.yaml` | test 模式输出文件 |

### `math_generator_node`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `yaml_path` | `<share>/dogvision_vision/data/math_generator/math_results.yaml` | 输出 YAML 路径 |
| `min_val` | `1` | 操作数最小值 |
| `max_val` | `100` | 操作数最大值 |
| `interval` | `10` | 生成间隔，单位秒 |

## 10. 串口协议说明

设备通过 USB-CDC 连接，默认 VID:PID 为 `0483:5740`，波特率 `115200`。

反馈帧使用 `AA 01`：

| 字节范围 | 内容 |
|---|---|
| `[0]` | `0xAA` 帧头 |
| `[1]` | `0x01` 命令字 |
| `[2..33]` | 四个机械臂末端坐标，float 小端 |
| `[34..41]` | 云台 yaw、pitch，float 小端 |
| `[42..45]` | 电磁阀与微动开关状态 |
| `[46..47]` | `0xFF 0xEE` 帧尾 |
| `[48]` | CRC-8/SMBUS |

低层命令仍使用文本形式，例如：

```text
LF,X:100,Y:50
G,30,10
V,1,ON
P,ON,2500
A,0
```

## 11. 静态验证

本次迁移以静态验证为主：

检查源码和文档中是否仍有旧版 ROS1 API 或命令残留，并确认每个包都调用 `ament_package()`。

由于 MVS SDK 为强制依赖，未安装 `/opt/MVS` 时不要求本机完整构建成功。
