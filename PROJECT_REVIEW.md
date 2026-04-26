# dogvision26 项目结构审查与改进指南

## 一、当前状态总结

构建本身可以通过（`catkin_make` 返回 0），说明基础是可用的。  
但从 ROS 工程规范角度存在若干值得改进的问题，下文逐一说明。

---

## 二、现存问题清单

### 2.1 包结构：单包承载过多职责（高优先级）

当前 `dogvision26` 一个包包含了：视觉检测、OCR、海康相机驱动、机械臂串口通信、工具程序四个完全独立的功能域。

**问题：**
- 任意一个模块改动都需要重新编译整个包
- 不同功能域的依赖（OpenVINO、MVS SDK、串口库）全部耦合在一个 `CMakeLists.txt` 中
- 无法单独发布或复用某一模块

**建议拆分为多个包：**

```
src/
├── dogvision_msgs/         # 自定义消息（独立，被其他包依赖）
├── dogvision_arm/          # 机械臂串口通信
├── dogvision_vision/       # YOLO + OCR 视觉检测
├── dogvision_camera/       # 相机驱动（海康/USB）
└── dogvision_bringup/      # launch 文件、顶层配置（不含代码）
```

---

### 2.2 头文件放在 `src/` 而非 `include/`（高优先级）

| 当前位置 | 应该放置的位置 |
|---|---|
| `src/internation/arm_internation.hpp` | `include/dogvision26/arm_internation.hpp` |
| `src/camera/hikvision.hpp` | `include/dogvision26/hikvision.hpp` |
| `src/detect/detector.hpp` | `include/dogvision26/detector.hpp` |
| `src/common_structs.h` | `include/dogvision26/common_structs.h` |

ROS 规范要求对外暴露的头文件放在 `include/<package_name>/` 下，  
并在 `catkin_package(INCLUDE_DIRS include)` 中声明。  
目前 `CMakeLists.txt` 已声明了 `INCLUDE_DIRS include`，但头文件实际不在那里，  
导致 `include_directories` 不得不把 `src/` 整个目录加进去——这会让私有实现文件也变成"可被外部包包含"的头文件路径。

**修正方式（以拆包后为准）：**
```
include/
└── dogvision26/
    ├── arm_internation.hpp
    ├── common_structs.h
    ├── hikvision.hpp
    └── detector.hpp
src/
    ├── arm_internation.cpp
    ├── hikvision.cpp
    └── ...
```

---

### 2.3 自定义消息已定义但未启用（中优先级）

`src/msg/arm4_contorl.msg` 存在，但 `CMakeLists.txt` 中：

```cmake
# add_message_files(...)   ← 注释掉了
# generate_messages(...)   ← 注释掉了
```

同时 `package.xml` 缺少 `message_generation` / `message_runtime` 依赖。  
目前节点通过 `std_msgs/String` 传输文本命令作为替代方案——这是可接受的临时做法，  
但如果要正式使用结构化消息，需要同时完成以下三步：

**CMakeLists.txt：**
```cmake
find_package(catkin REQUIRED COMPONENTS
  roscpp std_msgs message_generation
)

add_message_files(FILES arm4_contorl.msg)
generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs message_runtime
)
```

**package.xml：**
```xml
<build_depend>message_generation</build_depend>
<exec_depend>message_runtime</exec_depend>
```

> 附：消息文件名有拼写错误 `arm4_contorl.msg` → 建议改为 `arm4_control.msg`

---

### 2.4 配置文件与数据文件位置不规范（中优先级）

| 当前位置 | 建议位置 |
|---|---|
| `../../fisheye_params.yaml`（工作空间根目录） | `dogvision26/config/fisheye_params.yaml` |
| `src/settings.json` | `dogvision26/config/settings.json` |
| `src/data/yolo/m26325.onnx` 等模型文件 | `dogvision26/models/` |
| `src/data/ppocr/` 推理模型 | `dogvision26/models/ppocr/` |

ROS 惯例是通过 `ros::package::getPath("dogvision26")` 获取包路径，  
再拼接相对路径来访问配置/数据文件，而不是写死绝对路径。

---

### 2.5 launch 文件位置不规范（中优先级）

| 当前位置 | 建议位置 |
|---|---|
| `src/start/internation_test.launch` | `launch/internation_test.launch` |

`launch/` 目录应在包根目录下，而不是 `src/` 子目录里。  
同时 launch 文件中 `arm_cmd_terminal_node` 使用了 `launch-prefix` 打开独立终端会更易用：

```xml
<node name="arm_cmd_terminal_node" pkg="dogvision26" type="arm_cmd_terminal_node"
      output="screen" launch-prefix="xterm -e"/>
```

---

### 2.6 工具程序用 `add_subdirectory` 嵌入主包（低优先级）

`src/tool/onnx2openvino` 和 `src/tool/Distortion_handling` 都有独立的 `CMakeLists.txt`，  
是独立的命令行工具，与 ROS 无关。

**建议：**
- 若与 ROS 无关，从 catkin 工作空间中移出，作为独立 CMake 项目维护
- 若需要保留，改为独立的 catkin 包，放在 `src/` 下

---

### 2.7 `package.xml` 信息不完整（低优先级）

```xml
<!-- 当前 -->
<version>0.0.0</version>
<maintainer email="toe@todo.todo">toe</maintainer>
<license>TODO</license>

<!-- 建议 -->
<version>1.0.0</version>
<maintainer email="your@email.com">Your Name</maintainer>
<license>MIT</license>  <!-- 或 BSD / GPLv3 -->
```

`package.xml` 中声明了 `rospy` 依赖，但项目中没有 Python 脚本，可以删除。

---

### 2.8 源码拼写错误（低优先级）

| 文件 | 错误 | 建议 |
|---|---|---|
| `src/detect/dtetctor.cpp` | `dtetctor` | `detector` |
| `src/msg/arm4_contorl.msg` | `contorl` | `control` |
| `src/camera/CMakeLists.txt` | 单独一个 CMakeLists 但未被使用 | 删除 |

---

## 三、建议的目标项目树

```
src/
├── dogvision_msgs/                  # 自定义消息包（无依赖，最先编译）
│   ├── CMakeLists.txt
│   ├── package.xml
│   └── msg/
│       ├── arm4_control.msg
│       └── detection_result.msg
│
├── dogvision_arm/                   # 机械臂串口通信包
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── include/dogvision_arm/
│   │   └── arm_internation.hpp
│   ├── src/
│   │   ├── arm_internation.cpp
│   │   ├── Arm_internation_node.cpp
│   │   └── arm_cmd_terminal_node.cpp
│   └── launch/
│       └── arm.launch
│
├── dogvision_camera/                # 相机驱动包
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── include/dogvision_camera/
│   │   ├── hikvision.hpp
│   │   └── usbcam.hpp
│   └── src/
│       ├── hikvision.cpp
│       └── usbcam.cpp
│
├── dogvision_vision/                # 视觉检测包（YOLO + OCR）
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── include/dogvision_vision/
│   │   ├── detector.hpp
│   │   ├── nuc_detect.hpp
│   │   └── ocr_detect.hpp
│   ├── src/
│   │   ├── detector.cpp
│   │   ├── nuc_detect.cpp
│   │   ├── ocr_detect.cpp
│   │   ├── ppocr_node.cpp
│   │   └── yolo_node.cpp
│   └── models/                      # 模型文件
│       ├── yolo/m26325.onnx
│       └── ppocr/...
│
└── dogvision_bringup/               # 顶层启动包（只含 launch/config，无代码）
    ├── CMakeLists.txt
    ├── package.xml
    ├── config/
    │   ├── fisheye_params.yaml
    │   └── settings.yaml
    └── launch/
        ├── arm_test.launch
        ├── vision.launch
        └── full_system.launch
```

---

## 四、迁移优先级建议

考虑到当前项目已能编译运行，建议按以下顺序渐进改进，不必一次完成：

| 优先级 | 任务 | 风险 | 状态 |
|---|---|---|---|
| 🔴 高 | 将 launch 文件移到包根的 `launch/` | 低，只需更新路径 | ✅ 已完成 |
| 🔴 高 | 将配置文件移到 `config/`，用 `ros::package::getPath` 引用 | 低 | ✅ 已完成 |
| 🟡 中 | 将头文件移到 `include/<pkg>/` | 中，需同步更新 `#include` 路径 | ✅ 已完成 |
| 🟡 中 | 正式启用或删除 `arm4_control.msg` | 低 | ✅ 已完成（启用） |
| 🟢 低 | 拆分为多个 catkin 包 | 高，需重写所有 CMakeLists.txt | ✅ 已完成 |
| 🟢 低 | 修复源码拼写错误（文件重命名） | 中，需更新所有引用 | ✅ 已完成（dtetctor→detector，contorl→control） |

---

## 五、快速验证命令

```bash
# 检查包依赖关系是否声明完整
catkin_lint src/dogvision26

# 查看当前话题连接情况（运行时）
rosnode info /arm_internation_node
rqt_graph

# 查看包安装路径（用于验证 getPath 结果）
rospack find dogvision26
```
