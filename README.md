# toe26_dogvision — ROS 视觉检测与 OCR 推理框架
作者：toe 水王 
项目归属： toe



## 1. 项目概述

本项目是一个基于 **ROS1 (catkin)** 的视觉处理工作空间，核心功能包括：

- **YOLO 目标检测**：使用 OpenVINO 加载 ONNX/IR 模型，完成预处理→推理→后处理全流程，目前使用yolov8模型
- **PP-OCR 文字识别**：基于 PaddleOCRV4 的检测
- **多相机采集**：支持海康工业相机（MVS SDK）和 USB 摄像头（OpenCV VideoCapture）


---

## 2. 系统依赖

| 依赖            | 最低版本         | 用途                     |
| --------------- | ---------------- | ------------------------ |
| ROS1 (Noetic)   | 1.15             | 节点通信框架              |
| OpenCV          | 4.x              | 图像处理                  |
| OpenVINO        | 2024+            | 模型推理                  |
| JsonCpp         | 1.9+             | JSON 配置解析             |
| MVS SDK         | 3.x（可选）      | 海康相机采集              |
| C++ 标准        | C++17            | 编译要求                  |
| CMake           | 3.0.2+           | 构建系统                  |

---

## 3. 目录结构

```text
toe26_dogvision/                    # catkin 工作空间根目录
├── README.md                       # ← 本文件（项目级说明）
├── build/                          # catkin 构建产物（自动生成）
├── devel/                          # catkin 开发空间（自动生成）
└── src/
    ├── CMakeLists.txt              # catkin 顶层 CMake（自动生成）
    └── dogvision26/                # 核心 ROS 包
        ├── CMakeLists.txt          # 包构建脚本
        ├── package.xml             # 包元信息与依赖声明
        ├── README.md               # 包级说明
        ├── include/dogvision26/    # 公开头文件（预留）
        └── src/
            ├── common_structs.h    # 全局数据结构定义
            ├── yolo_node.cpp       # YOLO 检测节点入口
            ├── ppocr_node.cpp      # PP-OCR 识别节点入口
            ├── camera/             # 相机采集模块
            │   ├── hikvision.hpp/cpp   # 海康工业相机封装
            │   ├── usbcam.hpp/cpp      # USB 摄像头封装
            │   ├── grab_test.cpp       # 相机独立测试程序
            │   └── CMakeLists.txt      # 相机测试独立构建
            ├── detect/             # 检测抽象层
            │   ├── detector.hpp    # 检测器基类（纯虚接口）
            │   ├── dtetctor.cpp    # 基类实现（配置加载/图像缓存/画框）
            │   ├── settings.json   # 运行时配置文件
            │   ├── yolo_detector/  # YOLO OpenVINO 实现
            │   │   ├── nuc_detect.hpp
            │   │   └── nuc_detect.cpp
            │   └── ppocr_detector/ # PP-OCR OpenVINO 实现
            │       ├── ocr_detect.hpp
            │       └── ocr_detect.cpp
            └── data/               # 模型与数据文件
                ├── yolo/           # YOLO 模型存放处（需手动放入）
                └── ppocr/          # PP-OCR 模型文件
                    └── ch_PP-OCRv4_rec_server_infer/
```

---

## 4. 类继承关系

```text
detector (基类, detector.hpp)
├── detect_oponvino (YOLO, nuc_detect.hpp)
│   └── 重写: preprocess / inference / postprocess
├── detect_det_ppocr (OCR 文本检测, ocr_detect.hpp)
│   └── 重写: preprocess / inference / postprocess
├── detect_rec_ppocr (OCR 文本识别, ocr_detect.hpp)
│   └── 重写: preprocess / inference / postprocess
└── detect_cls_ppocr (OCR 方向分类, ocr_detect.hpp)
    └── 预留骨架
```

---

## 5. 数据流与处理管线

### 5.1 YOLO 检测管线

```text
settings.json ──→ load_config() ──→ Appconfig
                                       │
相机/ROS话题 ──→ push_img() ──→ input_img_hik_
                                       │
                              ┌────────┴────────┐
                              │  preprocess()    │  letterbox + BGR→RGB + 归一化 → ov::Tensor
                              │  inference()     │  infer_request_.infer()
                              │  postprocess()   │  decode_output() + nms()
                              └────────┬────────┘       此处继承为一个yolo_run
                                       │
                              get_nms_results() ──→ std::vector<Detection>
                                       │
                              show_yolo_result() ──→ 可视化
```

### 5.2 PP-OCR 检测管线

```text
输入图像 ──→ detect_det_ppocr::preprocess()
             │  缩放（保持32对齐）+ /255 + 减均值 + 除标准差 → ov::Tensor
             ↓
         detect_det_ppocr::inference()
             │  set_input_tensor → infer → get_output_tensor
             ↓
         detect_det_ppocr::postprocess()
             │  概率图 → 二值化 → 轮廓提取 → minAreaRect
             │  → 顺时针排序 → 坐标还原 → 排序输出
             ↓
         ocr_det_out (std::vector<OCRBox>)
             ↓
         [后续] detect_rec_ppocr → 文本识别（待实现）
```

---

## 6. 各文件详细说明

### 6.1 节点入口

| 文件 | 作用 | 当前状态 |
| ---- | ---- | -------- |
| `yolo_node.cpp` | YOLO 节点主入口。加载 JSON 配置 → 初始化 OpenVINO 模型 → ROS 主循环 | 已实现配置加载与模型初始化；主循环暂为 chatter 示例，待接入图像订阅+推理 |
| `ppocr_node.cpp` | PP-OCR 节点骨架。`image_transport` 订阅图像 → 推理 → 发布 JSON | 框架已搭建，`initModel`/`detectText` 为 TODO |

### 6.2 检测抽象层

| 文件 | 作用 |
| ---- | ---- |
| `detector.hpp` | 基类接口：纯虚函数 `preprocess/inference/postprocess`；图像缓存队列（海康×1 + USB×4）；互斥锁；配置参数 |
| `dtetctor.cpp` | 基类实现：`load_config()` 从 JSON 读取所有参数；`push_img()` 按相机 ID 入队并带锁更新；`show_yolo_result()` 在图像上画框+标签 |

### 6.3 YOLO 检测器（OpenVINO）

| 文件 | 作用 |
| ---- | ---- |
| `nuc_detect.hpp` | 声明 `detect_oponvino`，管理 `ov::Core/CompiledModel/InferRequest/Tensor`；声明 letterbox、decode、NMS 接口 |
| `nuc_detect.cpp` | `inference_init()`：读取 XML/BIN → reshape → 编译 → 创建推理请求 → 缓存输出维度。`preprocess()`：letterbox → BGR→RGB → 按精度(FP32/FP16/INT8/UINT8)写入 NCHW tensor。`inference()`：`infer()` + 获取输出。`decode_output()`：解析 YOLOv8 格式 `[1, 4+classes, anchors]` → 还原原图坐标。`nms()`：NMS 过滤（**当前注释状态，需启用**） |

### 6.4 PP-OCR 检测器（OpenVINO）

| 文件 | 作用 |
| ---- | ---- |
| `ocr_detect.hpp` | 声明三个类：`detect_det_ppocr`（文本区域检测）、`detect_rec_ppocr`（文本识别）、`detect_cls_ppocr`（方向分类） |
| `ocr_detect.cpp` | **det 模块已实现**：`preprocess()` — 动态缩放(32对齐) + ImageNet归一化 + HWC→NCHW。`inference()` — OpenVINO 推理。`postprocess()` — 概率图→二值化→轮廓→旋转矩形→顺时针排序→坐标还原→按阅读顺序排序。**rec/cls 模块待实现** |

### 6.5 通用数据结构 (`common_structs.h`)

| 结构体 | 用途 |
| ------ | ---- |
| `s_detector_params` | 模型路径、NCHW尺寸、图像参数、检测阈值、OCR 专用参数 |
| `Detection` | YOLO 检测结果：`bbox[4]` + `conf` + `class_id` |
| `Appconfig` | 顶层配置聚合：检测参数 + 海康参数 + USB参数×4 |
| `OCRBox` | OCR 文本框：4个顶点 `std::array<cv::Point2f, 4>` |
| `OCRRecResult` | OCR 识别结果：文本 + 置信度 |
| `OCRItem` | OCR 完整结果：检测框 + 识别结果 |
| `DetResizeMeta` | 预处理缩放元数据：原图/缩放尺寸、比例 |

### 6.6 配置文件 (`settings.json`)

| 配置段 | 关键字段 | 说明 |
| ------ | -------- | ---- |
| `path` | `openvino_xml_file_path`, `openvino_bin_file_path`, `ppocr_det/rec/cls_model_path` | 模型文件路径（**部署前必须填写**） |
| `NCHW` | `batch_size`, `C`, `W`, `H` | YOLO 模型输入尺寸，默认 `1×3×640×640` |
| `thresh` | `nms_thresh`, `bbox_conf_thresh`, `merge_thresh` | 检测阈值 |
| `nums` | `classes` | YOLO 类别数 |
| `hikcamera` / `usbcamera0~3` | `device_id`, `width`, `height`, `exposure` 等 | 相机参数 |

> **注意**：`load_config()` 中读取海康参数使用的键名是 `camera_0`，而 JSON 中实际键名为 `hikcamera`，需统一后才能正确加载。

### 6.7 相机模块

| 文件 | 作用 |
| ---- | ---- |
| `hikvision.hpp/cpp` | 海康工业相机封装（依赖 MVS SDK `/opt/MVS`）。`HikGrab` 类：初始化、取帧、回调  |
| `usbcam.hpp/cpp` | USB 摄像头封装（OpenCV `VideoCapture`）。`usb_camera` 类：初始化、取帧、双缓冲 |
| `grab_test.cpp` | 相机采集独立测试，可脱离 ROS 单独编译运行 |
| `camera/CMakeLists.txt` | 相机测试独立构建脚本，链接 `MvCameraControl` + OpenCV |

---

## 7. 构建与运行

### 7.1 编译

```bash
cd ~/toe26_dogvision

# 首次构建 / 全量构建
catkin_make -j8

# 指定包构建
catkin_make --pkg dogvision26 -j8

# 清理后重编（推荐标准变更后使用）
catkin_make clean && catkin_make -j8
```

### 7.2 加载环境

```bash
source devel/setup.bash
```

### 7.3 运行节点

```bash
# YOLO 检测节点
rosrun dogvision26 yolo_node

# PP-OCR 节点
rosrun dogvision26 ppocr_node
```

### 7.4 相机独立测试（无需 ROS）

```bash
cd src/dogvision26/src/camera
mkdir -p build && cd build
cmake .. && make -j4
./grab_test
```

---

## 8. 配置要点

### 8.1 部署前必做

1. **填写模型路径**：编辑 `src/dogvision26/src/detect/settings.json`，将 `path` 段的空白路径替换为实际模型文件路径
2. **统一 JSON 键名**：`load_config()` 读取 `camera_0`，`settings.json` 中为 `hikcamera`，需统一
3. **放置模型文件**：YOLO 模型放入 `src/data/yolo/`，PP-OCR 模型放入 `src/data/ppocr/`

### 8.2 C++ 标准

CMake 中已通过三级保障设定 C++17：

```cmake
add_compile_options(-std=c++17)          # 全局编译选项
set(CMAKE_CXX_STANDARD 17)              # CMake 标准变量
target_compile_features(... cxx_std_17)  # 目标级特性要求
```

若 VS Code IntelliSense 仍按旧标准解析，需在 `.vscode/c_cpp_properties.json` 中设置 `"cppStandard": "c++17"`。

---

## 9. 常见问题

| 问题 | 原因 | 解决 |
| ---- | ---- | ---- |
| 链接报 OpenVINO `undefined reference` | `yolo_node` 未链接 `openvino::runtime` | CMake 中已有兜底逻辑，确认 `find_package(OpenVINO)` 成功 |
| 链接报 JsonCpp `undefined reference` | JsonCpp 目标名因发行版不同 | CMake 中已依次尝试 `jsoncpp_lib` → `jsoncpp_static` → `${JSONCPP_LIBRARIES}` → `jsoncpp` |
| `std::clamp` 报错 | 编译器未按 C++17 解析 | 确认 `CMAKE_CXX_STANDARD 17` 且 clean 重编 |
| 初始化成功但无检测结果 | NMS 代码被注释 / 阈值过高 / 模型路径错误 | 启用 `nms()`、检查 `settings.json` |
| `Mate` 赋值报"不可修改左值" | `preprocess()` 被声明为 `const` | 去掉 `const` 限定符 |

---

## 10. 待完成事项

- [ ] `yolo_node.cpp`：接入 `image_transport` 订阅图像并触发推理闭环
- [ ] `nuc_detect.cpp`：启用 `nms()` 函数体
- [ ] `ocr_detect.cpp`：实现 `detect_rec_ppocr`（文本识别）和 `detect_cls_ppocr`（方向分类）
- [ ] `ppocr_node.cpp`：接入 `detect_det_ppocr` + `detect_rec_ppocr` 完成 OCR 全流程
- [ ] `settings.json` / `load_config()`：统一相机配置键名
- [ ] 补充 launch 文件，参数化模型路径和话题名
- [ ] 提高图像对比度可以将图像的颜色增强

---
