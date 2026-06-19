# dogvision_vision

ROS 2 相机采集、YOLO 目标检测、PP-OCR 文字识别与算术题识别视觉包。

---

## 1. 功能概述

本包提供四个独立节点和一个共享库，覆盖完整的视觉感知流水线：

| 可执行文件 | 节点名 | 功能 |
|---|---|---|
| `yolo_node` | `yolo_node` | 触发式单帧抓帧 → YOLO 推理 → 2×4 网格分配 → JSON 发布 |
| `yolo_accuracy_test_node` | `yolo_accuracy_test_node` | 连续实时 YOLO 推理 → 标注可视化 → 带标注视频录制 |
| `ppocr_node` | `ppocr_node` | 海康相机取帧 → 白屏定位并扩张 ROI → 可选灰度化 → PP-OCR 数学字符识别 → 表达式计算 → 多帧投票稳定 → JSON/YAML 输出 |
| `math_generator_node` | `math_generator_node` | 随机生成复合四则运算题 → 全屏渲染显示 → 追加写入 YAML |

---

## 2. 包结构

```
dogvision_vision/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   └── settings.json              # 全局配置文件（相机、模型路径、阈值、类别名）
├── data/
│   ├── img/                       # 测试图片存放目录（预留）
│   ├── math_generator/
│   │   └── math_results.yaml      # 数学题生成器输出
│   ├── ocr_output/
│   │   ├── ocr_results.yaml       # PPOCR test 模式输出
│   │   └── result.txt             # 辅助测试记录
│   ├── yolorun/                   # YOLO 单帧结果图保存目录（.jpg）
│   └── yolotest/                  # YOLO 准确性测试视频保存目录（.avi）
├── include/dogvision_vision/
│   ├── common_structs.h           # 共享数据结构（Detection, Appconfig, OCRBox 等）
│   ├── detector.hpp               # 检测器基类
│   ├── nuc_detect.hpp             # YOLO OpenVINO 检测器
│   ├── ocr_detect.hpp             # PP-OCR 检测/识别/分类器
│   ├── ocr_MultiFrameVoter.hpp    # 多帧滑动窗口投票器
│   ├── ocr_utils.hpp              # OCR 工具函数（裁剪、绘制、算术解析、鱼眼去畸变）
│   ├── yolo_utils.hpp             # YOLO 工具函数（网格分配、NMS、JSON 序列化、可视化）
│   ├── camera/
│   │   ├── hikvision.hpp          # 海康 MVS 相机封装
│   │   └── usbcam.hpp             # USB 相机封装（预留，当前节点中未使用）
│   └── math/
│       └── math_generator.hpp     # 数学题生成器
├── launch/
│   ├── math_generator.launch
│   ├── ppocr_test.launch
│   └── yolo_accuracy_test.launch
├── models/
│   ├── ppocr/                     # PP-OCRv4 模型、完整字典与数学字符白名单
│   └── yolo/                      # YOLO OpenVINO 模型文件（.xml / .bin）
├── test/
│   ├── ocr_multi_frame_voter_test.cpp
│   └── ocr_roi_decode_test.cpp
└── src/
    ├── camera/
    │   ├── hikvision.cpp
    │   └── usbcam.cpp
    ├── core/
    │   └── detector.cpp           # 基类实现 + 鱼眼去畸变映射初始化
    ├── math/
    │   └── math_generator.cpp
    ├── nodes/
    │   ├── yolo_node.cpp
    │   ├── yolo_accuracy_test_node.cpp
    │   ├── ppocr_node.cpp
    │   └── math_generator_node.cpp
    ├── ocr/
    │   ├── ocr_detect.cpp
    │   ├── ocr_MultiFrameVoter.cpp
    │   └── ocr_utils.cpp
    └── yolo/
        ├── nuc_detect.cpp
        └── yolo_utils.cpp
```

---

## 3. 依赖与构建

### 3.1 系统依赖

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  python3-colcon-common-extensions \
  libopencv-dev \
  libjsoncpp-dev \
  libusb-1.0-0-dev
```

### 3.2 OpenVINO（强制依赖）

需要 OpenVINO 2024.6+（本项目验证使用 2024.6.0 和 2025.4）。

```bash
# 下载并安装 OpenVINO
curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.6/linux/l_openvino_toolkit_ubuntu24_2024.6.0.17404.4c0f47d2335_x86_64.tgz --output openvino_2024.6.0.tgz
tar -xf openvino_2024.6.0.tgz
sudo mv l_openvino_toolkit_ubuntu24_2024.6.0.17404.4c0f47d2335_x86_64 /opt/intel/openvino_2024.6.0
```

每次构建/运行前加载环境：

```bash
source /opt/intel/openvino_2024.6.0/setupvars.sh
```

### 3.3 海康 MVS SDK（强制依赖）

从海康官网下载 Linux MVS SDK，安装到 `/opt/MVS/`，确保以下文件存在：

```
/opt/MVS/include/MvCameraControl.h
/opt/MVS/lib/64/libMvCameraControl.so
```

配置动态库路径：

```bash
echo "/opt/MVS/lib/64" | sudo tee /etc/ld.so.conf.d/hikvision_mvs.conf
sudo ldconfig
```

> 缺少 MVS SDK 时 CMake 配置阶段会直接 `FATAL_ERROR`。

### 3.4 构建

```bash
cd ~/toe26_dogvision
source /opt/ros/jazzy/setup.bash
source /opt/intel/openvino_2024.6.0/setupvars.sh

colcon build --packages-up-to dogvision_vision --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

source install/setup.bash
```

> 若系统默认 Python 来自 Conda，需显式指定 `-DPython3_EXECUTABLE=/usr/bin/python3` 避免 ROS2 消息生成失败。

---

## 4. 配置

所有视觉参数在 `config/settings.json` 中配置（JSON 格式）。模型路径相对于包安装目录的 `share/dogvision_vision/` 解析。

```jsonc
{
  // 海康相机参数
  "hikcamera": {
    "device_id": 0,
    "width": 1440,
    "height": 1080,
    "offset_x": 0,
    "offset_y": 0,
    "exposure": 12000,
    "gain": 9
  },

  // 鱼眼畸变系数 D（4 个参数，从标定结果读取）
  "lens_distortion": {
    "D": [0.01947, 0.02210, -0.04101, 0.02622]
  },

  // USB 相机参数（预留，当前未使用）
  "usbcamera0": { "device_id": 0, "width": 1920, "height": 1080, "FPS": 120 },
  "usbcamera1": { "device_id": 1, "width": 1920, "height": 1080, "FPS": 120 },
  "usbcamera2": { "device_id": 2, "width": 1920, "height": 1080, "FPS": 120 },
  "usbcamera3": { "device_id": 3, "width": 1920, "height": 1080, "FPS": 120 },

  // 模型路径（相对于包 share 目录）
  "path": {
    "openvino_bin_file_path": "models/yolo/yolo/m26325.bin",
    "openvino_xml_file_path": "models/yolo/yolo/m26325.xml",
    "ppocr_det_model_path": "models/ppocr/ch_PP-OCRv4_det_infer/inference.pdmodel",
    "ppocr_rec_model_path": "models/ppocr/ch_PP-OCRv4_rec_infer/inference.pdmodel",
    "ppocr_cls_model_path": "",                          // 文字方向分类（暂未使用）
    "ppocr_dict_path": "models/ppocr/Dict/ppocr_keys_v1.txt",
    "ppocr_allowed_chars_path": "models/ppocr/Dict/math_chars.txt"
  },

  // OCR ROI
  "ocr_roi": {
    "expand_ratio": 0.05,
    "use_grayscale": false
  },

  // YOLO 输入张量形状 NCHW
  "NCHW": { "batch_size": 1, "C": 3, "W": 640, "H": 640 },

  // 图像信息（用于配置输入类型）
  "img": { "type": 0 /* 0=RGB 1=BGR */, "width": 1920, "height": 1080 },

  // 阈值
  "thresh": {
    "nms_thresh": 0.6,         // NMS IoU 阈值
    "bbox_conf_thresh": 0.6,   // 边界框置信度阈值
    "merge_thresh": 0.8,       // 多帧合并阈值
    "max_wellid_distance": 8
  },

  // 类别数量与名称（最多 4 类）
  "nums": {
    "classes": 4,
    "cls0": "food",
    "cls1": "tool",
    "cls2": "medicine",
    "cls3": "instrument"
  }
}
```

OCR ROI 参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `expand_ratio` | `0.05` | 白屏矩形每侧扩张比例，必须大于等于 0 |
| `use_grayscale` | `false` | `false` 使用原始彩色 ROI，`true` 使用三通道灰度 ROI |

`ppocr_keys_v1.txt` 必须保留为官方完整字典，不能直接裁剪或重排，否则会破坏识别模型的类别索引。`math_chars.txt` 是解码白名单，当前允许：

```text
0-9  +  -  *  /  ×  ÷  =  .  (  )
```

启动时会校验 ROI 参数、完整字典、白名单字符以及模型输出类别数；配置不合法时会直接报错。

鱼眼去畸变内参矩阵 K 在 `src/core/detector.cpp` 中以硬编码常量定义（从 `fisheye_params.yaml` 标定结果提取）。如需修改，需同步更新该文件。

---

## 5. 节点详细说明

### 5.1 `yolo_node` — 触发式 YOLO 推理节点

**工作流程**：
1. 加载配置文件，初始化 YOLO OpenVINO 模型和海康相机。
2. 等待触发信号（话题 `/yolo/trigger` 收到 `"start_infer"` 或终端按 Enter）。
3. 触发后抓取一帧 → 可选鱼眼去畸变 → YOLO 推理 → NMS → 检测结果光栅排序 → 2×4 网格 K-means 分配。
4. 发布 JSON 结果到 `/yolo/result` 和 `/yolo/block_grid`。
5. 可选保存带标注的结果图（.jpg）到指定目录。
6. 可选打开 OpenCV 窗口显示结果。

**ROS 2 接口**：

| 接口 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/yolo/trigger` | `std_msgs/msg/String` | 订阅 | 发布 `"start_infer"` 触发一次推理 |
| `/yolo/result` | `std_msgs/msg/String` | 发布 | transient_local, JSON 格式检测结果 |
| `/yolo/block_grid` | `std_msgs/msg/String` | 发布 | transient_local, JSON 格式 2×4 类别网格 |

**参数**：

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `config_path` | string | `<share>/config/settings.json` | 配置文件路径 |
| `result_topic` | string | `/yolo/result` | 检测结果发布话题 |
| `show_window` | bool | false | 是否显示 OpenCV 可视化窗口 |
| `enable_undistort` | bool | true | 是否启用鱼眼去畸变 |
| `save_images` | bool | true | 是否保存结果图 |
| `enable_keyboard_trigger` | bool | true | 是否允许 Enter 触发；组合 launch 默认关闭 |
| `save_dir` | string | `<share>/data/yolorun` | 结果图保存目录 |

**启动方式**：

```bash
# 方式一：通过 vision.launch 启动（含 ppocr_node）
ros2 launch dogvision_bringup vision.launch

# 方式二：直接运行
ros2 run dogvision_vision yolo_node

# 触发推理
ros2 topic pub --once /yolo/trigger std_msgs/msg/String "{data: start_infer}"
# 或在 yolo_node 终端按 Enter
```

**输出示例**：

`/yolo/result`:
```json
{"detections":[{"pos_id":1,"class":"food","conf":0.8821,"bbox":[120.0,80.0,200.0,150.0]}]}
```

`/yolo/block_grid`:
```json
{"block":[["food","tool","null","null"],["medicine","null","null","null"]]}
```

---

### 5.2 `yolo_accuracy_test_node` — YOLO 准确性测试节点

**工作流程**：
1. 加载配置，初始化 YOLO 模型和海康相机。
2. 循环抓帧 → 去畸变 → YOLO 推理 → 可视化标注 → 写入视频文件。
3. 按 Q 或 ESC 退出，自动关闭视频文件。

**ROS 2 接口**：无（纯本地运行）。

**参数**：

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `config_path` | string | `<share>/config/settings.json` | 配置文件路径 |
| `enable_undistort` | bool | true | 是否启用去畸变 |
| `output_dir` | string | `<share>/data/yolotest` | 测试视频输出目录 |
| `video_fps` | double | 20.0 | 输出视频帧率 |
| `visual_nms_thresh` | double | 0.7 | 可视化 NMS 阈值（0~1），越高越容易保留同帧多个目标 |

**启动方式**：

```bash
ros2 launch dogvision_vision yolo_accuracy_test.launch
# 或直接运行
ros2 run dogvision_vision yolo_accuracy_test_node
```

视频文件命名格式：`yolo_accuracy_{毫秒时间戳}.avi`（MP4V 编码）。

---

### 5.3 `ppocr_node` — PP-OCR 算术题识别节点

**工作流程**：
1. 加载配置，初始化 PP-OCR 文本检测模型、文本识别模型、完整字典、数学字符白名单和海康相机。
2. 两种运行模式：

   **test 模式**（连续测试 + YAML 输出）：
   - 循环取帧 → 去畸变 → 定位白屏 → 四边扩张 ROI → 可选灰度化 → 文本检测 → 数学字符约束识别 → 表达式解析计算 → 多帧投票
   - 稳定结果发生变化时追加写入 `ocr_results.yaml`

   **production 模式**（触发式生产）：
   - 空闲等待 `/ocr/trigger` 话题或 Enter 触发
   - 触发后重置投票器，开始连续跟踪
   - 首个稳定结果通过 `/ocr/result` 发布 JSON，并通过 `/ocr/answer` 发布 `UInt8 mod4`
   - 发布一次后停止跟踪，等待下一次触发

**OCR ROI 流程**：

1. 使用 `find_math_proble()` 定位白屏外接矩形。
2. 左右各扩张白屏宽度的 `expand_ratio`，上下各扩张白屏高度的 `expand_ratio`。
3. 将扩张矩形裁剪到原图边界。
4. 根据 `use_grayscale` 选择原始彩色或三通道灰度 ROI。
5. 文本检测和识别仅处理该 ROI，不再使用原白屏掩码二次过滤。

不再执行 CLAHE、高斯模糊或二值化。最终检测框与文字仍绘制在原始彩色画面上。

启用 `show_ocr_roi` 后，会在 `"Math OCR ROI"` 窗口中显示实际送入 OCR 的图像。

**数学字符约束**：

- 识别模型仍使用完整官方字典维护类别索引。
- 解码时只在 CTC blank 和 `math_chars.txt` 对应类别中选择最大概率。
- 即使中文或其他无关字符的原始分数更高，也不会出现在最终 OCR 文本中。

**ROS 2 接口**：

| 接口 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/ocr/trigger` | `std_msgs/msg/String` | 订阅 | 任意内容触发 production 模式开始跟踪 |
| `/ocr/result` | `std_msgs/msg/String` | 发布 | transient_local, JSON 格式稳定识别结果 |
| `/ocr/answer` | `std_msgs/msg/UInt8` | 发布 | reliable/volatile，稳定结果的 `mod4`（0-3） |

**参数**：

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `config_path` | string | `<share>/config/settings.json` | 配置文件路径 |
| `mode` | string | `"production"` | 运行模式：`"test"` 或 `"production"` |
| `show_visual` | bool | true | 是否显示 `"Math OCR"` 整帧结果窗口 |
| `show_ocr_roi` | bool | false | 是否显示实际送入 OCR 的扩张 ROI |
| `enable_keyboard_trigger` | bool | true | production 模式是否允许 Enter 触发 |
| `yaml_path` | string | `<share>/data/ocr_output/ocr_results.yaml` | test 模式下 YAML 输出路径 |

**启动方式**：

```bash
# test 模式（连续运行）
ros2 launch dogvision_vision ppocr_test.launch

# production 模式（由 vision.launch 默认启动）
ros2 launch dogvision_bringup vision.launch     # 含 ppocr 和 yolo

# production 同时显示整帧结果和 OCR ROI
ros2 launch dogvision_bringup vision.launch \
  ppocr_show_visual:=true \
  ppocr_show_ocr_roi:=true

# 手动启动 production 模式
ros2 run dogvision_vision ppocr_node --ros-args -p mode:=production

# 触发 OCR 跟踪
ros2 topic pub --once /ocr/trigger std_msgs/msg/String "{data: start}"

# 查看结果
ros2 topic echo /ocr/result
ros2 topic echo /ocr/answer
```

**输出示例**（`/ocr/result`）：
```json
{"expr":"12+3*4","result":24,"mod4":0}
```

**test 模式 YAML 输出格式**：
```yaml
ocr_results:
  - id: 1
    question: "12+3*4"
    answer: 24
    mod4: 0
  - id: 2
    question: "15-6/2"
    answer: 12
    mod4: 0
```

**多帧投票机制**：
- 滑动窗口大小：10 帧
- 某表达式出现次数 ≥ 6 次
- 在有效帧中占比 ≥ 60%
- 同时满足以上条件才标记为"稳定结果"
- 连续 10 帧全为无效帧 → 稳定结果丢失

---

### 5.4 `math_generator_node` — 数学题生成节点

**工作流程**：
1. 启动时全屏显示黑色窗口。
2. 按固定间隔生成一道复合四则运算题（同时包含 `+ - * /` 四种运算符），结果必为整数。
3. 全屏渲染白底黑字题目图片并按间隔切换。
4. 每道题追加写入 YAML 文件。

**ROS 2 接口**：无（纯本地运行）。

**参数**：

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `yaml_path` | string | `<share>/data/math_generator/math_results.yaml` | YAML 输出路径 |
| `min_val` | int | 1 | 操作数最小值 |
| `max_val` | int | 100 | 操作数最大值（生成复合题时子操作数限制为 ≤20） |
| `interval` | int | 10 | 题目切换间隔（秒） |
| `canvas_width` | int | 2560 | 渲染画布宽度 |
| `canvas_height` | int | 1440 | 渲染画布高度 |

**启动方式**：

```bash
ros2 launch dogvision_vision math_generator.launch
# 或直接运行
ros2 run dogvision_vision math_generator_node
# 按间隔自动切换题目，按 Q 或 ESC 退出
```

**YAML 输出格式**：
```yaml
math_problems:
  - id: 1
    question: "7 - 12 / 2 * 15 + 6 = "
    answer: -77
    mod4: 3
```

---

## 6. 共享库 API 参考

### 6.1 核心数据结构 (`common_structs.h`)

| 类型 | 描述 |
|---|---|
| `s_detector_params` | 检测器参数聚合（模型路径、NCHW 尺寸、阈值、PPOCR 参数、类别名称） |
| `s_hikcamera_params` | 海康相机参数（device_id, width, height, offset, exposure） |
| `s_usbcamera_params` | USB 相机参数（预留） |
| `Appconfig` | 顶层应用配置，包含 `detect_config`, `hikcamera_config`, `usbcamera_config[4]` |
| `Detection` | YOLO 检测结果（bbox[4], conf, class_id） |
| `OCRBox` | OCR 四点文本框（pts[4]） |
| `OCRRecResult` | OCR 识别结果（text, score） |
| `OCRItem` | OCR 检测+识别组合 |
| `DetResizeMeta` | 检测预处理缩放元信息 |

### 6.2 `detector` 基类 (`detector.hpp`)

```cpp
class detector {
public:
    detector(Appconfig* config);
    virtual ~detector();

    void push_img(cv::Mat &giab_img, int cam_id);         // 将图像写入相机缓存
    void show_yolo_result(cv::Mat &show_img, const Detection &det); // 绘制 YOLO 结果
    void show_ocr_result(void);                            // OCR 可视化（占位）
    cv::Mat diatorion(cv::Mat &show_img);                  // 鱼眼去畸变
    bool yolo_run(cv::Mat &input_img, std::vector<Detection> &res); // 完整 YOLO 推理
    bool get_ocr_result(void);                             // 获取 OCR 结果（占位）
    void load_config(Appconfig& config, std::string json_file_path); // 从 JSON 加载配置

protected:
    virtual void preprocess(cv::Mat &src) = 0;
    virtual void inference() = 0;
    virtual void postprocess() = 0;
};
```

### 6.3 `detect_oponvino` — YOLO OpenVINO 检测器 (`nuc_detect.hpp`)

继承自 `detector`，使用 OpenVINO 2025 API 进行 YOLO 推理。

```cpp
class detect_oponvino : public detector {
public:
    detect_oponvino(Appconfig* config) : detector(config) {}

    bool inference_init(void);                                 // 加载编译模型
    bool yolo_deect_run(cv::Mat &input_img, std::vector<Detection> &res); // 单帧推理
    const std::vector<Detection>& get_nms_results() const;     // 获取缓存结果
};
```

内部使用 letterbox 预处理（保持宽高比填充），支持 FP32/FP16/INT8 精度模型。

### 6.4 PP-OCR 系列 (`ocr_detect.hpp`)

| 类 | 说明 | 实现状态 |
|---|---|---|
| `detect_det_ppocr` | 文本检测（DB 模型）：输入图像→输出四点文本框 | ✅ 已实现 |
| `detect_rec_ppocr` | 文本识别（CRNN/SVTR）：裁剪文本行→输出字符串+置信度 | ✅ 已实现 |
| `detect_cls_ppocr` | 文字方向分类：判断文本是否倒置 | ❌ 未实现 |

关键方法：

```cpp
// 文本检测
void detect_det_ppocr::load_model(const std::string& model_path, const std::string& device);
const std::vector<OCRBox>& get_det_boxes() const;

// 文本识别
void detect_rec_ppocr::load_model(const std::string& model_path, const std::string& device);
void detect_rec_ppocr::loda_dict(const std::string& dict_path);
void detect_rec_ppocr::load_allowed_chars(const std::string& allowed_chars_path);
std::vector<OCRRecResult> Decode(const ov::Tensor& logits);
void set_max_wh_ratio(float r);
```

识别器加载顺序必须为：模型 → 完整字典 → 数学字符白名单。`Decode()` 会校验输入张量必须为 FP32 `[batch, time, classes]`，且类别数必须与完整字典一致。

### 6.5 `OCRMultiFrameVoter` — 多帧投票器 (`ocr_MultiFrameVoter.hpp`)

```cpp
class OCRMultiFrameVoter {
public:
    static constexpr std::size_t kWindowSize = 10;     // 滑动窗口大小
    static constexpr std::size_t kMinOccurrences = 6;  // 最低出现次数
    static constexpr double kMinValidRatio = 0.60;     // 有效帧最低占比

    OCRVoteEvent update(const std::optional<OCRVoteResult>& frame_result);
    void reset();
    bool has_stable_result() const;
    const OCRVoteResult& stable_result() const;
    std::size_t frame_count() const;
    std::size_t valid_result_count() const;
};
```

事件类型：

| 事件 | 说明 |
|---|---|
| `OCRVoteEvent::None` | 稳定结果未变化 |
| `OCRVoteEvent::StableChanged` | 产生了新的稳定结果 |
| `OCRVoteEvent::StableLost` | 稳定结果丢失（窗口满且无有效帧） |

### 6.6 工具函数 (`yolo_utils.hpp`)

| 函数 | 说明 |
|---|---|
| `reset_grid(GridBlock&)` | 将 2×4 网格所有单元重置为 `"null"` |
| `class_name_of(int, vector<string>)` | 类别编号转名称 |
| `load_class_names(Appconfig)` | 从配置加载类别名称列表 |
| `cross_frame_nms(all_dets, iou_thresh, num_classes)` | 跨帧按类别 NMS 合并检测结果 |
| `sort_raster(dets&)` | 光栅顺序排序（从上到下，从左到右） |
| `assign_grid_kmeans(dets, class_names, block&)` | K-means 按行聚类后分配到 2×4 网格 |
| `build_result_json(dets, class_names)` | 检测结果序列化为 JSON |
| `build_grid_json(block)` | 网格序列化为 JSON |
| `format_grid_lines(block)` | 网格格式化为可读文本行 |
| `render_yolo_result_image(dets, frame, class_names)` | 绘制带标注的结果图 |
| `save_yolo_result_image(...)` | 保存结果图到文件 |
| `show_viz_image(...)` | 在 OpenCV 窗口显示结果图 |
| `run_single_detection(hik, cam_params, detector, enable_undistort, frame, dets)` | 取帧+推理一站式函数 |
| `collect_detections(hik, cam_params, detector, enable_undistort, last_frame, duration_sec)` | 持续取帧+推理收集结果 |

### 6.7 OCR 工具函数 (`ocr_utils.hpp`)

| 函数 | 说明 |
|---|---|
| `expand_ocr_roi(roi, image_size, ratio)` | 按比例扩张白屏矩形并裁剪到图像边界 |
| `prepare_ocr_roi(input, use_grayscale)` | 输出原始彩色或三通道灰度 OCR ROI |
| `crop_text_region(src, box)` | 透视变换裁剪四点文本框 |
| `draw_ocr_result(vis, box, rec)` | 在图像上绘制 OCR 框和识别标签 |
| `parse_simple_expr(text, result, expr_str)` | 从 OCR 文本中解析并计算四则运算表达式 |
| `show_result_window(expr_str, mod_result)` | 显示 OCR 算术结果窗口 |
| `find_math_proble(input, mask_out, white_s_max, white_v_min)` | 定位白底算术题区域（HSV 白色掩码+形态学+轮廓筛选） |
| `init_fisheye_undistort(image_width, image_height)` | 初始化鱼眼去畸变映射表 |
| `undistort_image(input)` | 对单帧图像执行鱼眼去畸变 |

`parse_simple_expr` 支持：
- 中文/全角运算符归一化（`×`→`*`, `÷`→`/`, `＋`→`+`, `－`→`-`）
- 括号运算
- 先乘除后加减
- 浮点数运算

`find_math_proble` 算法流程：
1. BGR→HSV，提取低饱和度+高明度白色掩码
2. 多级形态学处理（闭运算填充文字孔洞+开运算去除噪点）
3. 大核+小核双通道合并（兼顾近距离大目标和远距离小目标）
4. 轮廓检测，按面积(≥0.5%图像)、宽高比(0.3~8.0)、白色覆盖率(≥45%)筛选
5. 综合评分（面积×白色比例）取最优候选

### 6.8 `MathGenerator` — 数学题生成器 (`math_generator.hpp`)

```cpp
class MathGenerator {
public:
    MathGenerator(const std::string &yaml_path, int min_val = 1, int max_val = 100);
    std::tuple<std::string, int, int> generateProblem(); // 返回 (题目, 答案, 答案%4)
    cv::Mat renderImage(const std::string &text, int canvas_width, int canvas_height) const;
    void appendToYaml(const std::string &problem, int answer, int mod4);
};
```

生成的题目格式示例：`7 - 12 / 2 * 15 + 6 = `（同时包含 `+ - * /` 四种运算符，除法保证整除）。

### 6.9 `HikGrab` — 海康相机封装 (`hikvision.hpp`)

```cpp
class HikGrab {
public:
    HikGrab(s_camera_params param);
    bool get_one_frame(cv::Mat& img, int id);  // 获取一帧 BGR 图像
    void Hik_init();                             // 初始化相机并开始采集
    void Hik_end();                              // 停止采集释放资源
};
```

使用海康 MVS SDK 回调模式采集图像，内部进行 BayerRG→BGR 转换。

### 6.10 `usb_camera` — USB 相机封装 (`usbcam.hpp`)

```cpp
class usb_camera {
public:
    bool usb_camera_init(cv::VideoCapture &capture);
    void usb_camera_show_frame(void);
    bool usb_camera_get_frame(cv::VideoCapture &capture, cv::Mat &frame);
};
```

> 注意：当前所有节点均使用海康相机，USB 相机类为预留实现，未在节点中使用。

---

## 7. Launch 文件

### 7.1 `yolo_accuracy_test.launch`

启动 YOLO 准确性测试节点。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `config_path` | string | `<share>/config/settings.json` | 视觉配置文件 |
| `enable_undistort` | bool | true | 是否启用去畸变 |
| `output_dir` | string | `<share>/data/yolotest` | 测试视频输出目录 |
| `video_fps` | double | 20.0 | 输出视频帧率 |
| `visual_nms_thresh` | double | 0.7 | 可视化 NMS 阈值 |

```bash
ros2 launch dogvision_vision yolo_accuracy_test.launch
ros2 launch dogvision_vision yolo_accuracy_test.launch enable_undistort:=false video_fps:=30.0
```

### 7.2 `ppocr_test.launch`

启动 PPOCR 连续测试模式（test mode）。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `config_path` | string | `<share>/config/settings.json` | 视觉配置文件 |
| `show_visual` | bool | true | 是否显示整帧 OCR 结果窗口 |
| `show_ocr_roi` | bool | true | 是否显示实际送入 OCR 的扩张 ROI |
| `yaml_path` | string | `<share>/data/ocr_output/ocr_results.yaml` | 输出 YAML 路径 |

```bash
ros2 launch dogvision_vision ppocr_test.launch
ros2 launch dogvision_vision ppocr_test.launch show_visual:=false
ros2 launch dogvision_vision ppocr_test.launch \
  show_visual:=false show_ocr_roi:=true
```

### 7.3 `math_generator.launch`

启动数学题生成节点。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `yaml_path` | string | `<share>/data/math_generator/math_results.yaml` | YAML 输出路径 |
| `min_val` | int | 1 | 操作数最小值 |
| `max_val` | int | 100 | 操作数最大值 |
| `interval` | int | 10 | 切换间隔（秒） |
| `canvas_width` | int | 1920 | 渲染宽度 |
| `canvas_height` | int | 1080 | 渲染高度 |

```bash
ros2 launch dogvision_vision math_generator.launch
ros2 launch dogvision_vision math_generator.launch interval:=5 min_val:=1 max_val:=20
```

### 7.4 通过 `dogvision_bringup` 启动

```bash
# 全系统（含机械臂 + 视觉）
ros2 launch dogvision_bringup full_system.launch

# 仅视觉（yolo_node + ppocr_node production 模式）
ros2 launch dogvision_bringup vision.launch
```

详见 `dogvision_bringup` 包的 README。

---

## 8. 话题汇总

| 话题 | 类型 | 方向 | 说明 | 所属节点 |
|---|---|---|---|---|
| `/yolo/trigger` | `std_msgs/msg/String` | 订阅 | 发布 `"start_infer"` 触发单帧 YOLO 推理 | `yolo_node` |
| `/yolo/result` | `std_msgs/msg/String` | 发布 | YOLO 检测结果 JSON（transient_local） | `yolo_node` |
| `/yolo/block_grid` | `std_msgs/msg/String` | 发布 | 2×4 类别网格 JSON（transient_local） | `yolo_node` |
| `/ocr/trigger` | `std_msgs/msg/String` | 订阅 | 任意内容触发 PPOCR production 模式跟踪 | `ppocr_node` |
| `/ocr/result` | `std_msgs/msg/String` | 发布 | 稳定 OCR 算术结果 JSON（transient_local） | `ppocr_node` |
| `/ocr/answer` | `std_msgs/msg/UInt8` | 发布 | 单次稳定结果的 `mod4`（reliable/volatile） | `ppocr_node` |

```bash
# 查看话题列表
ros2 topic list

# 实时查看结果
ros2 topic echo /yolo/result
ros2 topic echo /yolo/block_grid
ros2 topic echo /ocr/result
ros2 topic echo /ocr/answer

# 手动触发
ros2 topic pub --once /yolo/trigger std_msgs/msg/String "{data: start_infer}"
ros2 topic pub --once /ocr/trigger std_msgs/msg/String "{data: start}"
```

---

## 9. 测试

### 9.1 单元测试

```bash
# 构建并启用测试
colcon build --packages-select dogvision_vision --cmake-args -DBUILD_TESTING=ON

# 运行全部 dogvision_vision 单元测试
ctest --test-dir build/dogvision_vision --output-on-failure

# 或分别直接运行
./build/dogvision_vision/ocr_multi_frame_voter_test
./build/dogvision_vision/ocr_roi_decode_test
```

`test/ocr_multi_frame_voter_test.cpp` 覆盖：

- 6/10 帧达标时触发 `StableChanged`
- 无效帧不计入占比分母
- 5 次出现不达标
- 稳定结果在 9 帧无效后仍保留，第 10 帧丢失
- 新稳定结果替换旧结果
- A-B-A 切换模式
- `reset()` 清空状态

`test/ocr_roi_decode_test.cpp` 覆盖：

- 5% ROI 扩张、边界裁剪和零扩张
- 负数扩张比例校验
- 彩色 ROI 原样传递
- 可选三通道灰度转换及尺寸保持
- 数学字符白名单加载及缺失字符检查
- 构造 CTC logits，验证无关字符分数更高时仍只输出数学字符
- CTC 重复折叠、blank 删除和字典类别数校验
- `(12+36)×5÷(18-9)=` 表达式归一化与计算回归

### 9.2 YOLO 准确性测试

```bash
ros2 launch dogvision_vision yolo_accuracy_test.launch
```

实时显示标注窗口，按 Q 或 ESC 退出，自动保存标注视频到 `data/yolotest/`。

### 9.3 PPOCR 连续测试

```bash
ros2 launch dogvision_vision ppocr_test.launch
```

持续运行 OCR 识别并将所有稳定结果变化写入 `data/ocr_output/ocr_results.yaml`。

---

## 10. 数据输出格式

### YOLO 结果 JSON (`/yolo/result`)

```json
{
  "detections": [
    {
      "pos_id": 1,
      "class": "food",
      "conf": 0.8821,
      "bbox": [120.0, 80.0, 200.0, 150.0]
    }
  ]
}
```

`bbox` 格式为 `[x, y, width, height]`（像素坐标，原图尺寸）。

### 2×4 网格 JSON (`/yolo/block_grid`)

```json
{
  "block": [
    ["food", "tool", "null", "null"],
    ["medicine", "null", "null", "null"]
  ]
}
```

### OCR 结果 JSON (`/ocr/result`)

```json
{
  "expr": "12+3*4",
  "result": 24,
  "mod4": 0
}
```

### 数学题生成器 YAML

```yaml
math_problems:
  - id: 1
    question: "7 - 12 / 2 * 15 + 6 = "
    answer: -77
    mod4: 3
```

### PPOCR 测试模式 YAML

```yaml
ocr_results:
  - id: 1
    question: "12+3*4"
    answer: 24
    mod4: 0
```

---

## 11. 快速启动速查

```bash
# 1. 加载环境
source /opt/ros/jazzy/setup.bash
source /opt/intel/openvino_2024.6.0/setupvars.sh
source ~/toe26_dogvision/install/setup.bash

# 2. 启动全系统
ros2 launch dogvision_bringup full_system.launch

# 或仅启动视觉
ros2 launch dogvision_bringup vision.launch

# 3. 触发 YOLO 推理
ros2 topic pub --once /yolo/trigger std_msgs/msg/String "{data: start_infer}"

# 4. 触发 OCR 跟踪
ros2 topic pub --once /ocr/trigger std_msgs/msg/String "{data: start}"

# 5. 查看结果
ros2 topic echo /yolo/result
ros2 topic echo /yolo/block_grid
ros2 topic echo /ocr/result

# 6. 单独启动测试
ros2 launch dogvision_vision yolo_accuracy_test.launch
ros2 launch dogvision_vision ppocr_test.launch
ros2 launch dogvision_vision math_generator.launch
```

---

## 12. 常见问题

**Q: 缺少 MVS SDK 编译失败？**
A: 确保已安装海康 MVS SDK 到 `/opt/MVS/`，且 `libMvCameraControl.so` 在链接器搜索路径中。

**Q: OpenVINO 模型加载失败？**
A: 确保 `settings.json` 中 `path` 的模型路径正确，且已执行 `source /opt/intel/openvino_2024.6.0/setupvars.sh`。

**Q: 相机取帧失败？**
A: 检查相机物理连接，确认设备号与配置一致。PPOCR 节点有自动重连机制（最多 5 次尝试）。

**Q: 鱼眼去畸变无效？**
A: 确认 `src/core/detector.cpp` 中 `ENABLE_FISHEYE_UNDISTORT` 宏已定义（默认启用），且 K/D 矩阵与标定结果一致。
