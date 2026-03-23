# dogvision26 使用说明与结构脉络

## 1. 包定位

`dogvision26` 是一个 专用于toe2026足式机器人任务赛的ROS1（catkin）视觉包，目标是实现：

- YOLO 检测（OpenVINO 推理）--即识别算术题
- PPOCR 文本识别（当前为骨架）--即识别和先验物块类型
- 相机采集（海康 + USB）

当前代码中，YOLO 推理能力主要集中在 `detect/yolo_detector`，节点层（`yolo_node.cpp`）还在从“初始化”向“完整在线订阅推理”过渡。

---

## 2. 目录结构总览

```text
dogvision26/
├── CMakeLists.txt
├── package.xml
├── README.md
├── include/dogvision26/
└── src/
		├── common_structs.h
		├── yolo_node.cpp
		├── ppocr_node.cpp
		├── camera/
		│   ├── CMakeLists.txt
		│   ├── grab_test.cpp
		│   ├── hikvision.cpp/.hpp
		│   └── usbcam.cpp/.hpp
		├── detect/
		│   ├── detector.hpp
		│   ├── dtetctor.cpp
		│   ├── settings.json
		│   ├── ppocr_detector/
		│   │   ├── preprocess.cpp
		│   │   └── preprocess.h
		│   └── yolo_detector/
		│       ├── nuc_detect.cpp
		│       └── nuc_detect.hpp
		└── data/
				├── yolo/
				└── ppocr/
```

---

## 3. 核心数据流（目标流程）

完整流程建议按如下链路理解：

1. 图像输入（ROS topic 或相机模块取图）
2. 进入 `detector` 基类缓存（`push_img`）
3. `detect_oponvino` 预处理（letterbox、颜色与精度转换、NCHW 填充）
4. OpenVINO 推理（`infer_request_.infer()`）
5. 后处理（decode + NMS）
6. 将 `Detection` 结果发布为 ROS 消息，或做可视化输出

当前实现状态：

- `nuc_detect.cpp`：已实现 3/4/5 的主要逻辑（NMS 代码目前注释状态）
- `yolo_node.cpp`：已实现配置加载和模型初始化，在线订阅推理链路待完整接入

---

## 4. 关键文件作用与用法

### 4.1 构建与依赖

#### `CMakeLists.txt`

作用：

- 声明 catkin 依赖（`roscpp`、`image_transport`、`cv_bridge` 等）
- 查找 OpenCV/OpenVINO/JsonCpp
- 构建可执行程序：`ppocr_node` 与 `yolo_node`

用法要点：

- `yolo_node` 需要链接：
	- OpenVINO runtime（`openvino::runtime` 或 `${OpenVINO_LIBRARIES}`）
	- JsonCpp（不同发行版目标名可能不同）
- 当前脚本已采用“多分支兜底”方式链接 JsonCpp。

#### `package.xml`

作用：

- 描述 ROS 包元信息与 catkin 依赖。

建议：

- 可补充系统依赖说明（README 中写清 OpenVINO/JsonCpp 安装方式）。

---

### 4.2 节点入口

#### `src/yolo_node.cpp`

作用：

- YOLO 节点主入口。
- 当前已做：加载 `settings.json`、初始化 `detect_oponvino` 模型。

建议用法（推荐改造方向）：

- 使用 `image_transport` 订阅图像话题。
- 回调中执行：
	- `detector_ov.push_img(frame, cam_id)`
	- `detector_ov.preprocess()`
	- `detector_ov.inference()`
	- `detector_ov.postprocess()`
	- `detector_ov.get_nms_results()` 发布检测结果。

#### `src/ppocr_node.cpp`

作用：

- PPOCR 节点骨架，具备标准图像订阅 + 字符串发布结构。

现状：

- `initModel` 和 `detectText` 仍为 TODO。

---

### 4.3 检测抽象层

#### `src/detect/detector.hpp`

作用：

- 定义检测器基类接口：`preprocess/inference/postprocess`。
- 提供图像缓存、互斥锁、配置参数、结果绘制接口。

#### `src/detect/dtetctor.cpp`

作用：

- 实现 JSON 配置加载（`load_config`）。
- 实现 `push_img`（按相机 ID 入队并更新缓存）。
- 提供检测框绘制函数 `show_yolo_result`。

注意：

- 配置读取里使用了 `camera_0` 键，而 `settings.json` 里是 `hikcamera`，建议统一命名。

---

### 4.4 YOLO OpenVINO 实现

#### `src/detect/yolo_detector/nuc_detect.hpp`

作用：

- 声明 `detect_oponvino`（继承 `detector`）。
- 管理 OpenVINO 资源：`ov::Core`、`CompiledModel`、`InferRequest`、输入输出 Tensor。
- 声明后处理与结果读取接口。

#### `src/detect/yolo_detector/nuc_detect.cpp`

作用：

- `inference_init`：读取模型、reshape、创建 infer request、缓存输出维度。
- `preprocess`：letterbox、BGR->RGB、按输入精度写入 tensor。
- `inference`：执行推理并取输出。
- `decode_output`：解析 YOLO 输出并映射回原图坐标。
- `nms`：NMS 过滤（目前为注释状态，需启用）。

使用前提：

- `settings.json` 中 `openvino_xml_file_path/openvino_bin_file_path` 必须为有效路径。

---

### 4.5 通用结构与配置

#### `src/common_structs.h`

作用：

- 定义全局结构：
	- `s_detector_params`
	- `Detection`
	- `Appconfig`
	- 相机参数结构体等

#### `src/detect/settings.json`

作用：

- 集中配置模型路径、输入尺寸、阈值、类别数、相机参数。

重点字段：

- `path.openvino_xml_file_path`
- `path.openvino_bin_file_path`
- `NCHW`（模型输入尺寸）
- `thresh`（`bbox_conf_thresh`、`nms_thresh`）
- `nums.classes`

---

### 4.6 相机模块

#### `src/camera/hikvision.hpp/.cpp`

作用：

- 海康工业相机封装（依赖 MVS SDK）。

#### `src/camera/usbcam.hpp/.cpp`

作用：

- USB 摄像头采集封装（基于 OpenCV VideoCapture）。

#### `src/camera/grab_test.cpp`

作用：

- 相机采集独立测试程序。

---

## 5. 构建与运行（推荐步骤）

在工作空间根目录（`toe26_dogvision`）执行：

```bash
catkin_make --pkg dogvision26 -j8
source devel/setup.bash
```

运行 YOLO 节点：

```bash
rosrun dogvision26 yolo_node
```

运行 PPOCR 节点：

```bash
rosrun dogvision26 ppocr_node
```

如果要走 ROS 图像输入链路，需保证有图像话题（例如 `/camera/image_raw`）正在发布。

---

## 6. 常见问题与排查

### 6.1 链接时报 OpenVINO undefined reference

原因通常是 `yolo_node` 未正确链接 OpenVINO runtime。

排查方向：

- 优先链接 `openvino::runtime`
- 若目标不存在，回退 `${OpenVINO_LIBRARIES}`

### 6.2 链接时报 JsonCpp undefined reference

原因通常是 JsonCpp 目标名在系统上不一致。

排查方向：

- 依次尝试 `jsoncpp_lib`、`jsoncpp_static`、`${JSONCPP_LIBRARIES}`、`jsoncpp`

### 6.3 初始化成功但无检测结果

重点检查：

- `settings.json` 模型路径是否正确
- `bbox_conf_thresh` 是否过高
- `nms()` 是否实际启用
- 节点是否真的把图像送入 `preprocess/inference/postprocess` 链路

---

## 7. 建议的最小闭环改造

优先完成以下三项可获得“可在线推理”的闭环：

1. 将 `yolo_node.cpp` 改成 `image_transport` 订阅图像并触发推理。
2. 启用 `nuc_detect.cpp` 中 `nms()` 逻辑，输出最终 `nms_results_`。
3. 统一 `settings.json` 与 `load_config()` 的键名（`hikcamera`/`camera_0`）。

完成后，包即可形成“订阅图像 -> 推理 -> 发布结果”的稳定主流程。

