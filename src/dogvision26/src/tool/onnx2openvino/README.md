# onnx2openvino

将 ONNX 模型转换为 OpenVINO IR（`.xml` + `.bin`）的小工具。

## 1. 构建

在 catkin 工作空间根目录执行：

```bash
catkin_make
```

构建完成后，可执行文件通常位于：

- `devel/lib/dogvision26/onnx2openvino`

## 2. 使用方法

```bash
onnx2openvino <input.onnx> <output.xml | output_dir>
```

- `input.onnx`：输入 ONNX 模型文件
- 第二个参数可为：
  - 目标 `xml` 文件路径（例如 `./model.xml`）
  - 目标输出目录（例如 `./openvino_model`）

### 示例 1：指定 xml 文件

```bash
./devel/lib/dogvision26/onnx2openvino ./model.onnx ./ir/model.xml
```

输出：

- `./ir/model.xml`
- `./ir/model.bin`

### 示例 2：指定输出目录

```bash
./devel/lib/dogvision26/onnx2openvino ./model.onnx ./ir/
```

若输入是 `model.onnx`，输出为：

- `./ir/model.xml`
- `./ir/model.bin`

## 3. 常见问题

- 若提示无法找到 OpenVINO 相关库，请先确认系统已安装 OpenVINO Runtime，并且 CMake 可找到 `OpenVINOConfig.cmake`。
- 若模型转换失败，请检查 ONNX 文件是否有效、是否受当前 OpenVINO 版本支持。
