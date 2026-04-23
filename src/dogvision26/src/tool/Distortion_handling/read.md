# Distortion Handling 使用说明

本工具用于通过 OpenCV 采集棋盘格图像并计算相机标定参数（相机内参矩阵与畸变系数）。

## 1. 构建

在工作空间根目录执行：

```bash
catkin_make --pkg dogvision26
```

## 2. 运行

默认运行：

```bash
rosrun dogvision26 diatortion_handing
```

带参数运行：

```bash
rosrun dogvision26 diatortion_handing <camera_id> <board_cols> <board_rows> <square_size> <min_samples> <output_path>
```

参数说明：

- `camera_id`：相机编号（默认 `0`）
- `board_cols`：棋盘格内角点列数（默认 `9`）
- `board_rows`：棋盘格内角点行数（默认 `6`）
- `square_size`：棋盘格单格边长（单位米，默认 `0.025`）
- `min_samples`：最少采样帧数（默认 `12`）
- `output_path`：标定结果输出路径（默认 `camera_calibration.yaml`）

示例：

```bash
rosrun dogvision26 diatortion_handing 0 9 6 0.025 15 /tmp/camera_calibration.yaml
```

## 3. 窗口按键

程序运行后会打开实时画面窗口：

- `c`：当检测到棋盘格时，采集当前帧角点
- `s`：当采样数达到 `min_samples` 后执行标定并保存结果
- `q` 或 `Esc`：退出程序

## 4. 输出内容

标定完成后会输出：

- `camera_matrix`：相机内参矩阵
- `distortion_coefficients`：畸变系数
- `reprojection_error`：重投影误差

保存到 YAML 文件中（默认 `camera_calibration.yaml`）。

## 5. 使用建议

- 棋盘格要尽量覆盖画面不同位置（中心、边缘、四角）。
- 拍摄时尽量包含不同角度和距离，提升标定稳定性。
- 若重投影误差较大，建议增加采样数量并重新标定。
