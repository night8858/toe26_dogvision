# =============================================================================
# dogvision_vision — ROS2 相机 + YOLO + PPOCR + 视觉测试包
# =============================================================================
# 本包提供以下功能模块：
#   - 海康 MVS 工业相机 / USB 相机采集
#   - YOLO OpenVINO 实时检测
#   - PP-OCRv4 文本检测与识别（数学算术题）
#   - 多帧滑动窗口投票器
#   - 数学题生成与渲染
#   - 准确性测试视频录制
#
# 依赖：
#   - OpenVINO 2025 Runtime
#   - OpenCV 4.x
#   - jsoncpp
#   - Hikvision MVS SDK（/opt/MVS）
#   - libusb-1.0
# =============================================================================

cmake_minimum_required(VERSION 3.8)
project(dogvision_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── ROS2 基础依赖 ──
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# ── 第三方库依赖 ──
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

# ── 海康 MVS SDK 路径 ──
find_path(MVS_INCLUDE_DIR
  NAMES MvCameraControl.h
  PATHS /opt/MVS/include
  NO_DEFAULT_PATH
)
find_library(MVS_LIB
  NAMES MvCameraControl
  PATHS /opt/MVS/lib /opt/MVS/lib/64 /opt/MVS/lib/32
  NO_DEFAULT_PATH
)

if(NOT MVS_INCLUDE_DIR OR NOT MVS_LIB)
  message(FATAL_ERROR "Hikvision MVS SDK is required. Expected /opt/MVS/include/MvCameraControl.h and MvCameraControl library.")
endif()

# ── 核心库：包含所有非节点模块 ──
add_library(dogvision_vision_lib
  src/camera/camera_source.cpp   # settings.json 驱动的相机适配层
  src/camera/hikvision.cpp       # 海康 MVS 相机封装
  src/camera/usbcam.cpp          # USB 相机封装
  src/core/detector.cpp          # 检测器基类（加载配置、鱼眼去畸变）
  src/math/math_generator.cpp    # 数学题生成器
  src/ocr/ocr_MultiFrameVoter.cpp  # OCR 多帧投票器
  src/ocr/ocr_detect.cpp         # PPOCR 检测/识别/分类流水线
  src/ocr/ocr_utils.cpp          # OCR 工具函数
  src/yolo/nuc_detect.cpp        # YOLO OpenVINO 推理引擎
  src/yolo/yolo_utils.cpp        # YOLO 结果处理工具函数
)
target_compile_features(dogvision_vision_lib PUBLIC cxx_std_17)
target_include_directories(dogvision_vision_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${JSONCPP_INCLUDE_DIRS}
  ${LIBUSB_INCLUDE_DIRS}
  ${MVS_INCLUDE_DIR}
)
target_link_libraries(dogvision_vision_lib
  ${OpenCV_LIBRARIES}
  ${JSONCPP_LIBRARIES}
  ${LIBUSB_LIBRARIES}
  ${MVS_LIB}
  Threads::Threads
)

# OpenVINO 库链接（兼容 find_package 和 pkg_check_modules 两种方式）
if(TARGET openvino::runtime)
  target_link_libraries(dogvision_vision_lib openvino::runtime)
else()
  target_link_libraries(dogvision_vision_lib ${OpenVINO_LIBRARIES})
endif()

# ── 可执行目标：YOLO 单帧检测节点 ──
add_executable(yolo_node
  src/nodes/yolo_node.cpp
)
target_link_libraries(yolo_node dogvision_vision_lib Threads::Threads)
ament_target_dependencies(yolo_node rclcpp std_msgs ament_index_cpp)

# ── 可执行目标：YOLO 准确性测试节点（视频录制） ──
add_executable(yolo_accuracy_test_node
  src/nodes/yolo_accuracy_test_node.cpp
)
target_link_libraries(yolo_accuracy_test_node dogvision_vision_lib Threads::Threads)
ament_target_dependencies(yolo_accuracy_test_node rclcpp ament_index_cpp)

# ── 可执行目标：PPOCR 算术题识别节点 ──
add_executable(ppocr_node
  src/nodes/ppocr_node.cpp
)
target_include_directories(ppocr_node PRIVATE ${JSONCPP_INCLUDE_DIRS})
target_link_libraries(ppocr_node dogvision_vision_lib Threads::Threads ${JSONCPP_LIBRARIES})
ament_target_dependencies(ppocr_node rclcpp std_msgs ament_index_cpp)

# ── 可执行目标：数学题生成节点 ──
add_executable(math_generator_node
  src/nodes/math_generator_node.cpp
)
target_link_libraries(math_generator_node dogvision_vision_lib Threads::Threads)
ament_target_dependencies(math_generator_node rclcpp ament_index_cpp)

# ── 单元测试 ──
if(BUILD_TESTING)
  enable_testing()
  add_executable(ocr_multi_frame_voter_test
    test/ocr_multi_frame_voter_test.cpp
  )
  target_link_libraries(ocr_multi_frame_voter_test dogvision_vision_lib)
  add_test(NAME ocr_multi_frame_voter_test COMMAND ocr_multi_frame_voter_test)

  add_executable(ocr_math_filter_decode_test
    test/ocr_math_filter_decode_test.cpp
  )
  target_compile_definitions(ocr_math_filter_decode_test PRIVATE
    DOGVISION_VISION_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(ocr_math_filter_decode_test dogvision_vision_lib)
  add_test(NAME ocr_math_filter_decode_test COMMAND ocr_math_filter_decode_test)
endif()

# ── 安装规则 ──
install(TARGETS
  dogvision_vision_lib           # 核心库
  yolo_node                      # YOLO 检测节点
  yolo_accuracy_test_node        # YOLO 准确性测试节点
  ppocr_node                     # PPOCR 识别节点
  math_generator_node            # 数学题生成节点
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
# 安装头文件
install(DIRECTORY include/
  DESTINATION include
)
# 安装配置文件、模型和数据
install(DIRECTORY config models data
  DESTINATION share/${PROJECT_NAME}
)
# 安装启动脚本
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.launch"
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include ${JSONCPP_INCLUDE_DIRS} ${LIBUSB_INCLUDE_DIRS} ${MVS_INCLUDE_DIR})
ament_export_libraries(dogvision_vision_lib ${JSONCPP_LIBRARIES} ${LIBUSB_LIBRARIES} ${MVS_LIB})
ament_export_dependencies(OpenCV OpenVINO)
ament_package()
