cmake_minimum_required(VERSION 3.8)
project(dogvision_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

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

add_library(dogvision_vision_lib
  src/camera/hikvision.cpp
  src/camera/usbcam.cpp
  src/core/detector.cpp
  src/math/math_generator.cpp
  src/ocr/ocr_MultiFrameVoter.cpp
  src/ocr/ocr_detect.cpp
  src/ocr/ocr_utils.cpp
  src/yolo/nuc_detect.cpp
  src/yolo/yolo_utils.cpp
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

if(TARGET openvino::runtime)
  target_link_libraries(dogvision_vision_lib openvino::runtime)
else()
  target_link_libraries(dogvision_vision_lib ${OpenVINO_LIBRARIES})
endif()

add_executable(yolo_node
  src/nodes/yolo_node.cpp
)
target_link_libraries(yolo_node dogvision_vision_lib Threads::Threads)
ament_target_dependencies(yolo_node rclcpp std_msgs ament_index_cpp)

add_executable(yolo_accuracy_test_node
  src/nodes/yolo_accuracy_test_node.cpp
)
target_link_libraries(yolo_accuracy_test_node dogvision_vision_lib Threads::Threads)
ament_target_dependencies(yolo_accuracy_test_node rclcpp ament_index_cpp)

add_executable(ppocr_node
  src/nodes/ppocr_node.cpp
)
target_include_directories(ppocr_node PRIVATE ${JSONCPP_INCLUDE_DIRS})
target_link_libraries(ppocr_node dogvision_vision_lib Threads::Threads ${JSONCPP_LIBRARIES})
ament_target_dependencies(ppocr_node rclcpp std_msgs ament_index_cpp)

add_executable(math_generator_node
  src/nodes/math_generator_node.cpp
)
target_link_libraries(math_generator_node dogvision_vision_lib Threads::Threads)
ament_target_dependencies(math_generator_node rclcpp ament_index_cpp)

if(BUILD_TESTING)
  enable_testing()
  add_executable(ocr_multi_frame_voter_test
    test/ocr_multi_frame_voter_test.cpp
  )
  target_link_libraries(ocr_multi_frame_voter_test dogvision_vision_lib)
  add_test(NAME ocr_multi_frame_voter_test COMMAND ocr_multi_frame_voter_test)
endif()

install(TARGETS
  dogvision_vision_lib
  yolo_node
  yolo_accuracy_test_node
  ppocr_node
  math_generator_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY config models data
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.launch"
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include ${JSONCPP_INCLUDE_DIRS} ${LIBUSB_INCLUDE_DIRS} ${MVS_INCLUDE_DIR})
ament_export_libraries(dogvision_vision_lib ${JSONCPP_LIBRARIES} ${LIBUSB_LIBRARIES} ${MVS_LIB})
ament_export_dependencies(OpenCV OpenVINO)
ament_package()
