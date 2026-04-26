cmake_minimum_required(VERSION 3.0.2)
project(dogvision_vision)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  std_msgs
  dogvision_camera
)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES dogvision_vision_lib
  CATKIN_DEPENDS roscpp roslib std_msgs dogvision_camera
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${OpenVINO_INCLUDE_DIRS}
  ${JSONCPP_INCLUDE_DIRS}
)

# ---- 视觉检测核心库 ----
add_library(dogvision_vision_lib
  src/detector.cpp
  src/ocr_detect.cpp
)
target_compile_features(dogvision_vision_lib PUBLIC cxx_std_17)
add_dependencies(dogvision_vision_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dogvision_vision_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

if(TARGET openvino::runtime)
  target_link_libraries(dogvision_vision_lib openvino::runtime)
else()
  target_link_libraries(dogvision_vision_lib ${OpenVINO_LIBRARIES})
endif()

if(TARGET jsoncpp_lib)
  target_link_libraries(dogvision_vision_lib jsoncpp_lib)
elseif(TARGET jsoncpp_static)
  target_link_libraries(dogvision_vision_lib jsoncpp_static)
elseif(JSONCPP_FOUND)
  target_link_libraries(dogvision_vision_lib ${JSONCPP_LIBRARIES})
else()
  target_link_libraries(dogvision_vision_lib jsoncpp)
endif()

# ---- ppocr_node ----
add_executable(ppocr_node
  src/ppocr_node.cpp
  src/ocr_utils.cpp
)
target_compile_features(ppocr_node PRIVATE cxx_std_17)
add_dependencies(ppocr_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(ppocr_node
  dogvision_vision_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Threads::Threads
)

# ---- yolo_node ----
add_executable(yolo_node
  src/yolo_node.cpp
  src/yolo_utils.cpp
  src/nuc_detect.cpp
)
target_compile_features(yolo_node PRIVATE cxx_std_17)
add_dependencies(yolo_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(yolo_node
  dogvision_vision_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Threads::Threads
)

if(TARGET openvino::runtime)
  target_link_libraries(yolo_node openvino::runtime)
else()
  target_link_libraries(yolo_node ${OpenVINO_LIBRARIES})
endif()
