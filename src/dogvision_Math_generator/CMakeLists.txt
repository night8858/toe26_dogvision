cmake_minimum_required(VERSION 3.0.2)
project(dogvision_Math_generator)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  std_msgs
)

find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES dogvision_Math_generator_lib
  CATKIN_DEPENDS roscpp roslib std_msgs
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# ---- 数学问题生成器 ----
add_library(dogvision_Math_generator_lib
  src/math_generator.cpp
)

target_compile_features(dogvision_Math_generator_lib PUBLIC cxx_std_17)
add_dependencies(dogvision_Math_generator_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dogvision_Math_generator_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# ---- 可执行节点 ----
add_executable(math_generator_node
  src/math_generator_node.cpp
)

target_compile_features(math_generator_node PUBLIC cxx_std_17)
add_dependencies(math_generator_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(math_generator_node
  dogvision_Math_generator_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

