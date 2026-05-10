cmake_minimum_required(VERSION 3.0.2)
project(dogvision_arm)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  dogvision_msgs
)
find_package(Threads REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES dogvision_arm_lib
  CATKIN_DEPENDS roscpp std_msgs dogvision_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${LIBUSB_INCLUDE_DIRS}
)

# ---- 串口通信库 ----
add_library(dogvision_arm_lib
  src/arm_internation.cpp
)
target_compile_features(dogvision_arm_lib PUBLIC cxx_std_17)
add_dependencies(dogvision_arm_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dogvision_arm_lib
  ${catkin_LIBRARIES}
  Threads::Threads
    ${LIBUSB_LIBRARIES}
)

# ---- Arm_internation_node ----
add_executable(Arm_internation_node
  src/Arm_internation_node.cpp
)
target_compile_features(Arm_internation_node PRIVATE cxx_std_17)
add_dependencies(Arm_internation_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(Arm_internation_node
  dogvision_arm_lib
  ${catkin_LIBRARIES}
  Threads::Threads
  ${LIBUSB_LIBRARIES}
)

# ---- arm_cmd_terminal_node ----
add_executable(arm_cmd_terminal_node
  src/arm_cmd_terminal_node.cpp
)
target_compile_features(arm_cmd_terminal_node PRIVATE cxx_std_17)
add_dependencies(arm_cmd_terminal_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(arm_cmd_terminal_node
  ${catkin_LIBRARIES}
  Threads::Threads
  ${LIBUSB_LIBRARIES}
)

# ---- arm_mission_node ----
add_executable(arm_mission_node
  src/arm_mission_node.cpp
)
target_compile_features(arm_mission_node PRIVATE cxx_std_17)
add_dependencies(arm_mission_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(arm_mission_node
  ${catkin_LIBRARIES}
  Threads::Threads
)
