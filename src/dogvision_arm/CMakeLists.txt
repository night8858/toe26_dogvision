cmake_minimum_required(VERSION 3.8)
project(dogvision_arm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "dogvision_arm: protocol = BB/4DOF")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

add_library(dogvision_arm_lib
  src/arm_internation.cpp
  src/arm_internation_bb.cpp
  src/arm_internation_cc.cpp
)
target_compile_features(dogvision_arm_lib PUBLIC cxx_std_17)
target_include_directories(dogvision_arm_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${LIBUSB_INCLUDE_DIRS}
)
target_link_libraries(dogvision_arm_lib
  Threads::Threads
  ${LIBUSB_LIBRARIES}
)

add_executable(arm_internation_node
  src/arm_internation_node.cpp
)
target_link_libraries(arm_internation_node dogvision_arm_lib Threads::Threads ${LIBUSB_LIBRARIES})
ament_target_dependencies(arm_internation_node rclcpp std_msgs)

add_executable(arm_cmd_terminal_node
  src/arm_cmd_terminal_node.cpp
)
target_link_libraries(arm_cmd_terminal_node Threads::Threads)
ament_target_dependencies(arm_cmd_terminal_node rclcpp std_msgs)

add_executable(arm_mission_node
  src/arm_mission_node.cpp
)
target_include_directories(arm_mission_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(arm_mission_node Threads::Threads)
ament_target_dependencies(arm_mission_node rclcpp std_msgs)

if(BUILD_TESTING)
  enable_testing()
  add_executable(answer_frame_test
    test/answer_frame_test.cpp
  )
  target_link_libraries(answer_frame_test dogvision_arm_lib Threads::Threads)
  add_test(NAME answer_frame_test COMMAND answer_frame_test)

  add_executable(mission_command_test
    test/mission_command_test.cpp
  )
  target_include_directories(mission_command_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  add_test(NAME mission_command_test COMMAND mission_command_test)
endif()

install(TARGETS
  dogvision_arm_lib
  arm_internation_node
  arm_cmd_terminal_node
  arm_mission_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
  PATTERN "*.in" EXCLUDE
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.launch"
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(dogvision_arm_lib)
ament_export_dependencies(rclcpp std_msgs)
ament_package()
