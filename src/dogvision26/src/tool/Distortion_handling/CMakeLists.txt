add_executable(diatortion_handing
  diatortion_handing.cpp
)

target_compile_features(diatortion_handing PRIVATE cxx_std_17)

target_link_libraries(diatortion_handing
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_dependencies(diatortion_handing ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})