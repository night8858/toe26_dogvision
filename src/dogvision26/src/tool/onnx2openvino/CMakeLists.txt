add_executable(onnx2openvino
	onnx2openvino.cpp
)

target_compile_features(onnx2openvino PRIVATE cxx_std_17)

if(TARGET openvino::runtime)
	target_link_libraries(onnx2openvino openvino::runtime)
else()
	target_link_libraries(onnx2openvino ${OpenVINO_LIBRARIES})
endif()

add_dependencies(onnx2openvino ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
