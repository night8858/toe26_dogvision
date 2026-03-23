#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include <sstream>
#include <string>
#include <vector>

#include "nuc_detect.hpp"
#include "detector.hpp"

using namespace std;


Appconfig config;			 // 全局配置对象，供 detector_ov 使用
detect_oponvino detector_ov(&config); // 传入配置对象

string json_config_path = "/home/toe/toe26_dogvision/src/dogvision26/src/detect/settings.json"; // 配置文件路径

int main(int argc, char **argv)
{
	// loda区
	detector_ov.load_config(config, json_config_path); // 从配置文件加载参数
	// 模型初始化
	if (!detector_ov.inference_init())
	{
		cerr << "Failed to initialize YOLO detector!" << endl;
		return -1;
	}
	ros::init(argc, argv, "yolo_node");
	ros::NodeHandle n;
	ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);

	ros::Rate loop_rate(10);

	int count = 0;
	while (ros::ok())
	{

		std_msgs::String msg;
		std::stringstream ss;
		ss << "hello world " << count;
		msg.data = ss.str();
		ROS_INFO("%s", msg.data.c_str());
		chatter_pub.publish(msg);

		ros::spinOnce();

		loop_rate.sleep();
		++count;
	}
	ros::spin();
	return 0;
}
