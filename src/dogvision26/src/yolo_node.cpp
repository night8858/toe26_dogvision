#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "nuc_detect.hpp"
#include "detector.hpp"

using namespace std;


//当前是尝试在node中直接调用detect_oponvino类进行视频推理，后续会根据需要调整代码结构，例如将推理部分封装成一个ROS节点类，或者使用ROS服务/动作接口等方式进行交互。
int main(int argc, char **argv)
{
	//ros节点初始化
	ros::init(argc, argv, "yolo_node");
	//ROS节点句柄和图像发布者初始化
	ros::NodeHandle nh;
	ros::NodeHandle pnh("~");
	image_transport::ImageTransport it(nh);

	string video_path;
	string json_config_path;
	string output_video_path;
	string result_topic;
	string image_topic;
	bool show_window = true;

	pnh.param<string>("video_path", video_path, "/home/toe/toe26_dogvision/src/dogvision26/src/data/video/test1.mp4");
	pnh.param<string>("config_path", json_config_path, "/home/toe/toe26_dogvision/src/dogvision26/src/detect/settings.json");
	pnh.param<string>("output_video_path", output_video_path, "/home/toe/toe26_dogvision/src/dogvision26/src/data/video");
	pnh.param<string>("result_topic", result_topic, "/yolo/result_text");
	pnh.param<string>("image_topic", image_topic, "/yolo/result_image");
	pnh.param<bool>("show_window", show_window, true);

	if (video_path.empty())
	{
		ROS_ERROR("Missing required param ~video_path");
		return -1;
	}

	Appconfig config;
	//加载配置文件并初始化检测器
	detect_oponvino config_loader(nullptr);
	config_loader.load_config(config, json_config_path);

	//初始化YOLO检测器
	detect_oponvino detector_ov(&config);
	if (!detector_ov.inference_init())
	{
		ROS_ERROR("Failed to initialize YOLO detector");
		return -1;
	}

	cv::VideoCapture cap(video_path);
	if (!cap.isOpened())
	{
		ROS_ERROR_STREAM("Failed to open video: " << video_path);
		return -1;
	}

	const int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	const int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	double fps = cap.get(cv::CAP_PROP_FPS);
	if (fps <= 0.0)
	{
		fps = 25.0;
	}

	cv::VideoWriter writer;
	if (!output_video_path.empty())
	{
		const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
		writer.open(output_video_path, fourcc, fps, cv::Size(frame_width, frame_height));
		if (!writer.isOpened())
		{
			ROS_WARN_STREAM("Cannot open output video path, skip writing: " << output_video_path);
		}
	}

	image_transport::Publisher image_pub = it.advertise(image_topic, 1);
	ros::Publisher result_pub = nh.advertise<std_msgs::String>(result_topic, 10);

	ROS_INFO_STREAM("Start video inference: " << video_path);
	ROS_INFO_STREAM("Config path: " << json_config_path);

	int frame_id = 0;
	while (ros::ok())
	{
		cv::Mat frame;
		if (!cap.read(frame))
		{
			ROS_INFO("Video inference finished");
			break;
		}

		std::vector<Detection> results;
		detector_ov.yolo_run(frame, results);

		cv::Mat vis = frame.clone();
		for (const auto& det : results)
		{
			detector_ov.show_yolo_result(vis, det);
		}

		std_msgs::String msg;
		std::ostringstream oss;
		oss << "{\"frame_id\":" << frame_id << ",\"count\":" << results.size() << ",\"dets\":[";
		for (size_t i = 0; i < results.size(); ++i)
		{
			const auto& d = results[i];
			if (i > 0) oss << ",";
			oss << "{\"cls\":" << static_cast<int>(d.class_id)
			    << ",\"conf\":" << std::fixed << std::setprecision(4) << d.conf
			    << ",\"bbox\":[" << d.bbox[0] << "," << d.bbox[1] << "," << d.bbox[2] << "," << d.bbox[3] << "]}";
		}
		oss << "]}";
		msg.data = oss.str();
		result_pub.publish(msg);

		sensor_msgs::ImagePtr image_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", vis).toImageMsg();
		image_msg->header.stamp = ros::Time::now();
		image_pub.publish(image_msg);

		if (writer.isOpened())
		{
			writer.write(vis);
		}

		if (show_window)
		{
			cv::imshow("yolo_video_infer", vis);
			if (cv::waitKey(1) == 27)
			{
				ROS_INFO("ESC pressed, stop inference");
				break;
			}
		}

		ros::spinOnce();
		++frame_id;
	}

	cap.release();
	if (writer.isOpened())
	{
		writer.release();
	}
	if (show_window)
	{
		cv::destroyAllWindows();
	}

	return 0;
}
