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

struct Detection {
	int class_id;
	float score;
	cv::Rect box;
};

class YOLONode {
 public:
	YOLONode() : nh_(), pnh_("~"), it_(nh_) {
		// 参数
		pnh_.param<std::string>("image_topic", image_topic_, "/camera/image_raw");
		pnh_.param<std::string>("result_topic", result_topic_, "/yolo/detections");
		pnh_.param<std::string>("model_path", model_path_, "");

		// 订阅和发布
		image_sub_ = it_.subscribe(image_topic_, 1, &YOLONode::imageCallback, this);
		result_pub_ = nh_.advertise<std_msgs::String>(result_topic_, 1);

		// TODO: 初始化 OpenVINO 模型
		initModel();

		ROS_INFO_STREAM("yolo_node started. subscribe=" << image_topic_);
	}

 private:
	void initModel() {
		// TODO: 在此处初始化 OpenVINO 模型
		// 示例:
		// model_ = core_.read_model(model_path_);
		// compiled_model_ = core_.compile_model(model_, "CPU");
		// infer_request_ = compiled_model_.create_infer_request();
	}

	void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
		cv_bridge::CvImageConstPtr cv_ptr;
		try {
			cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
		} catch (const cv_bridge::Exception& e) {
			ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
			return;
		}

		const cv::Mat& image = cv_ptr->image;

		// TODO: 在此处实现推理逻辑
		std::vector<Detection> detections = detect(image);

		// 发布结果
		std_msgs::String out_msg;
		out_msg.data = toJson(detections);
		result_pub_.publish(out_msg);
	}

	std::vector<Detection> detect(const cv::Mat& image_bgr) {
		std::vector<Detection> results;
		if (image_bgr.empty()) {
			return results;
		}

		// TODO: 在此处实现 YOLO 推理逻辑
		// 1. 预处理图像
		// 2. 执行推理
		// 3. 解码输出
		// 4. NMS 后处理

		return results;
	}

	std::string toJson(const std::vector<Detection>& detections) const {
		std::ostringstream oss;
		oss << "[";
		for (size_t i = 0; i < detections.size(); ++i) {
			const Detection& d = detections[i];
			if (i > 0) oss << ",";
			oss << "{\"class_id\":" << d.class_id
			    << ",\"score\":" << d.score
			    << ",\"bbox\":[" << d.box.x << "," << d.box.y << ","
			    << d.box.width << "," << d.box.height << "]}";
		}
		oss << "]";
		return oss.str();
	}

 private:
	ros::NodeHandle nh_;
	ros::NodeHandle pnh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	ros::Publisher result_pub_;

	std::string image_topic_;
	std::string result_topic_;
	std::string model_path_;

	// TODO: OpenVINO 成员变量
	// ov::Core core_;
	// std::shared_ptr<ov::Model> model_;
	// ov::CompiledModel compiled_model_;
	// ov::InferRequest infer_request_;
};

int main(int argc, char** argv) {
	ros::init(argc, argv, "yolo_node");
	YOLONode node;
	ros::spin();
	return 0;
}

