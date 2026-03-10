#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <sstream>
#include <string>
#include <vector>

struct OcrResult {
	std::string text;
	double confidence;
	cv::Rect bbox;
};

class PPOCRNode {
 public:
	PPOCRNode() : nh_(), pnh_("~"), it_(nh_) {
		pnh_.param<std::string>("image_topic", image_topic_, "/camera/image_raw");
		pnh_.param<std::string>("text_topic", text_topic_, "/ppocr/text");
		pnh_.param<std::string>("model_path", model_path_, "");

		image_sub_ = it_.subscribe(image_topic_, 1, &PPOCRNode::imageCallback, this);
		text_pub_ = nh_.advertise<std_msgs::String>(text_topic_, 1);

		// TODO: 初始化模型
		initModel();

		ROS_INFO_STREAM("ppocr_node started. subscribe=" << image_topic_);
	}

 private:
	void initModel() {
		// TODO: 在此处初始化 OCR 模型
	}

	void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
		cv_bridge::CvImageConstPtr cv_ptr;
		try {
			cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
		} catch (const cv_bridge::Exception& e) {
			ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
			return;
		}

		const cv::Mat& input_bgr = cv_ptr->image;

		// TODO: 在此处实现推理逻辑
		std::vector<OcrResult> results = detectText(input_bgr);

		std_msgs::String text_msg;
		text_msg.data = toJson(results);
		text_pub_.publish(text_msg);
	}

	std::vector<OcrResult> detectText(const cv::Mat& image_bgr) {
		std::vector<OcrResult> results;
		if (image_bgr.empty()) {
			return results;
		}

		// TODO: 在此处实现 OCR 推理逻辑

		return results;
	}

	std::string toJson(const std::vector<OcrResult>& results) const {
		std::ostringstream oss;
		oss << "[";
		for (size_t i = 0; i < results.size(); ++i) {
			const OcrResult& r = results[i];
			if (i != 0) oss << ",";
			oss << "{\"text\":\"" << r.text << "\","
			    << "\"confidence\":" << r.confidence << ","
			    << "\"bbox\":[" << r.bbox.x << "," << r.bbox.y << ","
			    << r.bbox.width << "," << r.bbox.height << "]}";
		}
		oss << "]";
		return oss.str();
	}

 private:
	ros::NodeHandle nh_;
	ros::NodeHandle pnh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	ros::Publisher text_pub_;

	std::string image_topic_;
	std::string text_topic_;
	std::string model_path_;
};

int main(int argc, char** argv) {
	ros::init(argc, argv, "ppocr_node");
	PPOCRNode node;
	ros::spin();
	return 0;
}

