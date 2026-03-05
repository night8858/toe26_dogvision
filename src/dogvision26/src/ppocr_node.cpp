#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>

#include <opencv2/opencv.hpp>

#include <iomanip>
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
	PPOCRNode()
			: nh_(), pnh_("~"), it_(nh_) {
		pnh_.param<std::string>("image_topic", image_topic_, "/camera/image_raw");
		pnh_.param<std::string>("annotated_topic", annotated_topic_, "/ppocr/annotated_image");
		pnh_.param<std::string>("text_topic", text_topic_, "/ppocr/text");
		pnh_.param<std::string>("model_path", model_path_, "");
		pnh_.param<bool>("debug", debug_, false);

		image_sub_ = it_.subscribe(image_topic_, 1, &PPOCRNode::imageCallback, this);
		image_pub_ = it_.advertise(annotated_topic_, 1);
		text_pub_ = nh_.advertise<std_msgs::String>(text_topic_, 1);

		ROS_INFO_STREAM("ppocr_node started. subscribe=" << image_topic_
										<< ", publish_image=" << annotated_topic_
										<< ", publish_text=" << text_topic_);
	}

 private:
	void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
		cv_bridge::CvImageConstPtr cv_ptr;
		try {
			cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
		} catch (const cv_bridge::Exception& e) {
			ROS_ERROR_STREAM("cv_bridge exception: " << e.what());
			return;
		}

		const cv::Mat& input_bgr = cv_ptr->image;
		std::vector<OcrResult> results = detectText(input_bgr);

		std_msgs::String text_msg;
		text_msg.data = toJson(results);
		text_pub_.publish(text_msg);

		cv::Mat annotated = input_bgr.clone();
		drawAnnotations(annotated, results);
		image_pub_.publish(cv_bridge::CvImage(msg->header, "bgr8", annotated).toImageMsg());

		if (debug_) {
			ROS_DEBUG_STREAM("OCR result count=" << results.size());
		}
	}

	std::vector<OcrResult> detectText(const cv::Mat& image_bgr) {
		std::vector<OcrResult> results;

		if (image_bgr.empty()) {
			return results;
		}

		cv::Mat gray;
		cv::cvtColor(image_bgr, gray, cv::COLOR_BGR2GRAY);
		const cv::Scalar mean_val = cv::mean(gray);
		if (mean_val[0] < 5.0) {
			return results;
		}

		// TODO: 在此处接入 PP-OCR 推理，填充 results。
		OcrResult demo;
		demo.text = "demo_text";
		demo.confidence = 0.90;
		demo.bbox = cv::Rect(image_bgr.cols / 4, image_bgr.rows / 2 - 20,
												 image_bgr.cols / 2, 40);
		results.push_back(demo);

		return results;
	}

	void drawAnnotations(cv::Mat& image, const std::vector<OcrResult>& results) {
		for (const auto& item : results) {
			cv::rectangle(image, item.bbox, cv::Scalar(0, 255, 0), 2);
			std::ostringstream label;
			label << item.text << " (" << std::fixed << std::setprecision(2) << item.confidence << ")";
			int baseline = 0;
			const cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
			cv::Point origin(item.bbox.x, std::max(0, item.bbox.y - 6));
			cv::rectangle(image,
										cv::Point(origin.x, origin.y - text_size.height - 2),
										cv::Point(origin.x + text_size.width, origin.y + baseline),
										cv::Scalar(0, 255, 0),
										cv::FILLED);
			cv::putText(image, label.str(), origin, cv::FONT_HERSHEY_SIMPLEX, 0.6,
									cv::Scalar(0, 0, 0), 2);
		}
	}

	std::string toJson(const std::vector<OcrResult>& results) const {
		std::ostringstream oss;
		oss << "[";
		for (size_t i = 0; i < results.size(); ++i) {
			const OcrResult& r = results[i];
			if (i != 0) {
				oss << ",";
			}
			oss << "{\"text\":\"" << escapeJson(r.text) << "\",";
			oss << "\"confidence\":" << std::fixed << std::setprecision(4) << r.confidence << ",";
			oss << "\"bbox\":[" << r.bbox.x << "," << r.bbox.y << ","
					<< (r.bbox.x + r.bbox.width) << "," << (r.bbox.y + r.bbox.height) << "]}";
		}
		oss << "]";
		return oss.str();
	}

	static std::string escapeJson(const std::string& in) {
		std::ostringstream oss;
		for (const char c : in) {
			switch (c) {
				case '"': oss << "\\\""; break;
				case '\\': oss << "\\\\"; break;
				case '\b': oss << "\\b"; break;
				case '\f': oss << "\\f"; break;
				case '\n': oss << "\\n"; break;
				case '\r': oss << "\\r"; break;
				case '\t': oss << "\\t"; break;
				default: oss << c; break;
			}
		}
		return oss.str();
	}

 private:
	ros::NodeHandle nh_;
	ros::NodeHandle pnh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	image_transport::Publisher image_pub_;
	ros::Publisher text_pub_;

	std::string image_topic_;
	std::string annotated_topic_;
	std::string text_topic_;
	std::string model_path_;
	bool debug_;
};

int main(int argc, char** argv) {
	ros::init(argc, argv, "ppocr_node");
	PPOCRNode node;
	ros::spin();
	return 0;
}

