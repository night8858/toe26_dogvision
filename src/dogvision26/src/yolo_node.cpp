#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>

#include <opencv2/opencv.hpp>

#ifdef USE_OPENVINO
#include <openvino/openvino.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

struct Detection {
	int class_id;
	float score;
	cv::Rect box;
};

class YOLOOpenVINONode {
 public:
	YOLOOpenVINONode()
			: nh_(), pnh_("~"), it_(nh_) {
		pnh_.param<std::string>("image_topic", image_topic_, "/camera/image_raw");
		pnh_.param<std::string>("annotated_topic", annotated_topic_, "/yolo/annotated_image");
		pnh_.param<std::string>("result_topic", result_topic_, "/yolo/detections");
		pnh_.param<std::string>("model_path", model_path_, "");
		pnh_.param<std::string>("device", device_, "CPU");
		pnh_.param<float>("score_threshold", score_threshold_, 0.25f);
		pnh_.param<float>("nms_threshold", nms_threshold_, 0.45f);
		pnh_.param<bool>("debug", debug_, false);

		image_sub_ = it_.subscribe(image_topic_, 1, &YOLOOpenVINONode::imageCallback, this);
		image_pub_ = it_.advertise(annotated_topic_, 1);
		result_pub_ = nh_.advertise<std_msgs::String>(result_topic_, 1);

		bool backend_ready = initBackend();
		ROS_INFO_STREAM("yolo_node started. backend=" << (backend_ready ? "openvino" : "stub")
										<< ", subscribe=" << image_topic_
										<< ", publish_image=" << annotated_topic_
										<< ", publish_result=" << result_topic_);
	}

 private:
	bool initBackend() {
#ifdef USE_OPENVINO
		if (model_path_.empty()) {
			ROS_WARN("~model_path is empty, yolo_node will run in stub mode.");
			openvino_ready_ = false;
			return false;
		}

		try {
			model_ = core_.read_model(model_path_);
			compiled_model_ = core_.compile_model(model_, device_);
			infer_request_ = compiled_model_.create_infer_request();

			const ov::Shape input_shape = compiled_model_.input().get_shape();
			if (input_shape.size() != 4) {
				ROS_WARN_STREAM("Unexpected input shape rank=" << input_shape.size()
												<< ", expected NCHW(4). yolo_node continues in stub mode.");
				openvino_ready_ = false;
				return false;
			}

			input_h_ = static_cast<int>(input_shape[2]);
			input_w_ = static_cast<int>(input_shape[3]);
			openvino_ready_ = true;
			ROS_INFO_STREAM("OpenVINO model loaded: " << model_path_
											<< ", device=" << device_
											<< ", input=" << input_w_ << "x" << input_h_);
			return true;
		} catch (const std::exception& e) {
			ROS_ERROR_STREAM("OpenVINO init failed: " << e.what() << ". Fallback to stub mode.");
			openvino_ready_ = false;
			return false;
		}
#else
		ROS_WARN("Built without OpenVINO. Rebuild with OpenVINO found by CMake to enable inference.");
		openvino_ready_ = false;
		return false;
#endif
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
		std::vector<Detection> detections = detect(image);

		std_msgs::String out_msg;
		out_msg.data = toJson(detections);
		result_pub_.publish(out_msg);

		cv::Mat annotated = image.clone();
		drawDetections(annotated, detections);
		image_pub_.publish(cv_bridge::CvImage(msg->header, "bgr8", annotated).toImageMsg());

		if (debug_) {
			ROS_DEBUG_STREAM("publish detections=" << detections.size());
		}
	}

	std::vector<Detection> detect(const cv::Mat& image_bgr) {
		std::vector<Detection> results;
		if (image_bgr.empty()) {
			return results;
		}

#ifdef USE_OPENVINO
		if (!openvino_ready_) {
			return mockResult(image_bgr);
		}

		cv::Mat resized;
		cv::resize(image_bgr, resized, cv::Size(input_w_, input_h_));
		resized.convertTo(resized, CV_32F, 1.0 / 255.0);

		std::vector<cv::Mat> chw(3);
		for (int c = 0; c < 3; ++c) {
			chw[c] = cv::Mat(input_h_, input_w_, CV_32F);
		}
		cv::split(resized, chw);

		std::vector<float> input_data(static_cast<size_t>(input_w_ * input_h_ * 3));
		const int plane = input_w_ * input_h_;
		for (int c = 0; c < 3; ++c) {
			std::memcpy(input_data.data() + c * plane, chw[c].data, sizeof(float) * plane);
		}

		ov::Tensor input_tensor(compiled_model_.input().get_element_type(),
														compiled_model_.input().get_shape(),
														input_data.data());
		infer_request_.set_input_tensor(input_tensor);
		infer_request_.infer();

		ov::Tensor output = infer_request_.get_output_tensor(0);
		results = decodeOutput(output, image_bgr.cols, image_bgr.rows);
		return nms(results, nms_threshold_);
#else
		return mockResult(image_bgr);
#endif
	}

#ifdef USE_OPENVINO
	std::vector<Detection> decodeOutput(const ov::Tensor& output, int src_w, int src_h) {
		std::vector<Detection> results;

		const ov::Shape shape = output.get_shape();
		if (shape.size() < 2) {
			ROS_WARN_THROTTLE(2.0, "Unsupported YOLO output shape rank.");
			return results;
		}

		const float* data = output.data<const float>();
		size_t num_boxes = 0;
		size_t num_values = 0;

		if (shape.size() == 3) {
			// 常见格式: [N, num_boxes, attrs]
			num_boxes = shape[1];
			num_values = shape[2];
		} else {
			// 兜底：把最后一维当 attrs
			num_values = shape.back();
			size_t total = 1;
			for (size_t v : shape) {
				total *= v;
			}
			if (num_values == 0) {
				return results;
			}
			num_boxes = total / num_values;
		}

		if (num_values < 6) {
			ROS_WARN_THROTTLE(2.0, "YOLO output attrs < 6, cannot decode.");
			return results;
		}

		for (size_t i = 0; i < num_boxes; ++i) {
			const float* row = data + i * num_values;
			float cx = row[0];
			float cy = row[1];
			float w = row[2];
			float h = row[3];

			int best_id = -1;
			float best_score = 0.0f;
			for (size_t c = 4; c < num_values; ++c) {
				if (row[c] > best_score) {
					best_score = row[c];
					best_id = static_cast<int>(c - 4);
				}
			}

			if (best_score < score_threshold_) {
				continue;
			}

			int x1 = static_cast<int>((cx - 0.5f * w) * src_w / static_cast<float>(input_w_));
			int y1 = static_cast<int>((cy - 0.5f * h) * src_h / static_cast<float>(input_h_));
			int bw = static_cast<int>(w * src_w / static_cast<float>(input_w_));
			int bh = static_cast<int>(h * src_h / static_cast<float>(input_h_));

			cv::Rect box(x1, y1, bw, bh);
			box &= cv::Rect(0, 0, src_w, src_h);
			if (box.area() <= 0) {
				continue;
			}

			Detection det;
			det.class_id = best_id;
			det.score = best_score;
			det.box = box;
			results.push_back(det);
		}
		return results;
	}
#endif

	static float iou(const cv::Rect& a, const cv::Rect& b) {
		const int inter = (a & b).area();
		if (inter <= 0) {
			return 0.0f;
		}
		const int uni = a.area() + b.area() - inter;
		if (uni <= 0) {
			return 0.0f;
		}
		return static_cast<float>(inter) / static_cast<float>(uni);
	}

	std::vector<Detection> nms(const std::vector<Detection>& input, float iou_th) {
		std::vector<Detection> sorted = input;
		std::sort(sorted.begin(), sorted.end(),
							[](const Detection& a, const Detection& b) { return a.score > b.score; });

		std::vector<Detection> picked;
		std::vector<bool> removed(sorted.size(), false);
		for (size_t i = 0; i < sorted.size(); ++i) {
			if (removed[i]) {
				continue;
			}
			picked.push_back(sorted[i]);
			for (size_t j = i + 1; j < sorted.size(); ++j) {
				if (removed[j]) {
					continue;
				}
				if (sorted[i].class_id == sorted[j].class_id && iou(sorted[i].box, sorted[j].box) > iou_th) {
					removed[j] = true;
				}
			}
		}
		return picked;
	}

	std::vector<Detection> mockResult(const cv::Mat& image_bgr) {
		std::vector<Detection> results;
		if (image_bgr.empty()) {
			return results;
		}
		Detection d;
		d.class_id = 0;
		d.score = 0.50f;
		d.box = cv::Rect(image_bgr.cols / 4, image_bgr.rows / 4,
										 image_bgr.cols / 2, image_bgr.rows / 2);
		results.push_back(d);
		return results;
	}

	void drawDetections(cv::Mat& image, const std::vector<Detection>& detections) {
		for (const auto& d : detections) {
			cv::rectangle(image, d.box, cv::Scalar(0, 255, 0), 2);
			std::ostringstream oss;
			oss << "cls=" << d.class_id << " score=" << std::fixed << std::setprecision(2) << d.score;
			cv::putText(image, oss.str(), cv::Point(d.box.x, std::max(0, d.box.y - 8)),
									cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
		}
	}

	std::string toJson(const std::vector<Detection>& detections) const {
		std::ostringstream oss;
		oss << "[";
		for (size_t i = 0; i < detections.size(); ++i) {
			const Detection& d = detections[i];
			if (i > 0) {
				oss << ",";
			}
			oss << "{\"class_id\":" << d.class_id
					<< ",\"score\":" << std::fixed << std::setprecision(4) << d.score
					<< ",\"bbox\":[" << d.box.x << "," << d.box.y << ","
					<< (d.box.x + d.box.width) << "," << (d.box.y + d.box.height) << "]}";
		}
		oss << "]";
		return oss.str();
	}

 private:
	ros::NodeHandle nh_;
	ros::NodeHandle pnh_;
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub_;
	image_transport::Publisher image_pub_;
	ros::Publisher result_pub_;

	std::string image_topic_;
	std::string annotated_topic_;
	std::string result_topic_;
	std::string model_path_;
	std::string device_;
	float score_threshold_;
	float nms_threshold_;
	bool debug_;

	bool openvino_ready_ = false;
	int input_w_ = 640;
	int input_h_ = 640;

#ifdef USE_OPENVINO
	ov::Core core_;
	std::shared_ptr<ov::Model> model_;
	ov::CompiledModel compiled_model_;
	ov::InferRequest infer_request_;
#endif
};




int main(int argc, char** argv) {
	ros::init(argc, argv, "yolo_node");
	YOLOOpenVINONode node;
	ros::spin();
	return 0;
}

