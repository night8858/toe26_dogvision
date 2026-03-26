#include <iostream>
#include <opencv2/opencv.hpp>
#include "nuc_detect.hpp"

Appconfig config;			 // 全局配置对象，供 detector_ov 使用
detect_oponvino detector(&config); // 传入配置对象

int main(int argc, char** argv)
 {





    // 这里可以添加测试代码，例如加载一张图片进行推理
    // cv::Mat test_img = cv::imread("test.jpg");
    // detector.preprocess(test_img);
    // detector.inference();
    // detector.postprocess();

    std::cout << "YOLO detector initialized successfully!" << std::endl;
    return 0;
}