#pragma once 

#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>
#include <nlohmann/json.hpp>

#include "common_structs.h"



class detector
{
public:
    detector(Appconfig* config);
    virtual ~detector();

    void push_img(cv::Mat &giab_img , int cam_id );

    void show_yolo_result(cv::Mat &show_img , const Detection &det);
    void show_ocr_result(void);
    cv::Mat diatorion(cv::Mat &show_img);
    bool yolo_run(cv::Mat &input_img , std::vector<Detection> &res);
    bool get_ocr_result(void);

    void load_config(Appconfig& config, std::string json_file_path);

    
protected:

    virtual const std::vector<Detection>* yolo_results_ptr() const;

    //virtual void load_model(const std::string& model_path, const std::string& device) = 0;
    virtual void preprocess(cv::Mat &src) = 0;
    virtual void inference() = 0;
    virtual void postprocess() = 0;

    const int max_size_ = 10;

    std::vector<cv::Mat> input_imgs_hikvion;
    std::vector<cv::Mat> input_imgs_usb[4];

    std::mutex hik_img_mutex_;
    std::mutex usb_img_mutex_[4];
    std::mutex yolo_infer_mutex_;

    int hik_img_flag; // hik相机图像标志位
    int usb_img_flag[4]; // usb相机图像标志位

    cv::Mat input_img_hik_;
    cv::Mat input_img_usb_[4];

    cv::Mat show_img_hik;
    cv::Mat show_img_usb[4];

    s_detector_params detect_config_;

};