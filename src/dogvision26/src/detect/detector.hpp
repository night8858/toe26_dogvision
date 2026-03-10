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
    ~detector();

    const int max_size_ = 10;

    std::vector<cv::Mat> input_imgs_hikvion;
    std::vector<cv::Mat> input_imgs_usb[4];

    std::mutex hik_img_mutex_;
    std::mutex usb_img_mutex_[4];

    int hik_img_flag; // hik相机图像标志位
    int usb_img_flag[4]; // usb相机图像标志位

    cv::Mat input_img_hik_;
    cv::Mat input_img_usb_[4];

    cv::Mat show_img_hik;
    cv::Mat show_img_usb[4];

    s_detector_params detect_config_;

    //int yolo_inputtensor_size = 640 *640 * 3; // 640*640*3
    
public:
    void push_img(cv::Mat &giab_img , int cam_id );

    void show_yolo_result(cv::Mat &show_img , Detection &det);
    void show_ocr_result(void);

    bool get_yolo_result(cv::Mat &input_img , std::vector<Detection> &res);
    bool get_ocr_result(void);

    virtual void preprocess() = 0;
    virtual void inference() = 0;
    virtual void postprocess() = 0;

private:

    // int flag = 2;    //用来判断单路推理双路推理

    // cv::Mat m_img_src;
    // AffineMat m_dst2src;

    // Detection det;
    // Detection det1;
    // Detection det2;

    // std::vector<Detection> res;
    // std::vector<Detection> res1;
    // std::vector<Detection> res2;

    // nvinfer1::Dims m_output_dims;
    // int m_output_area;
    // int m_total_objects;

    // std::vector<unsigned char> engine_data_;
    // std::unique_ptr<nvinfer1::IRuntime> runtime_;
    // std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    // std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // int inputIndex;
    // int outputIndex;

    // // input
    // float *input_host;

    // float *device_buffers[2];          //存储输入输出的数据
    // // output
    // float *output_device_host;

    // uint8_t *img_buffer_host_1;
    // uint8_t *img_buffer_device_1;

    // uint8_t *img_buffer_host_2;
    // uint8_t *img_buffer_device_2;


    // float *output_device;
    // float *output_objects_device;
    // float *output_objects_host;
////////////////////////////////////////////

   



};