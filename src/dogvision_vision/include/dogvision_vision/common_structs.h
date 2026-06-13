/*
* common_structs.h
* Created on: 20230605
* Author: sumang
* Description: some common structs
*/
#ifndef COMMON_STRUCTS_H_
#define COMMON_STRUCTS_H_

#include <string>
#include <opencv2/opencv.hpp>
typedef struct 
{
    // file_path
        std::string bin_file_path;
        std::string xml_file_path;

        std::string yolo_device = "CPU";
    // NCHW
        int batch_size;
        int h;
        int w;
        int c;

    // img
        int type; // rgb, bgr, yuv, bayerrg8 ...
        int width;
        int height;

    // thresh
        float nms_thresh;
        float bbox_conf_thresh;
        float merge_thresh;

    // nums
        int classes;

    
    // anchors
        std::vector<float> a1;
        std::vector<float> a2;
        std::vector<float> a3;
        std::vector<float> a4;
    
    float z_scale;
    float z_scale_right;

/////////////OCR////////////////

    std::string ppocr_det_model_path;
    std::string ppocr_rec_model_path;
    std::string ppocr_cls_model_path;

    std::string det_device = "CPU";
    std::string rec_device = "CPU";
    std::string cls_device = "CPU";

    bool use_angle_cls = false;

    int det_limit_side_len = 960;
    std::string det_limit_type = "max";
    std::string det_box_type = "quad";

    float det_db_thresh = 0.3f;
    float det_db_box_thresh = 0.6f;
    float det_db_unclip_ratio = 1.5f;

    int rec_img_c = 3;
    int rec_img_h = 48;
    int rec_img_w = 320;
    int rec_batch_num = 6;

    int cls_img_c = 3;
    int cls_img_h = 48;
    int cls_img_w = 192;
    int cls_batch_num = 6;
    float cls_thresh = 0.9f;

    float drop_score = 0.5f;

    std::string rec_char_dict_path;

    std::string class0;
    std::string class1;
    std::string class2;
    std::string class3;

    float D_matrix[4];
    
}s_detector_params;

struct alignas(float) Detection
{
    float bbox[4];
    float conf;
    float class_id;
};

typedef struct
{
    int device_id;
    int width;
    int height;
    int offset_x;
    int offset_y;
    int exposure;

}s_hikcamera_params;


typedef struct
{
    int device_id;
    int width;
    int height;

}s_usbcamera_params;


typedef struct
{
    s_detector_params detect_config;
    s_hikcamera_params hikcamera_config;
    s_usbcamera_params usbcamera_config[4];

}Appconfig;


typedef struct
{
    int idx;
    int stride;
    int num_anchor;
    int num_out;
}s_OutLayer;


typedef struct{
    int id;
    std::vector<cv::Point2f> merge_pts;
    std::vector<float> merge_confs;
}pick_merge_store;


typedef struct  {


}OCRConfig;

typedef struct  {
    std::array<cv::Point2f, 4> pts;
}OCRBox;

typedef struct  {
    std::string text;
    float score = 0.0f;
}OCRRecResult;

typedef struct  {
    OCRBox box;
    OCRRecResult rec;
}OCRItem;


typedef struct  {
    int src_h ;
    int src_w ;
    float ratio_h ;
    float ratio_w ;
    int resize_h ;
    int resize_w ;
}DetResizeMeta;


#endif
