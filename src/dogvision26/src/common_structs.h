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
#include "hikvision.hpp"
typedef struct 
{
    // file_path
        std::string ppocr_file_path;
        std::string bin_file_path;
        std::string xml_file_path;

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



// typedef struct
// {
//     std::vector<s_armor> armor;
// }s_detections;

typedef struct
{
    s_detector_params detect_config;
    s_hikcamera_params hikcamera_config;
    s_usbcamera_params usbcamera_config[4];
#ifdef TWO_CAMERAS
    s_camera_params camera2_config;
#else
#endif
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


#endif
