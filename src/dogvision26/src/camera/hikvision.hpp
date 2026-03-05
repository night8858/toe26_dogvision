/*
* GrabImg.h
* Created on: 20230613
* Author: sumang
* Description: grab img
*/
#ifndef GRABIMG_H_
#define GRABIMG_H_

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"
#include "common_struct.hpp"

class HikGrab
{
private:
    s_camera_params params_;
    cv::Mat img_bayerrg_;
    void* handle;
    int nRet;
    // ch:获取数据包大小 | en:Get payload size
    MVCC_INTVALUE stParam;
    MV_FRAME_OUT_INFO_EX stImageInfo;

    unsigned char * pData;
    unsigned int nDataSize;


public:
    HikGrab(s_camera_params param){params_ = param;};
    bool get_one_frame(cv::Mat& img, int id);
    ~HikGrab(){};
    void Hik_init();
    void Hik_end();


};
extern void __stdcall ImageCallBackEx(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
extern cv::Mat img_rgb_;
extern std::mutex img_mutex;
#endif