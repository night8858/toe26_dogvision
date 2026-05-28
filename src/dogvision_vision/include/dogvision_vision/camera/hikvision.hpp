/*
* GrabImg.h
* Created on: 20230613
* Author: sumang
* Description: grab img
*/
// #ifndef HIKVISION_HPP_
// #define HIKVISION_HPP_

#pragma once

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"


// 相机内参
typedef struct
{
    int device_id;
    int width;
    int height;
    int offset_x;
    int offset_y;
    int exposure;

} s_camera_params;


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
    /**
     * @brief 构造海康相机封装对象。
     * @param param 相机初始化参数。
     * @retval 无
     */
    HikGrab(s_camera_params param){params_ = param;};

    /**
     * @brief 从海康相机获取一帧图像。
     * @param img 输出 BGR 图像。
     * @param id 需要取帧的相机编号。
     * @retval bool 成功获取非空图像时返回 true。
     */
    bool get_one_frame(cv::Mat& img, int id);

    /**
     * @brief 析构相机封装对象。
     * @param 无
     * @retval 无
     */
    ~HikGrab(){};

    /**
     * @brief 初始化海康 MVS 相机。
     * @param 无
     * @retval void
     */
    void Hik_init();

    /**
     * @brief 停止采集并释放海康 MVS 相机。
     * @param 无
     * @retval void
     */
    void Hik_end();


};
/**
 * @brief 海康 SDK 帧回调函数。
 * @param pData 原始帧数据缓冲区。
 * @param pFrameInfo SDK 帧元数据。
 * @param pUser 传递给 SDK 的用户指针。
 * @retval void
 */
extern void __stdcall ImageCallBackEx(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
extern cv::Mat img_rgb_;
extern std::mutex img_mutex;

// #endif
