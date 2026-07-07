/*
* GrabImg.h
* Created on: 20230613
* Author: sumang
* Description: grab img
*/
// #ifndef HIKVISION_HPP_
// #define HIKVISION_HPP_

#pragma once

// Hik/MVS support is currently disabled from CMake and CameraSource.
// Keep this header for future restore.

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "MvCameraControl.h"


/**
 * @brief 相机基本参数
 */
typedef struct
{
    int device_id; ///< 相机设备编号（枚举索引）
    int width;     ///< 采集宽度（像素）
    int height;    ///< 采集高度（像素）
    int offset_x;  ///< ROI 水平偏移
    int offset_y;  ///< ROI 垂直偏移
    int exposure;  ///< 曝光时间（微秒）

} s_camera_params;


/**
 * @brief 海康相机封装类，包装 MVS SDK 的设备管理与图像采集接口
 */
class HikGrab
{
private:
    s_camera_params params_;   ///< 相机初始化参数
    cv::Mat img_bayerrg_;      ///< Bayer RG 原始格式缓存
    void* handle = nullptr;    ///< MVS 设备句柄
    int nRet = 0;              ///< MVS API 返回值暂存
    MVCC_INTVALUE stParam{};   ///< 传输层参数（如 PayloadSize）
    MV_FRAME_OUT_INFO_EX stImageInfo{}; ///< 帧信息（宽、高、时间戳等）

    unsigned char * pData = nullptr; ///< 帧数据缓冲区
    unsigned int nDataSize = 0;      ///< 缓冲区大小
    bool device_open_ = false;       ///< 设备是否已打开
    bool grabbing_ = false;          ///< 是否正在取流


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

    /**
     * @brief 开始海康相机取流，保留已打开设备句柄。
     * @retval bool 成功或已在取流时返回 true。
     */
    bool start_grabbing();

    /**
     * @brief 暂停海康相机取流，保留已打开设备句柄。
     * @retval bool 成功或已暂停时返回 true。
     */
    bool stop_grabbing();


};
/**
 * @brief 海康 SDK 帧回调函数。
 * @param pData 原始帧数据缓冲区。
 * @param pFrameInfo SDK 帧元数据。
 * @param pUser 传递给 SDK 的用户指针。
 * @retval void
 */
extern void __stdcall ImageCallBackEx(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser);
extern cv::Mat img_rgb_;       ///< 全局 RGB 图像缓存（回调填充）
extern std::mutex img_mutex;   ///< 图像缓存互斥锁

// #endif
