#include "hikvision.hpp"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include "MvCameraControl.h"

using namespace cv;
using namespace std;

cv::Mat img_rgb_left_;
cv::Mat img_rgb_right_;

int l_camera = 0;
int r_camera = 1;

void HikGrab::Hik_init()
{
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    // 枚举设备
    // enum device
    nRet = MV_CC_EnumDevices(MV_USB_DEVICE, &stDeviceList);
    if (MV_OK != nRet)
    {
        printf("MV_CC_EnumDevices fail! nRet [%x]\n", nRet);
        return ;
    }
    if (stDeviceList.nDeviceNum > 0)
    {
        for (int i = 0; i < stDeviceList.nDeviceNum; i++)
        {
            // printf("[device %d]:\n", i);
            MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
            if (NULL == pDeviceInfo)
            {
                return ;
            }         
        }  
    } 
    else
    {
        printf("Find No Devices!\n");
        return ;
    }

    unsigned int nIndex = params_.device_id;

    // 选择设备并创建句柄
    // select device and create handle
    nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
    if (MV_OK != nRet)
    {
        printf("MV_CC_CreateHandle fail! nRet [%x]\n", nRet);
        return ;
    }

    // 打开设备
    // open device
    nRet = MV_CC_OpenDevice(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_OpenDevice fail! nRet [%x]\n", nRet);
        return ;
    }
    
    // 设置触发模式为off
    // set trigger mode as off
    nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
    if (MV_OK != nRet)
    {
        printf("MV_CC_SetTriggerMode fail! nRet [%x]\n", nRet);
        return ;
    }
    /*
    // 注册抓图回调
    // register image callback
    void* l = &l_camera;
    void* r = &r_camera;
    if (nIndex)
        nRet = MV_CC_RegisterImageCallBackEx(handle, ImageCallBackEx, l);
    else
        nRet = MV_CC_RegisterImageCallBackEx(handle, ImageCallBackEx, r);

    if (MV_OK != nRet)
    {
        printf("MV_CC_RegisterImageCallBackEx fail! nRet [%x]\n", nRet);
        return ; 
    }
    */
    // 开始取流
    // start grab image
    nRet = MV_CC_StartGrabbing(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_StartGrabbing fail! nRet [%x]\n", nRet);
        return ;
    }
    nRet = MV_OK;
    memset(&stParam, 0, sizeof(MVCC_INTVALUE));
    void* pUser;
    if (nIndex == 0) pUser = &l_camera;
    else pUser = &r_camera;
    
    nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
    if (MV_OK != nRet)
    {
        printf("Get PayloadSize fail! nRet [0x%x]\n", nRet);
        return ;
    }
    stImageInfo = {0};
    memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
    pData = (unsigned char *)malloc(sizeof(unsigned char) * stParam.nCurValue);
    if (NULL == pData)
    {
        return ;
    }
    nDataSize = stParam.nCurValue;

    cout << "hik init" << endl;
}


void HikGrab::Hik_end()
{

    // 停止取流
    // end grab image
    nRet = MV_CC_StopGrabbing(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_StopGrabbing fail! nRet [%x]\n", nRet);
        return ;
    }

    // 关闭设备
    // close device
    nRet = MV_CC_CloseDevice(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_CloseDevice fail! nRet [%x]\n", nRet);
        return ;
    }

    // 销毁句柄
    // destroy handle
    nRet = MV_CC_DestroyHandle(handle);
    if (MV_OK != nRet)
    {
        printf("MV_CC_DestroyHandle fail! nRet [%x]\n", nRet);
        return ;
    }


    if (nRet != MV_OK)
    {
        if (handle != NULL)
        {
            MV_CC_DestroyHandle(handle);
            handle = NULL;
        }
    }
}



bool HikGrab::get_one_frame(cv::Mat& img, int id)
{
    nRet = MV_CC_GetOneFrameTimeout(handle, pData, nDataSize, &stImageInfo, 1000);
    //cout << nRet << endl;
    if (nRet != MV_OK) return false;
    Mat img_bayerrg_ = Mat(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC1, pData);
    if (id == 0)
    {
        cvtColor(img_bayerrg_, img, COLOR_BayerRG2RGB);
        //cv::resize(img_rgb_left_, img,cv::Size(640,640));
    }
    else if (id == 1)
    {
        cvtColor(img_bayerrg_, img_rgb_right_, COLOR_BayerRG2RGB);
        cv::resize(img_rgb_right_, img,cv::Size(640,640));
    }
    return true;

}
/*
void __stdcall ImageCallBackEx(unsigned char * pData, MV_FRAME_OUT_INFO_EX* pFrameInfo, void* pUser)
{
    if (pFrameInfo)
    {   
        Mat img_bayerrg_ = Mat(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC1, pData);
        // push wherever you want
        if (*(int*)pUser == 0)
        {
            cvtColor(img_bayerrg_, img_rgb_left_, COLOR_BayerRG2RGB);
            cv::resize(img_rgb_left_,img_rgb_left_, cv::Size(640,640));
            nv_detector.push_img_left(img_rgb_left_);
        }
        else if (*(int*)pUser == 1)
        {
            cvtColor(img_bayerrg_, img_rgb_right_, COLOR_BayerRG2RGB);
            cv::resize(img_rgb_right_,img_rgb_right_, cv::Size(640,640));
            nv_detector.push_img_right(img_rgb_right_);
        }
        // cout << *(int*)pUser << endl;

    }
}
*/
