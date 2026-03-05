#include <iostream>
#include <opencv2/opencv.hpp>
#include "hikvision.hpp"
#include "usbcam.hpp" 
#include "common_struct.hpp"
#include <mutex>

using namespace std;
using namespace cv;
int main()
{

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    toe::usb_camera usb1;

    s_camera_params cam;
    cam.exposure = 6000;
    cam.device_id = 0;
    cam.width = 1440;
    cam.height = 1080;
    // HikGrab hik(cam);
    // hik.Hik_init();
    // hik.Hik_end();
    // sleep(2);
    // hik.Hik_init();
    // namedWindow("a");
    usb1.usb_camera_init(usb1, cap);
    Mat img_rgb_;
    while (1)
    {
        usb1.usb_camera_get_frame(cap,img_rgb_);
        usb1.usb_camera_show_frame();
        // hik.get_one_frame(img_rgb_, 0);
        // Mat s = img_rgb_;
        // if (s.size[0] > 0)
        // {
        //     cv::imshow("a", s);
        // }

        // cv::waitKey(2);
        
    }
    // hik.Hik_end();
    cv::destroyAllWindows();
    return 0;

}