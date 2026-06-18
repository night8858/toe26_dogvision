#include <dogvision_vision/camera/usbcam.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>



/**
 * @brief 初始化 USB 相机
 *
 * 设置 MJPG 编码、640×480 分辨率、120 FPS，
 * 并通过实际读取参数验证设置是否生效。
 *
 * @param capture 需初始化的 OpenCV VideoCapture 对象
 * @retval true  初始化成功
 * @retval false 分辨率校验失败或打开相机失败
 */
bool usb_camera::usb_camera_init(cv::VideoCapture &capture)
{
    cv::VideoWriter writer;
    // 设置为 MJPG 编码格式以减少带宽占用
    int fourcc = writer.fourcc('M', 'J', 'P', 'G');
    capture.set(cv::CAP_PROP_FOURCC, fourcc);
    // 设置目标分辨率
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    // 设置目标帧率
    capture.set(cv::CAP_PROP_FPS, 120);

    // 验证实际分辨率是否与设置一致
    double actualWidth = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    double actualHeight = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (actualWidth != 640 || actualHeight != 480)
    {
        std::cerr << "Error: USB camera resolution mismatch (set 640x480, got "
                  << actualWidth << "x" << actualHeight << ")" << std::endl;
        return false;
    }
    // 验证相机是否已打开
    if (!capture.isOpened())
    {
        std::cout << "Error: USB camera failed to open" << std::endl;
        return false;
    }
    std::cout << "USB camera initialized successfully" << std::endl;
    return true;
}

/**
 * @brief 从 USB 相机读取一帧图像
 *
 * 线程安全：使用互斥锁保护 capture >> frame 操作。
 *
 * @param capture 已打开的 OpenCV 采集对象
 * @param frame   输出图像帧
 * @retval true  始终返回 true
 */
bool usb_camera::usb_camera_get_frame(cv::VideoCapture &capture, cv::Mat &frame)
{
    // 加锁读取，防止多线程同时访问 capture
    usb_mutex_array[0].lock();
    capture >> frame;
    usb_mutex_array[0].unlock();
    return true;
}

/**
 * @brief 在 OpenCV 窗口中显示最近一帧 USB 图像
 */
void usb_camera::usb_camera_show_frame(void)
{
    cv::imshow("USB Camera", usb_frame_array[0]);
    cv::waitKey(1);
}

/**
 * @brief 判断图像帧中是否同时含有黄、蓝、白三种颜色区域
 *
 * 将 BGR 图像转为 HSV，分别提取黄、蓝、白的掩码，
 * 若各颜色非零像素数均 > 5 则认为满足条件。
 *
 * @param frame 输入 BGR 图像帧
 * @retval true  同时含有黄、蓝、白
 * @retval false 缺少至少一种颜色或图像为空
 */
bool color_judge(cv::Mat &frame)
{
    if (frame.empty())
    {
        return false;
    }

    cv::Mat hsvImage;
    cv::cvtColor(frame, hsvImage, cv::COLOR_BGR2HSV);

    cv::Scalar lowerYellow(20, 100, 100);
    cv::Scalar upperYellow(30, 255, 255);
    cv::Scalar lowerBlue(100, 100, 100);
    cv::Scalar upperBlue(140, 255, 255);
    cv::Scalar lowerWhite(0, 0, 200); // 白色在HSV中通常有很高的Value
    cv::Scalar upperWhite(180, 50, 255);

    cv::Mat maskYellow, maskBlue, maskWhite;
    cv::inRange(hsvImage, lowerYellow, upperYellow, maskYellow);
    cv::inRange(hsvImage, lowerBlue, upperBlue, maskBlue);
    cv::inRange(hsvImage, lowerWhite, upperWhite, maskWhite);

    bool hasYellow = (cv::countNonZero(maskYellow) > 5);
    bool hasBlue = (cv::countNonZero(maskBlue) > 5);
    bool hasWhite = (cv::countNonZero(maskWhite) > 5);

    if (hasYellow && hasBlue && hasWhite)
    {
        return true;
    }
    else
    {
        return false;
    }
}
/**
 * @brief 将面积值限制在指定范围 [limit_min, limit_max] 内
 *
 * 当 input < limit_min 时返回 limit_min，
 * 当 input > limit_max 时返回 limit_max，
 * 否则返回 input 本身。
 *
 * @param input     输入面积值
 * @param limit_min 下限
 * @param limit_max 上限
 * @return 限幅后的面积值
 */
int rect_area_limit(int input, int limit_min, int limit_max)
{
    if (input < limit_min)
    {
        return limit_min;
    }
    else if (input > limit_max)
    {
        return limit_max;
    }
    else
    {
        return input;
    }
}

/*   这是一套暂存的可用代码，
    auto temp_para = input_json["usbcamera"];
    int deep_num = temp_para["deep_num"].get<int>();

    bool ball_flag = 0;

    // 定义橙色的HSV范围
    cv::Scalar lowerOrange(8, 100, 5);
    cv::Scalar upperOrange(30, 255, 255);

    cv::Mat image = frame.clone();
    if (image.empty())
    {
        std::cerr << "error:图像为空" << std::endl;
        return;
    }

    // 转换颜色空间到HSV
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    cv::Scalar lowerOrange1(10, 100, 100);
    cv::Scalar upperOrange1(30, 255, 200);
    cv::Scalar lowerOrange2(0, 100, 100);
    cv::Scalar upperOrange2(10, 255, 200);

    // 橙色的掩膜
    cv::Mat mask1, mask2;
    cv::inRange(hsvImage, lowerOrange1, upperOrange1, mask1);
    cv::inRange(hsvImage, lowerOrange2, upperOrange2, mask2);
    cv::Mat mask = mask1 | mask2;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));
    // 应用开操作
    cv::Mat openedImage;
    cv::morphologyEx(mask, openedImage, cv::MORPH_OPEN, kernel);

    // 检测
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(openedImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 绘制圆圈
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contours[i], center, radius);

        // 绘制最小包围圆
        cv::circle(image, center, radius, cv::Scalar(0, 255, 0), 2);
        if (i == contours.size() - 1)
        {
            ball_posion.x = center.x;
            ball_posion.y = center.y;
            ball_posion.Deep = deep_num * (1 / radius);

            ball_flag = 1;

            std::string text1 = "X:" + std::to_string(ball_posion.x);
            std::string text2 = "Y:" + std::to_string(ball_posion.y);
            std::string text3 = "D:" + std::to_string(ball_posion.Deep);
            cv::Point org1(center.x, center.y);
            cv::Point org2(center.x, center.y +15);
            cv::Point org3(center.x, center.y +30);
            int fontFace = cv::FONT_HERSHEY_COMPLEX;
            double fontScale = 0.7;
            cv::Scalar color(200, 0, 0);
            int thickness = 1;

            cv::putText(image, text1, org1, fontFace, fontScale, color, thickness);
            cv::putText(image, text2, org2, fontFace, fontScale, color, thickness);
            cv::putText(image, text3, org3, fontFace, fontScale, color, thickness);

        }
    }

    if (ball_flag == 0)
    {
        ball_posion.x = 320;
        ball_posion.y = 240;
        ball_posion.Deep = 0;
    }
    result = image;

    // std::cout<<"ball_posion.x:"<<ball_posion.x<<"ball_posion.y:"<<ball_posion.y<<"ball_posion.Deep:"<<ball_posion.Deep <<std::endl;

    cv::imshow("m", mask);
    cv::imshow("o", openedImage);
*/

/*霍夫曼圆形检测

    auto temp_para = input_json["usbcamera"];
    int deep_num = temp_para["deep_num"].get<int>();

    bool ball_flag = 0;

    cv::Mat image = frame.clone();
    if (image.empty())
    {
        std::cerr << "error:图像为空" << std::endl;
        return;
    }
    
    // 转换颜色空间到HSV
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    cv::Scalar lowerOrange1(10, 100, 50);
    cv::Scalar upperOrange1(30, 255, 200);
    cv::Scalar lowerOrange2(0, 100, 50);
    cv::Scalar upperOrange2(10, 255, 200);

    // 橙色的掩膜
    cv::Mat mask1, mask2;
    cv::inRange(hsvImage, lowerOrange1, upperOrange1, mask1);
    cv::inRange(hsvImage, lowerOrange2, upperOrange2, mask2);
    cv::Mat mask = mask1 | mask2;

    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));

    // 应用开操作
    cv::Mat openedImage;
    cv::morphologyEx(mask, openedImage, cv::MORPH_OPEN, kernel);
    

    cv::Mat gery;
    cv::cvtColor(image, gery, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(gery, gery, cv::Size(3, 3), 0);
    // 检测
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gery, circles, cv::HOUGH_GRADIENT, 1.1,
                     gery.rows / 8, // change this value to detect circles with different distances to each other
                     130, 40, 30, 300);     // 调整最后两个参数可检测不同大小的圆
    
    // 绘制圆圈
    for (size_t i = 0; i < circles.size(); i++)
    {
        cv::Point2f center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // 绘制圆心
        cv::circle(image, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
        // 绘制圆轮廓
        cv::circle(image, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);

        if (i == circles.size() - 1)
        {
             ball_posion.x = center.x;
            ball_posion.y = center.y;
            ball_posion.Deep = deep_num * (1 / circles[i][2]);

            ball_flag = 1;

            std::string text1 = "X:" + std::to_string(ball_posion.x);
            std::string text2 = "Y:" + std::to_string(ball_posion.y);
            std::string text3 = "D:" + std::to_string(ball_posion.Deep);

            cv::Point org1(center.x, center.y);
            cv::Point org2(center.x, center.y + 15);
            cv::Point org3(center.x, center.y + 30);

            int fontFace = cv::FONT_HERSHEY_COMPLEX;
            double fontScale = 0.7;
            cv::Scalar color(200, 0, 0);
            int thickness = 1;

            cv::putText(image, text1, org1, fontFace, fontScale, color, thickness);
            cv::putText(image, text2, org2, fontFace, fontScale, color, thickness);
            cv::putText(image, text3, org3, fontFace, fontScale, color, thickness);
        
        }
        
           
    }

    if (ball_flag == 0)
    {
        ball_posion.x = 320;
        ball_posion.y = 240;
        ball_posion.Deep = 0;
    }
    result = image;

    // std::cout<<"ball_posion.x:"<<ball_posion.x<<"ball_posion.y:"<<ball_posion.y<<"ball_posion.Deep:"<<ball_posion.Deep <<std::endl;

    cv::imshow("m", mask);
    //cv::imshow("o", openedImage);

*/

/*cv::Mat grad_x, grad_y, grey_img;
   cv::Mat abs_grad_x, abs_grad_y;
   cv::Mat temp1; // 临时变量
   cv::Mat frame_clone;

   frame_clone = frame.clone();

   int scale = 6;
   int delta = 0;
   int ddepth = CV_16S;

   // 求图像梯度部分
   cv::cvtColor(frame_clone, grey_img, cv::COLOR_BGR2GRAY);

   cv::Sobel(grey_img, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
   cv::convertScaleAbs(grad_x, abs_grad_x);
   cv::Sobel(grey_img, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
   cv::convertScaleAbs(grad_y, abs_grad_y);

   cv::Mat abs_grad_xy;
   cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, abs_grad_xy);

   cv::Mat binary;
   cv::GaussianBlur(abs_grad_xy, binary, cv::Size(9, 9), 2, 2);
   // 以上都是计算图像梯度//

   // 二值化，开操作部分//
   cv::Mat threshold_img; // 二值化结果
   cv::threshold(binary, threshold_img, 120, 255, cv::THRESH_BINARY);

   // 尝试开操作之类的处理一下
   cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));

   // 开操作
   cv::Mat opening;
   cv::morphologyEx(threshold_img, opening, cv::MORPH_OPEN, kernel);

   // 结束//

   // result = opening.clone();
   //判断圆形区域，绘制圆形
   std::vector<cv::Vec3f> circles;
   cv::HoughCircles(opening, circles, cv::HOUGH_GRADIENT, 1,
                    opening.rows / 4, // change this value to detect circles with different distances to each other
                    200, 30, 5, 300); // 调整最后两个参数可检测不同大小的圆

   //cv::Mat circles_img;
   //cv::cvtColor(frame_clone, circles_img, cv::COLOR_BGR2HSV);

   cv::Mat circles_img;
   circles_img = frame.clone();

   cv::Point possible_ball_center;
   int possible_ball_radius = 0;
   // 绘制圆形
   for (size_t i = 0; i < circles.size(); i++)
   {
       cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
       int radius = cvRound(circles[i][2]);
       // 绘制圆心
       cv::circle(circles_img, center, 1, cv::Scalar(0, 255, 0), -1, 8, 0);
       // 绘制圆轮廓
       cv::circle(circles_img, center, radius, cv::Scalar(0, 0, 255), 3, 1, 0);
       if (i == circles.size() - 1)
       {
           possible_ball_center = center;
           possible_ball_radius = radius;
       }
   }

   result = circles_img.clone();

   // 可能的球的矩形
   if (possible_ball_center.x >= 0 && possible_ball_center.y >= 0 && possible_ball_radius > 0 && possible_ball_radius > 0 &&
       possible_ball_center.x + possible_ball_radius <= circles_img.cols &&
       possible_ball_center.y + possible_ball_radius <= circles_img.rows)
   {
       // ROI是有效的，可以继续处理
       cv::Rect possible_ball(rect_area_limit(possible_ball_center.x - possible_ball_radius, 1, 640),
                              rect_area_limit(possible_ball_center.y - possible_ball_radius, 1, 480),
                              possible_ball_radius + possible_ball_radius,
                              possible_ball_radius + possible_ball_radius);

       cv::Mat croppedImage = circles_img(possible_ball);

       cv::Scalar averageColor = cv::mean(croppedImage);

       if (averageColor[0] < 30 && color_judge(croppedImage))
       {
           cv::circle(frame_clone, possible_ball_center, 1, cv::Scalar(0, 255, 0), -1, 8, 0);
           // 绘制圆轮廓
           cv::circle(frame_clone, possible_ball_center, possible_ball_radius, cv::Scalar(0, 0, 255), 3, 4, 0);
       }
   }

   /// cv::imshow("circles", frame_clone);

   result = frame_clone.clone();
   */
