#pragma once 

#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector>

#include <dogvision_vision/common_structs.h>



class detector
{
public:
    /**
     * @brief 使用可选应用配置构造检测器基类。
     * @param config 需要复制到检测器的配置；为 nullptr 时仅用于加载配置。
     * @retval 无
     */
    detector(Appconfig* config);

    /**
     * @brief 析构检测器基类。
     * @param 无
     * @retval 无
     */
    virtual ~detector();

    /**
     * @brief 将图像写入内部相机缓存。
     * @param giab_img 需要克隆到缓存中的图像。
     * @param cam_id 相机编号，0 表示海康相机，1-4 表示 USB 相机。
     * @retval void
     */
    void push_img(cv::Mat &giab_img , int cam_id );

    /**
     * @brief 在图像上绘制 YOLO 检测结果。
     * @param show_img 需要原地绘制的图像。
     * @param det 需要绘制的检测结果。
     * @retval void
     */
    void show_yolo_result(cv::Mat &show_img , const Detection &det);

    /**
     * @brief OCR 可视化占位函数。
     * @param 无
     * @retval void
     */
    void show_ocr_result(void);

    /**
     * @brief 对图像应用已配置的鱼眼去畸变。
     * @param show_img 输入图像。
     * @retval cv::Mat 去畸变后的图像；映射表不可用时返回输入图像克隆。
     */
    cv::Mat diatorion(cv::Mat &show_img);

    /**
     * @brief 对单张图像执行完整 YOLO 推理流程。
     * @param input_img 需要处理的图像。
     * @param res 输出后处理后的检测结果。
     * @retval bool 至少返回一个检测结果时返回 true。
     */
    bool yolo_run(cv::Mat &input_img , std::vector<Detection> &res);

    /**
     * @brief OCR 结果获取占位函数。
     * @param 无
     * @retval bool 占位流程成功时返回 true。
     */
    bool get_ocr_result(void);

    /**
     * @brief 从 JSON 文件加载应用配置。
     * @param config 输出配置对象。
     * @param json_file_path JSON 配置文件路径。
     * @retval void
     */
    void load_config(Appconfig& config, std::string json_file_path);

    
protected:

    /**
     * @brief 获取最近一次 YOLO NMS 结果指针（供子类覆盖）
     * @return 检测结果指针；基类返回 nullptr
     */
    virtual const std::vector<Detection>* yolo_results_ptr() const;

    // 以下纯虚函数由各子类（YOLO / PPOCR）分别实现
    virtual void preprocess(cv::Mat &src) = 0;  ///< 图像预处理
    virtual void inference() = 0;               ///< 模型推理
    virtual void postprocess() = 0;             ///< 输出后处理

    const int max_size_ = 10;          ///< 图像缓存队列最大长度

    std::vector<cv::Mat> input_imgs_hikvion;  ///< 海康相机帧队列
    std::vector<cv::Mat> input_imgs_usb[4];   ///< USB 相机帧队列（最多 4 路）

    std::mutex hik_img_mutex_;          ///< 海康帧队列互斥锁
    std::mutex usb_img_mutex_[4];       ///< USB 帧队列互斥锁
    std::mutex yolo_infer_mutex_;       ///< YOLO 推理互斥锁（防止并发推理）

    int hik_img_flag;    ///< 海康相机新图像标志位（1=有新图）
    int usb_img_flag[4]; ///< USB 相机新图像标志位

    cv::Mat input_img_hik_;    ///< 海康相机当前帧缓存
    cv::Mat input_img_usb_[4]; ///< USB 相机当前帧缓存

    cv::Mat show_img_hik;    ///< 海康相机可视化帧
    cv::Mat show_img_usb[4]; ///< USB 相机可视化帧

    s_detector_params detect_config_; ///< 检测器本地配置副本

};
