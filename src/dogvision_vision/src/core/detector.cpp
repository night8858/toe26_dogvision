#include <iostream>
#include <dogvision_vision/detector.hpp>
#include <dogvision_vision/common_structs.h>
#include <filesystem>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <opencv2/calib3d.hpp>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------
// 鱼眼去畸变配置区
// ---------------------------------------------------------------
// 是否启用鱼眼去畸变。
// 方式一：取消下面这行的注释
// 方式二：在 CMakeLists.txt 的编译选项中添加 -DENABLE_FISHEYE_UNDISTORT
//
// 未定义此宏时，diatorion() 将直接返回原图拷贝。
#define ENABLE_FISHEYE_UNDISTORT

// balance 取值范围通常为 [0, 1]：
// 1) 越接近 0：裁切更多，黑边更少，视场更窄。
// 2) 越接近 1：保留更多原始视场，黑边可能更多。
constexpr double kFisheyeBalance = 0.0;

// fov_scale > 1 会扩大视场，< 1 会缩小视场；通常保持 1.0。
constexpr double kFisheyeFovScale = 1.0;

// --- 填充 K 矩阵 (3x3) ---
const cv::Mat K = (cv::Mat_<double>(3, 3) << 8.2631010840557929e+02, 0., 7.3508237365876721e+02,
                   0., 8.3234495506807673e+02, 5.6784864582942498e+02,
                   0., 0., 1.);

// --- 填充 D 矩阵 (4x1) ---
const cv::Mat D = (cv::Mat_<double>(4, 1) << 1.9474519085664992e-02,
                   2.2096711413330011e-02,
                   -4.1006640770500716e-02,
                   2.6220651979250005e-02);
// 用于处理畸变
cv::Mat map1, map2;
cv::Mat newCameraMatrix;
bool g_fisheye_map_ready = false;

void detector::load_config(Appconfig &config, std::string json_file_path)
{
    Json::Reader reader;
    Json::Value value;
    std::ifstream in(json_file_path, std::ios::binary);
    std::cout << "load json now..." << std::endl;
    if (!in.is_open())
    {
        std::cerr << "Failed to open file: " << json_file_path;
        exit(1);
    }
    if (reader.parse(in, value))
    {
        const std::filesystem::path config_file(json_file_path);
        std::filesystem::path pkg = config_file.parent_path();
        if (pkg.filename() == "config")
        {
            pkg = pkg.parent_path();
        }
        auto resolve = [&](const std::string &p) -> std::string
        {
            if (p.empty() || p[0] == '/')
                return p; // 已是绝对路径则直接使用
            const std::filesystem::path direct = pkg / p;
            if (std::filesystem::exists(direct))
                return direct.string();

            const std::string source_prefix = "src/dogvision_vision/";
            if (p.rfind(source_prefix, 0) == 0)
                return (pkg / p.substr(source_prefix.size())).string();
            return direct.string();
        };

        config.detect_config.bin_file_path = resolve(value["path"]["openvino_bin_file_path"].asString());
        config.detect_config.xml_file_path = resolve(value["path"]["openvino_xml_file_path"].asString());
        const std::string openvino_device =
            value["path"].get("openvino_device", "CPU").asString();
        if (openvino_device.empty())
            throw std::invalid_argument("path.openvino_device must not be empty");
        auto read_device = [&](const char* key) -> std::string
        {
            const std::string device =
                value["path"].get(key, openvino_device).asString();
            if (device.empty())
                throw std::invalid_argument(
                    std::string("path.") + key + " must not be empty");
            return device;
        };
        config.detect_config.yolo_device = read_device("yolo_device");
        config.detect_config.det_device = read_device("ppocr_det_device");
        config.detect_config.rec_device = read_device("ppocr_rec_device");
        config.detect_config.cls_device = read_device("ppocr_cls_device");

        config.detect_config.ppocr_det_model_path = resolve(value["path"]["ppocr_det_model_path"].asString());
        config.detect_config.ppocr_rec_model_path = resolve(value["path"]["ppocr_rec_model_path"].asString());
        config.detect_config.ppocr_cls_model_path = resolve(value["path"]["ppocr_cls_model_path"].asString());
        config.detect_config.ppocr_det_model_xml_path =
            resolve(value["path"]["ppocr_det_model_xml_path"].asString());
        config.detect_config.ppocr_det_model_bin_path =
            resolve(value["path"]["ppocr_det_model_bin_path"].asString());
        config.detect_config.ppocr_rec_model_xml_path =
            resolve(value["path"]["ppocr_rec_model_xml_path"].asString());
        config.detect_config.ppocr_rec_model_bin_path =
            resolve(value["path"]["ppocr_rec_model_bin_path"].asString());
        config.detect_config.rec_char_dict_path = resolve(value["path"]["ppocr_dict_path"].asString());
        config.detect_config.rec_allowed_chars_path =
            resolve(value["path"]["ppocr_allowed_chars_path"].asString());

        const Json::Value& math_filter = value["ocr_math_filter"];
        if (!math_filter.isNull())
        {
            config.detect_config.ocr_math_use_grayscale =
                math_filter.get("use_grayscale", false).asBool();
            config.detect_config.ocr_roi_enabled =
                math_filter.get("roi_enabled", false).asBool();
            config.detect_config.ocr_roi_quadrant =
                math_filter.get("roi_quadrant", "full").asString();
            config.detect_config.ocr_roi_mode =
                math_filter.get(
                    "roi_mode",
                    config.detect_config.ocr_roi_quadrant == "full"
                        ? "full"
                        : "quadrant").asString();
            const Json::Value& roi_ratio = math_filter["roi_rect_ratio"];
            if (!roi_ratio.isNull())
            {
                config.detect_config.ocr_roi_rect_ratio = cv::Rect2d(
                    roi_ratio.get("x", 0.0).asDouble(),
                    roi_ratio.get("y", 0.0).asDouble(),
                    roi_ratio.get("w", 1.0).asDouble(),
                    roi_ratio.get("h", 1.0).asDouble());
            }
            config.detect_config.ocr_math_min_surround_white_ratio =
                math_filter.get("min_surround_white_ratio", 0.50).asDouble();
            config.detect_config.ocr_math_surround_margin_ratio =
                math_filter.get("surround_margin_ratio", 0.50).asDouble();
            config.detect_config.ocr_math_white_s_max =
                math_filter.get("white_s_max", 110).asInt();
            config.detect_config.ocr_math_white_v_min =
                math_filter.get("white_v_min", 50).asInt();
        }

        const double min_white =
            config.detect_config.ocr_math_min_surround_white_ratio;
        if (min_white < 0.0 || min_white > 1.0)
            throw std::invalid_argument(
                "ocr_math_filter.min_surround_white_ratio must be in [0, 1]");
        if (config.detect_config.ocr_math_surround_margin_ratio <= 0.0 ||
            config.detect_config.ocr_math_surround_margin_ratio > 1.0)
            throw std::invalid_argument(
                "ocr_math_filter.surround_margin_ratio must be in (0, 1]");
        if (config.detect_config.ocr_math_white_s_max < 0 ||
            config.detect_config.ocr_math_white_s_max > 255)
            throw std::invalid_argument(
                "ocr_math_filter.white_s_max must be in [0, 255]");
        if (config.detect_config.ocr_math_white_v_min < 0 ||
            config.detect_config.ocr_math_white_v_min > 255)
            throw std::invalid_argument(
                "ocr_math_filter.white_v_min must be in [0, 255]");
        const std::string& roi_quadrant =
            config.detect_config.ocr_roi_quadrant;
        const std::string& roi_mode =
            config.detect_config.ocr_roi_mode;
        if (roi_mode != "full" &&
            roi_mode != "quadrant" &&
            roi_mode != "ratio")
        {
            throw std::invalid_argument(
                "ocr_math_filter.roi_mode must be one of full/quadrant/ratio");
        }
        if (roi_quadrant != "full" &&
            roi_quadrant != "top_left" &&
            roi_quadrant != "top_right" &&
            roi_quadrant != "bottom_left" &&
            roi_quadrant != "bottom_right")
        {
            throw std::invalid_argument(
                "ocr_math_filter.roi_quadrant must be one of "
                "full/top_left/top_right/bottom_left/bottom_right");
        }
        if (roi_mode == "ratio")
        {
            if (math_filter["roi_rect_ratio"].isNull())
            {
                throw std::invalid_argument(
                    "ocr_math_filter.roi_rect_ratio is required when roi_mode=ratio");
            }
            const cv::Rect2d& ratio =
                config.detect_config.ocr_roi_rect_ratio;
            constexpr double eps = 1e-9;
            if (ratio.x < 0.0 || ratio.x >= 1.0 ||
                ratio.y < 0.0 || ratio.y >= 1.0 ||
                ratio.width <= 0.0 || ratio.width > 1.0 ||
                ratio.height <= 0.0 || ratio.height > 1.0 ||
                ratio.x + ratio.width > 1.0 + eps ||
                ratio.y + ratio.height > 1.0 + eps)
            {
                throw std::invalid_argument(
                    "ocr_math_filter.roi_rect_ratio requires x/y in [0,1), "
                    "w/h in (0,1], and x+w/y+h <= 1");
            }
        }

        // ── OCR 测试模式窗口显示参数 ──
        const Json::Value& test_visual = value["ocr_test_visualization"];
        if (!test_visual.isNull())
        {
            config.detect_config.ocr_test_show_visual =
                test_visual.get("show_visual", true).asBool();
            config.detect_config.ocr_test_show_ocr_roi =
                test_visual.get("show_ocr_roi", true).asBool();
            config.detect_config.ocr_test_show_debug_panels =
                test_visual.get("show_debug_panels", true).asBool();
        }

        // ── YOLO 图像增强参数 ──
        const Json::Value& yolo_enh = value["yolo_enhance"];
        if (!yolo_enh.isNull())
        {
            config.detect_config.yolo_enhance_enabled =
                yolo_enh.get("enabled", true).asBool();
            config.detect_config.yolo_enhance_clahe_clip_limit =
                yolo_enh.get("clahe_clip_limit", 2.0).asFloat();
            config.detect_config.yolo_enhance_clahe_tile_grid_size =
                yolo_enh.get("clahe_tile_grid_size", 8).asInt();
            config.detect_config.yolo_enhance_saturation_scale =
                yolo_enh.get("saturation_scale", 1.3).asFloat();
        }
        if (config.detect_config.yolo_enhance_clahe_clip_limit <= 0.0f)
            throw std::invalid_argument(
                "yolo_enhance.clahe_clip_limit must be > 0");
        if (config.detect_config.yolo_enhance_clahe_tile_grid_size < 1 ||
            config.detect_config.yolo_enhance_clahe_tile_grid_size > 32)
            throw std::invalid_argument(
                "yolo_enhance.clahe_tile_grid_size must be in [1, 32]");
        if (config.detect_config.yolo_enhance_saturation_scale <= 0.0f)
            throw std::invalid_argument(
                "yolo_enhance.saturation_scale must be > 0");

        const Json::Value& output_save = value["output_save"];
        if (!output_save.isNull())
        {
            config.detect_config.save_ppocr_video =
                output_save.get("save_ppocr_video", true).asBool();
            config.detect_config.ppocr_video_save_dir =
                resolve(output_save.get(
                    "ppocr_video_save_dir",
                    "src/dogvision_vision/data/ocr_output/video").asString());
            config.detect_config.ppocr_video_fps =
                output_save.get("ppocr_video_fps", 20.0).asDouble();
            config.detect_config.max_ppocr_videos =
                output_save.get("max_ppocr_videos", 10).asInt();
            config.detect_config.save_ocr_result_images =
                output_save.get("save_ocr_result_images", true).asBool();
            config.detect_config.ocr_result_image_dir =
                resolve(output_save.get(
                    "ocr_result_image_dir",
                    "src/dogvision_vision/data/ocr_debug/auto").asString());
            config.detect_config.max_ocr_result_images =
                output_save.get("max_ocr_result_images", 30).asInt();
            config.detect_config.save_yolo_test_video =
                output_save.get("save_yolo_test_video", true).asBool();
        }
        else
        {
            config.detect_config.save_ppocr_video = true;
            config.detect_config.ppocr_video_save_dir =
                resolve("src/dogvision_vision/data/ocr_output/video");
            config.detect_config.max_ppocr_videos = 10;
            config.detect_config.ocr_result_image_dir =
                resolve("src/dogvision_vision/data/ocr_debug/auto");
            config.detect_config.max_ocr_result_images = 30;
            config.detect_config.save_yolo_test_video = true;
        }
        if (config.detect_config.ppocr_video_fps <= 0.0)
            throw std::invalid_argument(
                "output_save.ppocr_video_fps must be > 0");
        if (config.detect_config.max_ppocr_videos <= 0)
            throw std::invalid_argument(
                "output_save.max_ppocr_videos must be > 0");
        if (config.detect_config.max_ocr_result_images <= 0)
            throw std::invalid_argument(
                "output_save.max_ocr_result_images must be > 0");

        config.detect_config.batch_size = value["NCHW"]["batch_size"].asInt();
        config.detect_config.c = value["NCHW"]["C"].asInt();
        config.detect_config.w = value["NCHW"]["W"].asInt();
        config.detect_config.h = value["NCHW"]["H"].asInt();

        config.detect_config.type = value["img"]["type"].asInt();
        config.detect_config.width = value["img"]["width"].asInt();
        config.detect_config.height = value["img"]["height"].asInt();

        const Json::Value& lens_distortion = value["lens_distortion"];
        if (!lens_distortion.isNull())
        {
            config.detect_config.enable_undistort =
                lens_distortion.get("enable_undistort", true).asBool();
        }

        config.detect_config.nms_thresh = value["thresh"]["nms_thresh"].asFloat();
        config.detect_config.bbox_conf_thresh = value["thresh"]["bbox_conf_thresh"].asFloat();
        config.detect_config.merge_thresh = value["thresh"]["merge_thresh"].asFloat();

        config.detect_config.classes = value["nums"]["classes"].asInt();
        config.detect_config.class0 = value["nums"]["cls0"].asString();
        config.detect_config.class1 = value["nums"]["cls1"].asString();
        config.detect_config.class2 = value["nums"]["cls2"].asString();
        config.detect_config.class3 = value["nums"]["cls3"].asString();

        const Json::Value& camera = value["camera"];
        if (!camera.isNull())
        {
            config.camera_type = camera.get("type", "usb").asString();
            config.usb_camera_index = camera.get("usb_index", 0).asInt();
        }
        else
        {
            config.camera_type = "usb";
            config.usb_camera_index = 0;
        }
        if (config.camera_type != "usb")
        {
            throw std::invalid_argument(
                "camera.type must be \"usb\"; Hik/MVS camera support is disabled");
        }
        if (config.usb_camera_index < 0 || config.usb_camera_index >= 4)
        {
            throw std::invalid_argument(
                "camera.usb_index must be in [0, 3]");
        }

        // Hik/MVS is currently disabled. Keep this config mapping for future restore.
        // config.hikcamera_config.device_id = value["hikcamera"]["device_id"].asInt();
        // config.hikcamera_config.exposure = value["hikcamera"]["exposure"].asInt();
        // config.hikcamera_config.height = value["hikcamera"]["height"].asInt();
        // config.hikcamera_config.width = value["hikcamera"]["width"].asInt();
        // config.hikcamera_config.offset_x = value["hikcamera"]["offset_x"].asInt();
        // config.hikcamera_config.offset_y = value["hikcamera"]["offset_y"].asInt();

        for (int i = 0; i < 4; ++i)
        {
            const std::string key = "usbcamera" + std::to_string(i);
            config.usbcamera_config[i].device_path =
                value[key].get("device_path", "").asString();
            config.usbcamera_config[i].device_id = value[key]["device_id"].asInt();
            config.usbcamera_config[i].width = value[key]["width"].asInt();
            config.usbcamera_config[i].height = value[key]["height"].asInt();
            config.usbcamera_config[i].fps = value[key].get("FPS", 120).asInt();
        }

#ifdef TWO_CAMERAS
        // 此处可补充多个相机的初始化
#endif
    }
    else
    {
        std::cerr << "Load Json Error!!!" << std::endl;
        exit(1);
    }
    std::cout << "load json success" << std::endl;

    // -----------------------------------------------------------
    // 鱼眼去畸变映射初始化（只需在配置加载后执行一次）
    // -----------------------------------------------------------
    // 这里使用 OpenCV 的 cv::fisheye 专用模型，而不是普通 pinhole 模型。
    // 原因：鱼眼镜头的畸变形式与普通镜头不同，使用 fisheye 模型更稳定。
#ifdef ENABLE_FISHEYE_UNDISTORT
    if (!config.detect_config.enable_undistort)
    {
        g_fisheye_map_ready = false;
        std::cout << "fisheye undistort disabled by settings" << std::endl;
        return;
    }

    const int camera_width =
        config.usbcamera_config[config.usb_camera_index].width;
    const int camera_height =
        config.usbcamera_config[config.usb_camera_index].height;
    const cv::Size image_size(camera_width, camera_height);
    if (image_size.width <= 0 || image_size.height <= 0)
    {
        g_fisheye_map_ready = false;
        std::cerr << "fisheye init failed: invalid image size "
                  << image_size.width << "x" << image_size.height << std::endl;
        return;
    }

    // 1) 根据 K/D 与目标输出尺寸估计新的相机内参矩阵。
    // 2) R 使用单位阵（不做额外旋转）。
    // 3) balance 与 fov_scale 控制有效视场和黑边。
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        image_size,
        cv::Matx33d::eye(),
        newCameraMatrix,
        kFisheyeBalance,
        image_size,
        kFisheyeFovScale);

    // 预计算重映射表：运行时每帧只需要 remap，速度更稳定。
    // 这里使用 CV_16SC2，通常比 CV_32FC1 更省内存、更适合实时场景。
    cv::fisheye::initUndistortRectifyMap(
        K,
        D,
        cv::Matx33d::eye(),
        newCameraMatrix,
        image_size,
        CV_16SC2,
        map1,
        map2);

    g_fisheye_map_ready = !map1.empty() && !map2.empty();
    if (g_fisheye_map_ready)
    {
        std::cout << "fisheye distortion init success" << std::endl;
    }
    else
    {
        std::cerr << "fisheye distortion init failed: map is empty" << std::endl;
    }
#else
    g_fisheye_map_ready = false;
    std::cout << "fisheye undistort disabled" << std::endl;
#endif

}

/**
 * @brief 检测器构造函数
 *
 * 初始化图像标志位并从 config 中复制所有检测与 OCR 参数到本地成员。
 * config 为 nullptr 时仅置零标志位并提前返回，用于仅加载配置的场景。
 *
 * @param config 应用配置指针；可为 nullptr
 */
detector::detector(Appconfig *config)
{
    // 清零各相机的新图像标志位
    hik_img_flag = 0;
    for (int i = 0; i < 4; ++i)
    {
        usb_img_flag[i] = 0;
    }

    if (config == nullptr)
    {
        return;
    }

    // ── 复制模型路径 ──
    detect_config_.xml_file_path = config->detect_config.xml_file_path;
    detect_config_.bin_file_path = config->detect_config.bin_file_path;

    // ── 复制网络输入尺寸 ──
    detect_config_.batch_size = config->detect_config.batch_size;
    detect_config_.h = config->detect_config.h;
    detect_config_.w = config->detect_config.w;
    detect_config_.c = config->detect_config.c;

    // ── 复制图像属性 ──
    detect_config_.type = config->detect_config.type;
    detect_config_.width = config->detect_config.width;
    detect_config_.height = config->detect_config.height;

    // ── 复制阈值参数 ──
    detect_config_.nms_thresh = config->detect_config.nms_thresh;
    detect_config_.bbox_conf_thresh = config->detect_config.bbox_conf_thresh;
    detect_config_.merge_thresh = config->detect_config.merge_thresh;
    detect_config_.classes = config->detect_config.classes;

    // ── 复制 PPOCR 相关路径与参数 ──
    detect_config_.ppocr_det_model_path = config->detect_config.ppocr_det_model_path;
    detect_config_.ppocr_rec_model_path = config->detect_config.ppocr_rec_model_path;
    detect_config_.ppocr_cls_model_path = config->detect_config.ppocr_cls_model_path;
    detect_config_.ppocr_det_model_xml_path =
        config->detect_config.ppocr_det_model_xml_path;
    detect_config_.ppocr_det_model_bin_path =
        config->detect_config.ppocr_det_model_bin_path;
    detect_config_.ppocr_rec_model_xml_path =
        config->detect_config.ppocr_rec_model_xml_path;
    detect_config_.ppocr_rec_model_bin_path =
        config->detect_config.ppocr_rec_model_bin_path;
    detect_config_.rec_char_dict_path = config->detect_config.rec_char_dict_path;
    detect_config_.rec_allowed_chars_path = config->detect_config.rec_allowed_chars_path;
    detect_config_.yolo_device =
        config->detect_config.yolo_device;
    detect_config_.det_device =
        config->detect_config.det_device;
    detect_config_.rec_device =
        config->detect_config.rec_device;
    detect_config_.cls_device =
        config->detect_config.cls_device;
    detect_config_.ocr_math_use_grayscale =
        config->detect_config.ocr_math_use_grayscale;
    detect_config_.ocr_roi_enabled =
        config->detect_config.ocr_roi_enabled;
    detect_config_.ocr_roi_mode =
        config->detect_config.ocr_roi_mode;
    detect_config_.ocr_roi_quadrant =
        config->detect_config.ocr_roi_quadrant;
    detect_config_.ocr_roi_rect_ratio =
        config->detect_config.ocr_roi_rect_ratio;
    detect_config_.ocr_math_min_surround_white_ratio =
        config->detect_config.ocr_math_min_surround_white_ratio;
    detect_config_.ocr_math_surround_margin_ratio =
        config->detect_config.ocr_math_surround_margin_ratio;
    detect_config_.ocr_math_white_s_max =
        config->detect_config.ocr_math_white_s_max;
    detect_config_.ocr_math_white_v_min =
        config->detect_config.ocr_math_white_v_min;

    // ── 复制类别名称 ──
    detect_config_.class0 = config->detect_config.class0;
    detect_config_.class1 = config->detect_config.class1;
    detect_config_.class2 = config->detect_config.class2;
    detect_config_.class3 = config->detect_config.class3;

    // ── 复制 YOLO 图像增强参数 ──
    detect_config_.yolo_enhance_enabled = config->detect_config.yolo_enhance_enabled;
    detect_config_.yolo_enhance_clahe_clip_limit = config->detect_config.yolo_enhance_clahe_clip_limit;
    detect_config_.yolo_enhance_clahe_tile_grid_size = config->detect_config.yolo_enhance_clahe_tile_grid_size;
    detect_config_.yolo_enhance_saturation_scale = config->detect_config.yolo_enhance_saturation_scale;
    detect_config_.enable_undistort = config->detect_config.enable_undistort;
    detect_config_.save_ppocr_video = config->detect_config.save_ppocr_video;
    detect_config_.ppocr_video_save_dir = config->detect_config.ppocr_video_save_dir;
    detect_config_.ppocr_video_fps = config->detect_config.ppocr_video_fps;
    detect_config_.max_ppocr_videos = config->detect_config.max_ppocr_videos;
    detect_config_.save_ocr_result_images = config->detect_config.save_ocr_result_images;
    detect_config_.ocr_result_image_dir = config->detect_config.ocr_result_image_dir;
    detect_config_.max_ocr_result_images = config->detect_config.max_ocr_result_images;
    detect_config_.save_yolo_test_video = config->detect_config.save_yolo_test_video;
}

detector::~detector()
{
}

const std::vector<Detection> *detector::yolo_results_ptr() const
{
    return nullptr;
}

/**
 * @brief 将相机采集的图像推入检测器内部缓存
 *
 * cam_id = 0 对应海康相机，将图像克隆后同时存入环形缓存队列
 * （维护最大 max_size_ 帧）和单帧缓存，并置位新图像标志。
 * cam_id 1~4（USB 相机）的处理逻辑暂被注释，留作扩展。
 *
 * @param grab_img 需缓存的图像（BGR 格式）
 * @param cam_id   相机编号（0 = 海康，1-4 = USB）
 */
void detector::push_img(cv::Mat &grab_img, int cam_id)
{
    if (cam_id == 0)
    {
        // 海康相机图像入队
        {
            // 自动加锁，离开作用域自动解锁
            std::lock_guard<std::mutex> lock(hik_img_mutex_);

            // 环形队列：超出最大数量时移除最旧的一帧
            if (input_imgs_hikvion.size() >= max_size_)
            {
                input_imgs_hikvion.erase(input_imgs_hikvion.begin());
            }
            input_imgs_hikvion.push_back(grab_img.clone());

            // 更新单帧缓存并置标志位
            input_img_hik_ = grab_img.clone();
            hik_img_flag = 1; // 标记有新帧可用
        }
    }
    // USB 相机部分暂未启用，代码已注释保留
}

// 处理使用广角镜头后的畸变
cv::Mat detector::diatorion(cv::Mat &input_img)
{
    // 输入为空时直接返回空 Mat，避免后续 remap 触发异常。
    if (input_img.empty())
    {
        return cv::Mat();
    }

    // 未初始化成功时不做去畸变，返回原图拷贝，保证主流程可继续运行。
#ifdef ENABLE_FISHEYE_UNDISTORT
    if (!g_fisheye_map_ready)
    {
        return input_img.clone();
    }

    // 运行期通过 remap 执行去畸变：
    // 1) map1/map2 在 load_config 中已按 fisheye 模型预计算。
    // 2) INTER_LINEAR 提供较平滑插值效果，适合视觉检测前处理。
    // 3) BORDER_CONSTANT 处理边缘空洞区域。
    cv::Mat undistorted_image;
    cv::remap(
        input_img,          // 输入：原始畸变图像
        undistorted_image,  // 输出：去畸变后的图像
        map1,               // 输入：x坐标映射图
        map2,               // 输入：y坐标映射图
        cv::INTER_LINEAR,   // 输入：插值方法
        cv::BORDER_CONSTANT // 输入：边界填充模式
    );
    return undistorted_image;
#else
    return input_img.clone();
#endif
}

void detector::show_yolo_result(cv::Mat &show_img, const Detection &det)
{
    if (show_img.empty())
    {
        return;
    }

    // 在show_img上绘制检测结果det
    // 绘制边界框和类别标签

    // 提取边界框坐标 (假设bbox[4]为 x, y, width, height)
    int x = static_cast<int>(det.bbox[0]);
    int y = static_cast<int>(det.bbox[1]);
    int width = static_cast<int>(det.bbox[2]);
    int height = static_cast<int>(det.bbox[3]);

    // 计算右下角坐标
    int x2 = x + width;
    int y2 = y + height;

    // 确保坐标在图像范围内
    x = std::max(0, x);
    y = std::max(0, y);
    x2 = std::min(show_img.cols, x2);
    y2 = std::min(show_img.rows, y2);

    if (x2 <= x || y2 <= y)
    {
        return;
    }

    // 根据类别ID选择颜色
    cv::Scalar color;
    int class_id = static_cast<int>(det.class_id);
    switch (class_id % 5) // 5种颜色循环
    {
    case 0:
        color = cv::Scalar(0, 255, 0); // 绿色 (BGR格式)
        break;
    case 1:
        color = cv::Scalar(255, 0, 0); // 蓝色
        break;
    case 2:
        color = cv::Scalar(0, 0, 255); // 红色
        break;
    case 3:
        color = cv::Scalar(255, 255, 0); // 青色
        break;
    case 4:
        color = cv::Scalar(255, 0, 255); // 紫色
        break;
    default:
        color = cv::Scalar(0, 255, 255); // 黄色
        break;
    }

    // 绘制边界框
    int thickness = 2;
    cv::rectangle(show_img, cv::Point(x, y), cv::Point(x2, y2), color, thickness);

    // 准备标签文本 (类别 + 置信度)
    std::string label = "Class: " + std::to_string(class_id) +
                        " Conf: " + std::to_string(det.conf).substr(0, 4);

    // 获取文本大小以用于背景矩形
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5;
    int font_thickness = 1;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, font, font_scale, font_thickness, &baseline);

    // 绘制标签背景矩形
    const int text_top = std::max(0, y - text_size.height - 5);
    const int text_bottom = std::max(0, y);
    cv::rectangle(show_img,
                  cv::Point(x, text_top),
                  cv::Point(x + text_size.width, text_bottom),
                  color, -1); // 填充矩形

    // 绘制文本标签
    cv::putText(show_img, label, cv::Point(x, std::max(0, y - 5)),
                font, font_scale, cv::Scalar(255, 255, 255), font_thickness);
}

bool detector::yolo_run(cv::Mat &input_img, std::vector<Detection> &res)
{
    if (input_img.empty())
    {
        res.clear();
        return false;
    }

    std::lock_guard<std::mutex> lock(yolo_infer_mutex_);

    preprocess(input_img);
    inference();
    postprocess();

    const std::vector<Detection> *dets = yolo_results_ptr();
    if (dets == nullptr)
    {
        res.clear();
        return false;
    }

    res = *dets;
    return !res.empty();
}

void detector::show_ocr_result(void)
{
    // OCR结果显示函数
    // 这里可以实现对OCR结果的可视化，例如在图像上绘制识别的文本等
}

bool detector::get_ocr_result(void)
{
    // OCR结果处理函数
    // 这里可以实现对OCR结果的后处理，例如文本识别、结果过滤等
    return true; // 返回处理结果
}
