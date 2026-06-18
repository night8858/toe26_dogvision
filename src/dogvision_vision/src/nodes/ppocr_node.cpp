/**
 * @file ppocr_node.cpp
 * @brief ROS2 PP-OCR 算术题识别节点
 *
 * 支持两种运行模式：
 *   - test（测试模式）：连续 OCR + 多帧投票，稳定结果追加写入 YAML
 *   - production（生产模式）：通过 /ocr/trigger 话题触发，发布 /ocr/result
 *
 * 完整流水线：
 *   1. 从海康相机获取帧
 *   2. 鱼眼去畸变（可选）
 *   3. 定位白底算术题区域（find_math_proble）
 *   4. ROI 预处理（CLAHE + 高斯模糊 + Otsu 二值化）
 *   5. PPOCR 文本检测（detect_det_ppocr）
 *   6. PPOCR 文本识别（detect_rec_ppocr，含数学字符白名单）
 *   7. 算术表达式解析（parse_simple_expr）
 *   8. 多帧滑动窗口投票（OCRMultiFrameVoter）提高稳定性
 */

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <dogvision_vision/camera/hikvision.hpp>
#include <dogvision_vision/common_structs.h>
#include <dogvision_vision/ocr_MultiFrameVoter.hpp>
#include <dogvision_vision/ocr_detect.hpp>
#include <dogvision_vision/ocr_utils.hpp>

namespace fs = std::filesystem;

namespace
{
std::atomic<bool> g_ocr_triggered{false};
} // namespace

/**
 * @brief 在取帧失败时重连海康相机。
 * @param hik 需要重连的相机封装对象。
 * @param logger 用于输出过程信息的日志对象。
 * @param max_retries 最大重连次数。
 * @retval bool 重连后能够成功获取有效帧时返回 true。
 */
static bool ensure_camera(HikGrab& hik, const rclcpp::Logger& logger, int max_retries = 5)
{
    for (int i = 0; i < max_retries; ++i)
    {
        cv::Mat test;
        if (hik.get_one_frame(test, 0) && !test.empty())
        {
            return true;
        }

        RCLCPP_WARN(logger, "Camera lost (attempt %d/%d), reconnecting...", i + 1, max_retries);
        hik.Hik_end();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        hik.Hik_init();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    RCLCPP_ERROR(logger, "Camera reconnection failed after %d attempts.", max_retries);
    return false;
}

/**
 * @brief 处理 OCR 触发消息。
 * @param msg 收到的触发消息。
 * @retval void
 */
static void ocr_trigger_callback(const std_msgs::msg::String::SharedPtr msg)
{
    (void)msg;
    g_ocr_triggered.store(true);
}

/**
 * @brief 执行一次完整的 OCR 与算术表达式解析流程。
 * @param img 输入 BGR 图像。
 * @param det 文本检测模型封装对象。
 * @param rec 文本识别模型封装对象。
 * @param logger 用于输出流程信息的日志对象。
 * @param expr_str 输出识别到的表达式字符串。
 * @param int_result 输出四舍五入后的整数结果。
 * @param mod_result 输出非负的模 4 结果。
 * @param out_roi 可选输出的算术题区域。
 * @retval bool 成功识别并解析表达式时返回 true。
 */
static bool run_ocr_pipeline(cv::Mat& img,
                             detect_det_ppocr& det,
                             detect_rec_ppocr& rec,
                             const s_detector_params& ocr_config,
                             const rclcpp::Logger& logger,
                             std::string& expr_str,
                             int& int_result,
                             int& mod_result,
                             cv::Rect2f* out_roi = nullptr)
{
    cv::Mat det_input;
    cv::Mat white_mask;  // 白色区域精确掩码，用于过滤非白色区域内的文字
    cv::Rect2f math_roi = find_math_proble(img, &white_mask);

    // math_rect 在此处统一定义，后续 ROI 裁剪和掩码过滤共用
    cv::Rect math_rect(
        static_cast<int>(math_roi.x),
        static_cast<int>(math_roi.y),
        static_cast<int>(math_roi.width),
        static_cast<int>(math_roi.height));

    if (math_roi.area() > 0.0f)
    {
        math_rect &= cv::Rect(0, 0, img.cols, img.rows);
        if (math_rect.area() > 0)
        {
            const cv::Mat raw_roi = img(math_rect);
            det_input = preprocess_math_roi(raw_roi, ocr_config);
            RCLCPP_INFO(logger, "Math ROI  : [%d, %d, %d x %d]",
                        math_rect.x, math_rect.y, math_rect.width, math_rect.height);
        }
    }

    if (det_input.empty())
    {
        if (out_roi != nullptr)
        {
            *out_roi = cv::Rect2f();
        }
        RCLCPP_INFO(logger, "No math problem region in frame, skip inference.");
        return false;
    }

    if (out_roi != nullptr)
    {
        *out_roi = math_roi;
    }

    RCLCPP_INFO(logger, "Math ROI OK, starting OCR pipeline...");
    det.preprocess(det_input);
    det.inference();
    det.postprocess();
    const std::vector<OCRBox>& all_boxes = det.ocr_det_out_;
    RCLCPP_INFO(logger, "Detected   : %zu text region(s)", all_boxes.size());

    // ── 只保留中心点落在白色区域内的检测框 ──────────────────────────────────
    // white_mask 为原图尺寸，det_input 是 math_roi 裁剪图，
    // 因此将框中心从 det_input 坐标偏移 math_rect.tl() 即可得到原图坐标。
    std::vector<OCRBox> boxes;
    for (const auto& box : all_boxes)
    {
        // 计算四点中心（ROI 坐标系）
        cv::Point2f center(0.f, 0.f);
        for (int k = 0; k < 4; ++k)
            center += box.pts[k];
        center *= 0.25f;

        // 转换到原图坐标系
        const int cx = static_cast<int>(center.x + math_rect.x);
        const int cy = static_cast<int>(center.y + math_rect.y);

        // 检查是否在白色掩码内
        if (cx >= 0 && cx < white_mask.cols &&
            cy >= 0 && cy < white_mask.rows &&
            white_mask.at<uchar>(cy, cx) > 0)
        {
            boxes.push_back(box);
        }
    }
    RCLCPP_INFO(logger, "White-filter: %zu / %zu text region(s)",
                boxes.size(), all_boxes.size());

    std::vector<OCRItem> ocr_items;
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        cv::Mat crop = crop_text_region(det_input, boxes[i]);
        if (crop.empty())
        {
            continue;
        }

        rec.preprocess(crop);
        rec.inference();
        rec.postprocess();

        if (!rec.result.empty() && !rec.result[0].text.empty())
        {
            OCRItem item;
            item.box = boxes[i];
            item.rec = rec.result[0];
            ocr_items.push_back(item);
            RCLCPP_INFO(logger, "  [%zu] \"%s\"  score=%.3f",
                        i, rec.result[0].text.c_str(), rec.result[0].score);
        }
    }

    std::string all_text;
    for (const auto& item : ocr_items)
    {
        all_text += item.rec.text + " ";
    }

    RCLCPP_INFO(logger, "All OCR    : \"%s\"", all_text.c_str());
    double calc_result = 0.0;
    if (!parse_simple_expr(all_text, calc_result, expr_str))
    {
        RCLCPP_WARN(logger, "No arithmetic expression found.");
        return false;
    }

    int_result = static_cast<int>(std::round(calc_result));
    mod_result = ((int_result % 4) + 4) % 4;
    return true;
}

/**
 * @brief 在算术题区域附近绘制识别结果。
 * @param frame 需要绘制标注的图像。
 * @param math_roi 已定位的算术题区域。
 * @param expr_str 识别到的表达式。
 * @param int_result 四舍五入后的整数结果。
 * @param mod_result 非负的模 4 结果。
 * @retval void
 */
static void draw_result_overlay(cv::Mat& frame,
                                const cv::Rect2f& math_roi,
                                const std::string& expr_str,
                                int int_result,
                                int mod_result)
{
    std::ostringstream disp;
    disp << expr_str << " = " << int_result << "  (mod4: " << mod_result << ")";
    const std::string display_text = disp.str();

    int text_x = static_cast<int>(math_roi.x);
    int text_y = static_cast<int>(math_roi.y + math_roi.height) + 40;
    text_y = std::min(text_y, frame.rows - 10);

    const double font_scale = 0.8;
    const int thickness = 2;
    int baseline = 0;
    cv::Size ts = cv::getTextSize(display_text, cv::FONT_HERSHEY_DUPLEX, font_scale, thickness, &baseline);
    if (text_x + ts.width > frame.cols)
    {
        text_x = frame.cols - ts.width - 10;
    }

    cv::Rect bg(text_x - 4, text_y - ts.height - 4, ts.width + 8, ts.height + 8);
    bg &= cv::Rect(0, 0, frame.cols, frame.rows);
    cv::rectangle(frame, bg, cv::Scalar(0, 0, 0), -1);
    cv::rectangle(frame, bg, cv::Scalar(0, 200, 0), 2);
    cv::putText(frame, display_text, cv::Point(text_x, text_y - 4),
                cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

/**
 * @brief 运行连续 OCR 测试模式，并将稳定结果变化追加写入 YAML。
 * @param node ROS2 节点对象。
 * @param hik 相机封装对象。
 * @param det 文本检测模型封装对象。
 * @param rec 文本识别模型封装对象。
 * @param yaml_path YAML 输出文件路径。
 * @param show_visual 是否启用 OpenCV 可视化窗口。
 * @retval int 类进程退出码。
 */
static int run_test_mode(const rclcpp::Node::SharedPtr& node,
                         HikGrab& hik,
                         detect_det_ppocr& det,
                         detect_rec_ppocr& rec,
                         const s_detector_params& ocr_config,
                         const std::string& yaml_path,
                         bool show_visual)
{
    auto logger = node->get_logger();
    int problem_id = 0;
    bool yaml_header_written = false;
    OCRMultiFrameVoter voter;
    cv::Rect2f stable_roi;
    bool has_stable_roi = false;

    RCLCPP_INFO(logger, "MODE       : TEST (continuous OCR + YAML output)");
    RCLCPP_INFO(logger, "YAML output: %s", yaml_path.c_str());
    RCLCPP_INFO(logger, "Press Q or ESC to exit.");

    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);

        cv::Mat frame;
        if (!hik.get_one_frame(frame, 0))
        {
            if (!ensure_camera(hik, logger))
            {
                RCLCPP_ERROR(logger, "Cannot recover camera, exiting.");
                break;
            }
            continue;
        }

        cv::Mat undistorted = undistort_image(frame);
        if (!undistorted.empty())
        {
            frame = undistorted;
        }
        else
        {
            RCLCPP_WARN(logger, "Undistortion failed, using raw frame.");
        }

        std::string expr_str;
        int int_result = 0;
        int mod_result = 0;
        cv::Rect2f math_roi;
        std::optional<OCRVoteResult> frame_result;
        if (run_ocr_pipeline(
                frame, det, rec, ocr_config, logger,
                expr_str, int_result, mod_result, &math_roi))
        {
            RCLCPP_INFO(logger, "Expr: %s  =>  %d  %%4 = %d",
                        expr_str.c_str(), int_result, mod_result);
            frame_result = OCRVoteResult{expr_str, int_result, mod_result};
            stable_roi = math_roi;
            has_stable_roi = math_roi.area() > 0.0f;
        }

        const OCRVoteEvent event = voter.update(frame_result);
        if (event == OCRVoteEvent::StableChanged)
        {
            const OCRVoteResult& stable = voter.stable_result();
            ++problem_id;

            std::ofstream ofs(yaml_path, std::ios::app);
            if (ofs.is_open())
            {
                if (!yaml_header_written)
                {
                    ofs << "ocr_results:" << std::endl;
                    yaml_header_written = true;
                }
                ofs << "  - id: " << problem_id << std::endl;
                ofs << "    question: \"" << stable.expr << "\"" << std::endl;
                ofs << "    answer: " << stable.result << std::endl;
                ofs << "    mod4: " << stable.mod4 << std::endl;
                RCLCPP_INFO(logger, "Stable OCR : %s => %d (mod4=%d)",
                            stable.expr.c_str(), stable.result, stable.mod4);
                RCLCPP_INFO(logger, "YAML write : id=%d", problem_id);
            }
            else
            {
                RCLCPP_WARN(logger, "Cannot open YAML: %s", yaml_path.c_str());
            }
        }
        else if (event == OCRVoteEvent::StableLost)
        {
            has_stable_roi = false;
            RCLCPP_INFO(logger, "Stable OCR lost after 10 invalid frames.");
        }

        if (show_visual && voter.has_stable_result() && has_stable_roi)
        {
            const OCRVoteResult& stable = voter.stable_result();
            draw_result_overlay(
                frame, stable_roi, stable.expr, stable.result, stable.mod4);
        }

        if (show_visual)
        {
            cv::imshow("Math OCR", frame);
        }

        const int key = cv::waitKey(10);
        if (key == 'q' || key == 'Q' || key == 27)
        {
            RCLCPP_INFO(logger, "Exit by user.");
            break;
        }
    }

    RCLCPP_INFO(logger, "Test mode finished. Total stable changes: %d", problem_id);
    return 0;
}

/**
 * @brief 运行基于触发的话题生产模式 OCR。
 * @param node ROS2 节点对象。
 * @param hik 相机封装对象。
 * @param det 文本检测模型封装对象。
 * @param rec 文本识别模型封装对象。
 * @param show_visual 是否启用 OpenCV 可视化窗口。
 * @retval int 类进程退出码。
 */
static int run_production_mode(const rclcpp::Node::SharedPtr& node,
                               HikGrab& hik,
                               detect_det_ppocr& det,
                               detect_rec_ppocr& rec,
                               const s_detector_params& ocr_config,
                               bool show_visual)
{
    auto logger = node->get_logger();
    auto latched_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();
    auto trigger_sub = node->create_subscription<std_msgs::msg::String>(
        "/ocr/trigger", rclcpp::QoS(1), ocr_trigger_callback);
    auto result_pub = node->create_publisher<std_msgs::msg::String>("/ocr/result", latched_qos);

    RCLCPP_INFO(logger, "MODE       : PRODUCTION (trigger-based)");
    RCLCPP_INFO(logger, "Subscribed : /ocr/trigger");
    RCLCPP_INFO(logger, "Publishing : /ocr/result (transient_local)");
    RCLCPP_INFO(logger, "Waiting for trigger...");

    OCRMultiFrameVoter voter;
    cv::Rect2f stable_roi;
    bool has_stable_roi = false;
    bool tracking_active = false;
    rclcpp::WallRate idle_rate(20);
    while (rclcpp::ok())
    {
        rclcpp::spin_some(node);
        if (g_ocr_triggered.exchange(false))
        {
            voter.reset();
            stable_roi = cv::Rect2f();
            has_stable_roi = false;
            tracking_active = true;
            RCLCPP_INFO(logger, "Trigger received, OCR tracking reset and started.");
        }

        if (!tracking_active)
        {
            idle_rate.sleep();
            continue;
        }

        cv::Mat frame;
        if (!hik.get_one_frame(frame, 0))
        {
            RCLCPP_ERROR(logger, "Failed to grab frame, reconnecting...");
            if (ensure_camera(hik, logger))
            {
                continue;
            }
            RCLCPP_ERROR(logger, "Cannot recover camera, exiting.");
            break;
        }

        cv::Mat undistorted = undistort_image(frame);
        if (!undistorted.empty())
        {
            frame = undistorted;
        }

        std::string expr_str;
        int int_result = 0;
        int mod_result = 0;
        cv::Rect2f math_roi;
        std::optional<OCRVoteResult> frame_result;
        if (run_ocr_pipeline(
                frame, det, rec, ocr_config, logger,
                expr_str, int_result, mod_result, &math_roi))
        {
            frame_result = OCRVoteResult{expr_str, int_result, mod_result};
            stable_roi = math_roi;
            has_stable_roi = math_roi.area() > 0.0f;
        }

        const OCRVoteEvent event = voter.update(frame_result);
        if (event == OCRVoteEvent::StableChanged)
        {
            const OCRVoteResult& stable = voter.stable_result();
            Json::Value result_json;
            result_json["expr"] = stable.expr;
            result_json["result"] = stable.result;
            result_json["mod4"] = stable.mod4;

            Json::FastWriter writer;
            std_msgs::msg::String msg;
            msg.data = writer.write(result_json);
            result_pub->publish(msg);
            RCLCPP_INFO(logger, "Published  : %s", msg.data.c_str());
        }
        else if (event == OCRVoteEvent::StableLost)
        {
            has_stable_roi = false;
            RCLCPP_INFO(logger, "Stable OCR lost after 10 invalid frames.");
        }

        if (show_visual)
        {
            if (voter.has_stable_result() && has_stable_roi)
            {
                const OCRVoteResult& stable = voter.stable_result();
                draw_result_overlay(
                    frame, stable_roi, stable.expr, stable.result, stable.mod4);
            }

            cv::imshow("Math OCR", frame);
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27)
            {
                RCLCPP_INFO(logger, "Exit by user.");
                break;
            }
        }
    }

    (void)trigger_sub;
    RCLCPP_INFO(logger, "Production mode finished.");
    return 0;
}

/**
 * @brief 运行 ROS2 PPOCR 节点入口。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @retval int 进程退出码。
 */
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("ppocr_node");
    auto logger = node->get_logger();

    const std::string share_dir = ament_index_cpp::get_package_share_directory("dogvision_vision");
    node->declare_parameter<std::string>("config_path", share_dir + "/config/settings.json");
    node->declare_parameter<std::string>("mode", "production");
    node->declare_parameter<bool>("show_visual", true);
    node->declare_parameter<std::string>("yaml_path", share_dir + "/data/ocr_output/ocr_results.yaml");

    const std::string config_path = node->get_parameter("config_path").as_string();
    const std::string mode = node->get_parameter("mode").as_string();
    const bool show_visual = node->get_parameter("show_visual").as_bool();
    const std::string yaml_path = node->get_parameter("yaml_path").as_string();

    Appconfig config;
    {
        detect_det_ppocr loader(nullptr);
        loader.load_config(config, config_path);
    }
    RCLCPP_INFO(logger, "Config     : %s", config_path.c_str());

    detect_det_ppocr det(&config);
    det.load_model(config.detect_config.ppocr_det_model_path, config.detect_config.det_device);

    detect_rec_ppocr rec(&config);
    rec.load_model(config.detect_config.ppocr_rec_model_path, config.detect_config.rec_device);
    rec.loda_dict(config.detect_config.rec_char_dict_path);
    rec.load_allowed_chars(config.detect_config.rec_allowed_chars_path);

    const float default_wh_ratio =
        (config.detect_config.rec_img_h > 0)
            ? static_cast<float>(config.detect_config.rec_img_w) / static_cast<float>(config.detect_config.rec_img_h)
            : 320.0f / 48.0f;
    rec.set_max_wh_ratio(default_wh_ratio);

    s_camera_params cam_params{};
    cam_params.device_id = config.hikcamera_config.device_id;
    cam_params.width = config.hikcamera_config.width;
    cam_params.height = config.hikcamera_config.height;
    cam_params.offset_x = config.hikcamera_config.offset_x;
    cam_params.offset_y = config.hikcamera_config.offset_y;
    cam_params.exposure = config.hikcamera_config.exposure;

    HikGrab hik(cam_params);
    hik.Hik_init();
    RCLCPP_INFO(logger, "Camera     : device=%d  %dx%d",
                cam_params.device_id, cam_params.width, cam_params.height);

    init_fisheye_undistort(cam_params.width, cam_params.height);

    std::error_code ec;
    fs::create_directories(fs::path(yaml_path).parent_path(), ec);
    if (ec)
    {
        RCLCPP_WARN(logger, "Cannot create YAML dir: %s", ec.message().c_str());
    }

    int ret = 0;
    if (mode == "test")
    {
        ret = run_test_mode(
            node, hik, det, rec, config.detect_config, yaml_path, show_visual);
    }
    else if (mode == "production")
    {
        ret = run_production_mode(
            node, hik, det, rec, config.detect_config, show_visual);
    }
    else
    {
        RCLCPP_ERROR(logger, "Unsupported mode '%s'. Use 'test' or 'production'.", mode.c_str());
        ret = 1;
    }

    hik.Hik_end();
    if (show_visual)
    {
        cv::destroyAllWindows();
    }
    rclcpp::shutdown();
    return ret;
}
