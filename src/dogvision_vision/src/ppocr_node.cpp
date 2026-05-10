// ============================================================================
// ppocr_node.cpp — PPOCR 算术题识别节点（双模式）
//
// 模式切换：通过下方 #define 选择
//   OCR_TEST_MODE        测试版：摄像头连续识别 → 去重 → YAML 输出
//   OCR_PRODUCTION_MODE  正式版：等待 /ocr/trigger → 识别 → /ocr/result 发布
//
// 默认激活正式版。切换到测试版请取消 OCR_PRODUCTION_MODE 的注释
// 并注释掉 OCR_TEST_MODE。
// ============================================================================

#define OCR_TEST_MODE          // 🔬 测试版
//#define OCR_PRODUCTION_MODE       // 🚀 正式版

#define OCR_SHOW_VISUAL           // 🖼️ 显示可视化窗口（注释掉则静默运行）

// ── 公共头文件 ──────────────────────────────────────────────────────────────
#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>

#include <dogvision_vision/common_structs.h>
#include <dogvision_vision/ocr_detect.hpp>
#include <dogvision_vision/ocr_utils.hpp>

// ── 模式专用头文件 ──────────────────────────────────────────────────────────
#ifdef OCR_TEST_MODE
  #include <unordered_set>
#endif

#ifdef OCR_PRODUCTION_MODE
  #include <std_msgs/String.h>
  #include <jsoncpp/json/json.h>
#endif

// 两版都需要摄像头
#include <dogvision_camera/hikvision.hpp>

#include <thread>
#include <chrono>   // 用于 2s 间隔控制

#include <filesystem>
namespace fs = std::filesystem;

// ── 相机自动重连辅助 ────────────────────────────────────────────────────────
static bool ensure_camera(HikGrab &hik, int max_retries = 5)
{
    for (int i = 0; i < max_retries; ++i)
    {
        cv::Mat test;
        if (hik.get_one_frame(test, 0) && !test.empty())
            return true;

        ROS_WARN("Camera lost (attempt %d/%d), reconnecting...",
                 i + 1, max_retries);
        hik.Hik_end();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        hik.Hik_init();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    ROS_ERROR("Camera reconnection failed after %d attempts.", max_retries);
    return false;
}

// ── 正式版全局状态 ─────────────────────────────────────────────────────────
#ifdef OCR_PRODUCTION_MODE
static bool g_ocr_triggered = false;

static void ocr_trigger_callback(const std_msgs::String::ConstPtr &msg)
{
    ROS_INFO("Trigger received: \"%s\"", msg->data.c_str());
    g_ocr_triggered = true;
}
#endif

// ============================================================================
// 工具：执行一次完整的 OCR + 算术解析流水线
// 输入：BGR 图像
// 输出：表达式字符串、计算结果、mod4
// 返回：true 表示成功识别并解析出算术表达式
// ============================================================================
static bool run_ocr_pipeline(cv::Mat &img,
                             detect_det_ppocr &det,
                             detect_rec_ppocr &rec,
                             std::string &expr_str,
                             int &int_result,
                             int &mod_result,
                             cv::Rect2f *out_roi = nullptr)
{
    // NOTE: 去畸变已在主循环中提前执行，img 已是矫正后的图像

    // ── 1. 定位白底算术题 ROI — 找不到则跳过推理 ───────────────────────────
    cv::Mat det_input;
    cv::Rect2f math_roi = find_math_proble(img);
    if (math_roi.area() > 0.f)
    {
        cv::Rect math_rect(
            static_cast<int>(math_roi.x),
            static_cast<int>(math_roi.y),
            static_cast<int>(math_roi.width),
            static_cast<int>(math_roi.height));
        math_rect &= cv::Rect(0, 0, img.cols, img.rows);
        if (math_rect.area() > 0)
        {
            det_input = img(math_rect).clone();
            ROS_INFO("Math ROI  : [%d, %d, %d x %d]",
                     math_rect.x, math_rect.y, math_rect.width, math_rect.height);
        }
    }

    // 没有白底区域 → 跳过本次推理
    if (det_input.empty())
    {
        ROS_INFO_THROTTLE(5, "No math problem region in frame, skip inference.");
        if (out_roi) *out_roi = cv::Rect2f();
        return false;
    }

    // 输出 ROI 供调用方在图像上绘制结果
    if (out_roi) *out_roi = math_roi;

    ROS_INFO("Math ROI OK, starting OCR pipeline...");

    // ── 2. 文本检测 ─────────────────────────────────────────────────────────
    det.preprocess(det_input);
    det.inference();
    det.postprocess();
    const std::vector<OCRBox> &boxes = det.ocr_det_out_;
    ROS_INFO("Detected   : %zu text region(s)", boxes.size());

    // ── 3. 逐框识别 ─────────────────────────────────────────────────────────
    std::vector<OCRItem> ocr_items;
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        cv::Mat crop = crop_text_region(det_input, boxes[i]);
        if (crop.empty()) continue;

        rec.preprocess(crop);
        rec.inference();
        rec.postprocess();

        if (!rec.result.empty() && !rec.result[0].text.empty())
        {
            OCRItem item;
            item.box = boxes[i];
            item.rec = rec.result[0];
            ocr_items.push_back(item);
            ROS_INFO("  [%zu] \"%s\"  score=%.3f",
                     i, rec.result[0].text.c_str(), rec.result[0].score);
        }
    }

    // ── 4. 算术解析 ─────────────────────────────────────────────────────────
    std::string all_text;
    for (const auto &item : ocr_items)
        all_text += item.rec.text + " ";

    ROS_INFO("All OCR    : \"%s\"", all_text.c_str());

    double calc_result = 0.0;
    if (!parse_simple_expr(all_text, calc_result, expr_str))
    {
        ROS_WARN("No arithmetic expression found.");
        return false;
    }

    int_result = static_cast<int>(std::round(calc_result));
    mod_result = ((int_result % 4) + 4) % 4;
    return true;
}


// ============================================================================
// 主函数 — 根据编译宏进入不同模式
// ============================================================================
int main(int argc, char **argv)
{
    ros::init(argc, argv, "ppocr_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // ── 1. 加载配置 ─────────────────────────────────────────────────────────
    std::string config_path;
    pnh.param<std::string>("config_path", config_path,
                           ros::package::getPath("dogvision_vision") + "/config/settings.json");

    Appconfig config;
    {
        detect_det_ppocr loader(nullptr);
        loader.load_config(config, config_path);
    }
    ROS_INFO("Config     : %s", config_path.c_str());

    // ── 2. 初始化 OCR 模型（两版共用）───────────────────────────────────────
    detect_det_ppocr det(&config);
    det.load_model(config.detect_config.ppocr_det_model_path,
                   config.detect_config.det_device);

    detect_rec_ppocr rec(&config);
    rec.load_model(config.detect_config.ppocr_rec_model_path,
                   config.detect_config.rec_device);
    rec.loda_dict(config.detect_config.rec_char_dict_path);

    const float default_wh_ratio =
        (config.detect_config.rec_img_h > 0)
            ? static_cast<float>(config.detect_config.rec_img_w) /
                  static_cast<float>(config.detect_config.rec_img_h)
            : 320.0f / 48.0f;
    rec.set_max_wh_ratio(default_wh_ratio);

    // ── 3. 初始化 Hikvision 摄像头 ──────────────────────────────────────────
    s_camera_params cam_params;
    cam_params.device_id = config.hikcamera_config.device_id;
    cam_params.width     = config.hikcamera_config.width;
    cam_params.height    = config.hikcamera_config.height;
    cam_params.offset_x  = config.hikcamera_config.offset_x;
    cam_params.offset_y  = config.hikcamera_config.offset_y;
    cam_params.exposure  = config.hikcamera_config.exposure;

    HikGrab hik(cam_params);
    hik.Hik_init();
    ROS_INFO("Camera     : device=%d  %dx%d",
             cam_params.device_id, cam_params.width, cam_params.height);

    // ── 4. 初始化鱼眼去畸变（使用相机分辨率）────────────────────────────────
    init_fisheye_undistort(cam_params.width, cam_params.height);

    // =========================================================================
    // 🔬 测试版：持续抓帧 → OCR → 去重 → YAML 追加
    // =========================================================================
#ifdef OCR_TEST_MODE
    ROS_INFO("MODE       : TEST (continuous OCR + YAML output)");

    // 准备 YAML 文件路径（自动创建目录）
    const std::string yaml_path =
        ros::package::getPath("dogvision_vision") + "/data/ocr_output/ocr_results.yaml";
    std::error_code ec;
    fs::create_directories(
        ros::package::getPath("dogvision_vision") + "/data/ocr_output", ec);
    if (ec)
        ROS_WARN("Cannot create YAML dir: %s", ec.message().c_str());

    int problem_id = 0;
    bool yaml_header_written = false;
    std::unordered_set<std::string> seen_exprs;

    ROS_INFO("YAML output: %s", yaml_path.c_str());
    ROS_INFO("Press Q or ESC to exit.");

    while (ros::ok())
    {
        // ── 2 秒间隔控制 ────────────────────────────────────────────────────
        static auto last_infer = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        // if (std::chrono::duration_cast<std::chrono::seconds>(now - last_infer).count() < 0.02)
        // {
        //     int k = cv::waitKey(10);
        //     if (k == 'q' || k == 'Q' || k == 27) break;
        //     ros::spinOnce();
        //     continue;
        // }

        // ── 抓帧 + 断线重连 ────────────────────────────────────────────────
        cv::Mat frame;
        if (!hik.get_one_frame(frame, 0))
        {
            if (!ensure_camera(hik))
            {
                ROS_ERROR("Cannot recover camera, exiting.");
                break;
            }
            continue;
        }

        // ── 鱼眼去畸变（在所有处理之前执行，保证后续推理和可视化都基于矫正图）─
        {
            cv::Mat undistorted = undistort_image(frame);
            if (!undistorted.empty())
                frame = undistorted;
            else
                ROS_WARN_THROTTLE(5, "Undistortion failed, using raw frame.");
        }

        // ── OCR + 算术解析 ─────────────────────────────────────────────────
        last_infer = now;
        std::string expr_str;
        int int_result = 0, mod_result = 0;
        cv::Rect2f math_roi;
        if (run_ocr_pipeline(frame, det, rec, expr_str, int_result, mod_result, &math_roi))
        {
            ROS_INFO("Expr: %s  =>  %d  %%4 = %d",
                     expr_str.c_str(), int_result, mod_result);

            // ── 在 ROI 左下角外侧实时绘制公式 + 答案 ──────────────────────
#ifdef OCR_SHOW_VISUAL
            // 构建显示文本："12 + 3 * 5 - 8 / 2 = 23  (mod4: 3)"
            std::ostringstream disp;
            disp << expr_str  << " = " << int_result << "  (mod4: " << mod_result << ")";
            std::string display_text = disp.str();

            // 绘制位置：ROI 左下角外侧（x=roi.x, y=roi.bottom + 偏移）
            int text_x = static_cast<int>(math_roi.x);
            int text_y = static_cast<int>(math_roi.y + math_roi.height) + 40;
            // 避免超出图像底部
            text_y = std::min(text_y, frame.rows - 10);

            double font_scale = 0.8;
            int thickness = 2;
            int baseline = 0;
            cv::Size ts = cv::getTextSize(display_text, cv::FONT_HERSHEY_DUPLEX,
                                          font_scale, thickness, &baseline);
            // 确保不超出右边界
            if (text_x + ts.width > frame.cols)
                text_x = frame.cols - ts.width - 10;

            // 黑色背景半透明效果（填充矩形）
            cv::Rect bg(text_x - 4, text_y - ts.height - 4,
                        ts.width + 8, ts.height + 8);
            bg &= cv::Rect(0, 0, frame.cols, frame.rows);
            cv::rectangle(frame, bg, cv::Scalar(0, 0, 0), -1);           // 黑色填充
            cv::rectangle(frame, bg, cv::Scalar(0, 200, 0), 2);          // 绿色边框

            // 白色文字
            cv::putText(frame, display_text,
                        cv::Point(text_x, text_y - 4),
                        cv::FONT_HERSHEY_DUPLEX, font_scale,
                        cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
#endif

            // 去重：相同表达式不重复写入
            if (seen_exprs.find(expr_str) == seen_exprs.end())
            {
                seen_exprs.insert(expr_str);
                problem_id++;

                // 追加写入 YAML
                std::ofstream ofs(yaml_path, std::ios::app);
                if (ofs.is_open())
                {
                    if (!yaml_header_written)
                    {
                        ofs << "ocr_results:" << std::endl;
                        yaml_header_written = true;
                    }
                    ofs << "  - id: "       << problem_id            << std::endl;
                    ofs << "    question: \"" << expr_str << "\""     << std::endl;
                    ofs << "    answer: "   << int_result            << std::endl;
                    ofs << "    mod4: "     << mod_result            << std::endl;
                    ofs.close();
                    ROS_INFO("YAML write : id=%d", problem_id);
                }
                else
                {
                    ROS_WARN("Cannot open YAML: %s", yaml_path.c_str());
                }
            }
            else
            {
                ROS_INFO("Duplicate, skipped.");
            }
        }

        // ── 实时可视化（始终显示已矫正的 frame，识别成功时含文字标注）────
#ifdef OCR_SHOW_VISUAL
        cv::imshow("Math OCR", frame);
#endif

        // 键盘退出检测
        int key = cv::waitKey(10);
        if (key == 'q' || key == 'Q' || key == 27)
        {
            ROS_INFO("Exit by user.");
            break;
        }

        ros::spinOnce();
    }

    hik.Hik_end();
#ifdef OCR_SHOW_VISUAL
    cv::destroyAllWindows();
#endif
    ROS_INFO("Test mode finished. Total unique problems: %d", problem_id);
    return 0;
#endif // OCR_TEST_MODE

    // =========================================================================
    // 🚀 正式版：等待 /ocr/trigger → 单次识别 → 发布 /ocr/result
    // =========================================================================
#ifdef OCR_PRODUCTION_MODE
    ROS_INFO("MODE       : PRODUCTION (trigger-based)");

    // 触发器
    ros::Subscriber trigger_sub = nh.subscribe<std_msgs::String>(
        "/ocr/trigger", 1, ocr_trigger_callback);

    // 结果发布（latched，新节点加入也能拿到最新一次结果）
    ros::Publisher result_pub = nh.advertise<std_msgs::String>(
        "/ocr/result", 1, true);

    ROS_INFO("Subscribed : /ocr/trigger");
    ROS_INFO("Publishing : /ocr/result (latched)");
    ROS_INFO("Waiting for trigger...");

    while (ros::ok())
    {
        // ── IDLE：等待触发 ──────────────────────────────────────────────────
        while (!g_ocr_triggered && ros::ok())
        {
            ros::spinOnce();
            cv::waitKey(10);
        }
        if (!ros::ok()) break;

        // ── 触发：抓帧 → OCR → 发布 ─────────────────────────────────────────
        cv::Mat frame;
        if (!hik.get_one_frame(frame, 0))
        {
            ROS_ERROR("Failed to grab frame, reconnecting...");
            if (ensure_camera(hik))
            {
                g_ocr_triggered = false;
                continue;
            }
            else
            {
                ROS_ERROR("Cannot recover camera, exiting.");
                break;
            }
        }

        std::string expr_str;
        int int_result = 0, mod_result = 0;

        if (run_ocr_pipeline(frame, det, rec, expr_str, int_result, mod_result))
        {
            ROS_INFO("Expr: %s  =>  %d  %%4 = %d",
                     expr_str.c_str(), int_result, mod_result);

            // 显示结果窗口（自动关闭，不阻塞）
#ifdef OCR_SHOW_VISUAL
            show_result_window(expr_str, mod_result);
#endif

            // 构建 JSON 并发布
            Json::Value result_json;
            result_json["expr"]   = expr_str;
            result_json["result"] = int_result;
            result_json["mod4"]   = mod_result;

            Json::FastWriter writer;
            std_msgs::String msg;
            msg.data = writer.write(result_json);

            result_pub.publish(msg);
            ROS_INFO("Published  : %s", msg.data.c_str());
        }
        else
        {
            // 未识别到表达式，仍然发布一个空结果标记
            std_msgs::String msg;
            msg.data = "{\"error\": \"no expression found\"}";
            result_pub.publish(msg);
            ROS_WARN("Published error result.");
        }

        // 重置触发标志，等待下一次触发
        g_ocr_triggered = false;
#ifdef OCR_SHOW_VISUAL
        cv::waitKey(500);  // 短暂显示结果
        cv::destroyAllWindows();
#endif
    }

    hik.Hik_end();
    ROS_INFO("Production mode finished.");
    return 0;
#endif // OCR_PRODUCTION_MODE

    // 未定义任何模式
    ROS_ERROR("No mode defined! Please #define OCR_TEST_MODE or OCR_PRODUCTION_MODE.");
    return -1;
}
