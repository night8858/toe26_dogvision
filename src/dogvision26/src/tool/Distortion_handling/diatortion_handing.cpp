#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    // 参数：
    // argv[1] image_glob_pattern, 默认 "src/dogvision26/src/tool/Distortion_handling/img/*.jpg"
    // argv[2] board_cols, argv[3] board_rows, 默认 8 x 12
    // argv[4] square_size(mm), 默认 20.0
    // argv[5] output_path, 默认 "fisheye_params.yaml"
    const string image_glob =
        (argc > 1) ? argv[1] : "src/dogvision26/src/tool/Distortion_handling/img/*.jpg";
    const int board_cols = (argc > 2) ? std::stoi(argv[2]) : 12;
    const int board_rows = (argc > 3) ? std::stoi(argv[3]) : 8;
    const float square_size = (argc > 4) ? std::stof(argv[4]) : 20.0f;
    const string output_path = (argc > 5) ? argv[5] : "fisheye_params.yaml";
    const bool force_no_gui = (argc > 6) ? (std::stoi(argv[6]) != 0) : false;

    // 运行环境中存在 OpenCV 运行时 ABI 混用风险（highgui/core 版本不一致）时，imshow 易崩溃。
    // 因此默认关闭 GUI，仅在显式设置 DIATORTION_ENABLE_GUI=1 且未 force_no_gui 时才开启。
    bool gui_enabled = false;
    const char* enable_gui_env = std::getenv("DIATORTION_ENABLE_GUI");
    if (enable_gui_env != nullptr && std::string(enable_gui_env) == "1") {
        gui_enabled = true;
    }
    if (force_no_gui) {
        gui_enabled = false;
    }
#ifndef _WIN32
    const char* display_env = std::getenv("DISPLAY");
    if (display_env == nullptr || std::string(display_env).empty()) {
        gui_enabled = false;
    }
#endif

    if (board_cols <= 1 || board_rows <= 1 || square_size <= 0.0f) {
        cerr << "参数非法: board_cols/board_rows 需 >1，square_size 需 >0" << endl;
        return -1;
    }

    const Size board_size(board_cols, board_rows);
    vector<String> image_paths;
    cv::glob(image_glob, image_paths, false);

    if (image_paths.empty()) {
        cerr << "未找到标定图片，请检查路径模式: " << image_glob << endl;
        return -1;
    }

    // 1) 准备棋盘格世界坐标
    vector<Point3f> obj;
    obj.reserve(static_cast<size_t>(board_size.width * board_size.height));
    for (int row = 0; row < board_size.height; ++row) {
        for (int col = 0; col < board_size.width; ++col) {
            obj.emplace_back(col * square_size, row * square_size, 0.0f);
        }
    }

    vector<vector<Point3f>> object_points;
    vector<vector<Point2f>> image_points;
    Size image_size;
    bool image_size_initialized = false;
    Mat preview_img;

    cout << "开始检测角点，图片数量: " << image_paths.size() << endl;
    cout << "GUI显示: " << (gui_enabled ? "开启" : "关闭(默认无头模式)") << endl;
    if (!gui_enabled) {
        cout << "提示: 如需窗口预览，请先导出 DIATORTION_ENABLE_GUI=1" << endl;
    }

    // 2) 逐图检测角点
    for (size_t i = 0; i < image_paths.size(); ++i) {
        Mat img = imread(image_paths[i]);
        if (img.empty()) {
            cout << "警告: 无法读取图片 " << image_paths[i] << endl;
            continue;
        }

        if (!image_size_initialized) {
            image_size = img.size();
            image_size_initialized = true;
        } else if (img.size() != image_size) {
            cout << "警告: 跳过分辨率不一致图片 " << image_paths[i]
                 << "，size=" << img.cols << "x" << img.rows
                 << "，期望=" << image_size.width << "x" << image_size.height << endl;
            continue;
        }

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        vector<Point2f> corners;
        bool found = findChessboardCorners(
            gray,
            board_size,
            corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);

        // FAST_CHECK 失败时再做一次完整检测，提升成功率
        if (!found) {
            found = findChessboardCorners(
                gray,
                board_size,
                corners,
                CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        }

        if (!found) {
            cout << "未在 " << image_paths[i] << " 中找到角点。" << endl;
            continue;
        }

        cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1),
                     TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.1));

        object_points.push_back(obj);
        image_points.push_back(corners);
        if (preview_img.empty()) preview_img = img.clone();

        drawChessboardCorners(img, board_size, corners, true);

            if (gui_enabled) {
            try {
                imshow("Corner Detection", img);
                waitKey(80);
            } catch (const cv::Exception& e) {
                cerr << "OpenCV GUI 异常，切换到无头模式: " << e.what() << endl;
                gui_enabled = false;
                destroyAllWindows();
            }
        }
    }
    if (gui_enabled) {
        destroyAllWindows();
    }

    if (!image_size_initialized || object_points.empty()) {
        cerr << "未检测到有效角点，请检查图片质量、棋盘格规格或路径。" << endl;
        return -1;
    }

    if (object_points.size() < 6) {
        cerr << "有效样本过少(" << object_points.size()
             << ")，建议至少 10-15 张不同姿态图片。" << endl;
        return -1;
    }

    // 3) 鱼眼标定
    cout << "开始鱼眼标定..." << endl;
    Mat K = Mat::eye(3, 3, CV_64F);
    Mat D = Mat::zeros(4, 1, CV_64F);
    vector<Mat> rvecs, tvecs;

    int flags = 0;
    flags |= fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    // 不使用 CALIB_CHECK_COND：条件数差时该 flag 会抛 cv::Exception 导致 abort
    flags |= fisheye::CALIB_FIX_SKEW;

    double rms = 0.0;
    try {
        rms = fisheye::calibrate(
            object_points,
            image_points,
            image_size,
            K,
            D,
            rvecs,
            tvecs,
            flags,
            TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 200, 1e-6));
    } catch (const cv::Exception& e) {
        cerr << "fisheye::calibrate 失败: " << e.what() << endl;
        cerr << "建议: 检查标定图片质量；或增加图片数量/角度多样性。" << endl;
        return -1;
    }

    cout << "标定完成" << endl;
    cout << "RMS: " << rms << endl;
    cout << "K =\n" << K << endl;
    cout << "D =\n" << D << endl;

    // 4) 保存参数
    FileStorage fs(output_path, FileStorage::WRITE);
    if (!fs.isOpened()) {
        cerr << "无法写入标定文件: " << output_path << endl;
        return -1;
    }
    fs << "image_width" << image_size.width;
    fs << "image_height" << image_size.height;
    fs << "board_width" << board_size.width;
    fs << "board_height" << board_size.height;
    fs << "square_size_mm" << square_size;
    fs << "K" << K;
    fs << "D" << D;
    fs << "rms" << rms;
    fs.release();
    cout << "标定参数已保存: " << output_path << endl;

    // 5) 去畸变预览
    if (preview_img.empty()) {
        preview_img = imread(image_paths.front());
    }
    if (preview_img.empty()) {
        cerr << "预览图像为空，跳过去畸变演示。" << endl;
        return 0;
    }

    Mat newK;
    fisheye::estimateNewCameraMatrixForUndistortRectify(
        K, D, preview_img.size(), Matx33d::eye(), newK, 1.0);

    Mat map1, map2;
    fisheye::initUndistortRectifyMap(
        K, D, Matx33d::eye(), newK, preview_img.size(), CV_16SC2, map1, map2);

    Mat undistorted_img;
    remap(preview_img, undistorted_img, map1, map2, INTER_LINEAR);

    if (gui_enabled) {
        try {
            imshow("Original Distorted", preview_img);
            imshow("Undistorted Result", undistorted_img);
            waitKey(0);
            destroyAllWindows();
        } catch (const cv::Exception& e) {
            cerr << "GUI 显示失败: " << e.what() << endl;
            const string undistort_output = "undistorted_preview.jpg";
            imwrite(undistort_output, undistorted_img);
            cout << "已保存去畸变预览图到 " << undistort_output << endl;
        }
    } else {
        const string undistort_output = "undistorted_preview.jpg";
        imwrite(undistort_output, undistorted_img);
        cout << "无头模式：已保存去畸变预览图到 " << undistort_output << endl;
    }
    return 0;
}