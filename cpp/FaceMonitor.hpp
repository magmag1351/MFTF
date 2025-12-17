#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <deque>
#include <vector>

class FaceMonitor {
public:
    FaceMonitor();
    ~FaceMonitor();

    // Returns pitch. If face not found, returns -999.0 (or a status code)
    // frame is an output parameter to draw on.
    double get_head_pose(cv::Mat& frame);

private:
    cv::VideoCapture cap;
    cv::Ptr<cv::face::Facemark> facemark;
    cv::CascadeClassifier face_detector;
    
    cv::Mat cam_matrix;
    cv::Mat dist_matrix;
    std::deque<double> pitch_history;
    
    bool initialized;
    void init_camera_matrix(int width, int height);
};
