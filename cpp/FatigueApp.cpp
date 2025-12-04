#include "FatigueApp.hpp"
#include <iostream>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>

FatigueApp::FatigueApp() {
    smooth_fatigue = 0.0;
    smoothing_factor = 0.2;
    last_alert_time = 0;
    alert_cooldown = 3.0;
    time_counter = 0;
    
    start_time = (double)cv::getTickCount() / cv::getTickFrequency();
    last_update = start_time;
    update_interval = 1.0;
    
    input_mon.start();
}

void FatigueApp::play_warning_sound() {
    // Windows Beep (Frequency, Duration)
    // Run in a detached thread to avoid blocking
    std::thread([](){
        Beep(1000, 500);
    }).detach();
}

double FatigueApp::calculate_current_fatigue(int keys, int clicks, double pitch) {
    double activity_score = std::min((double)(keys * 5 + clicks * 10), 50.0);
    
    double posture_fatigue = 0;
    // Pitch thresholds in degrees
    if (pitch < -20.0) posture_fatigue = 50;
    else if (pitch < -5.0) posture_fatigue = 10;
    
    double base_fatigue = 50 - activity_score;
    return std::max(0.0, std::min(100.0, base_fatigue + posture_fatigue));
}

double FatigueApp::predict_fatigue() {
    if (fatigue_history.size() < 5) {
        return fatigue_history.empty() ? 0.0 : fatigue_history.back();
    }
    
    // Simple Linear Regression
    size_t n = fatigue_history.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    
    for (size_t i = 0; i < n; ++i) {
        double x = (double)i;
        double y = fatigue_history[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    
    double m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    double c = (sum_y - m * sum_x) / n;
    
    // Predict +60 seconds (indices)
    double x_pred = (double)(n - 1) + 60.0;
    double y_pred = m * x_pred + c;
    
    return std::max(0.0, std::min(100.0, y_pred));
}

void FatigueApp::draw_graph(cv::Mat& img) {
    int w = img.cols;
    int h = img.rows;
    
    // Background
    img.setTo(cv::Scalar(20, 20, 20));
    
    // Grid
    for (int i = 0; i < w; i += 50) cv::line(img, cv::Point(i, 0), cv::Point(i, h), cv::Scalar(50, 50, 50));
    for (int i = 0; i < h; i += 50) cv::line(img, cv::Point(0, i), cv::Point(w, i), cv::Scalar(50, 50, 50));
    
    if (fatigue_history.empty()) return;
    
    int center_x = w / 2;
    double scale_y = (double)h / 100.0;
    double scale_x = (double)(w / 2) / 60.0; // 60 seconds half width
    
    std::vector<cv::Point> points;
    for (size_t i = 0; i < fatigue_history.size(); ++i) {
        double t = timestamps[i] - time_counter;
        if (t < -60) continue;
        
        int x = center_x + (int)(t * scale_x);
        int y = h - (int)(fatigue_history[i] * scale_y);
        points.push_back(cv::Point(x, y));
    }
    
    if (points.size() > 1) {
        const cv::Point* pts = &points[0];
        int npts = (int)points.size();
        cv::polylines(img, &pts, &npts, 1, false, cv::Scalar(255, 255, 0), 2);
    }
    
    // Prediction
    double pred_val = predict_fatigue();
    int pred_x = center_x + (int)(60 * scale_x);
    int pred_y = h - (int)(pred_val * scale_y);
    int curr_y = h - (int)(smooth_fatigue * scale_y);
    
    cv::line(img, cv::Point(center_x, curr_y), cv::Point(pred_x, pred_y), cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
    
    // Text
    std::stringstream ss;
    ss << "Current: " << std::fixed << std::setprecision(1) << smooth_fatigue << "%";
    cv::putText(img, ss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    
    std::stringstream ss2;
    ss2 << "Pred (+60s): " << std::fixed << std::setprecision(1) << pred_val << "%";
    cv::putText(img, ss2.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);
}

void FatigueApp::run() {
    cv::Mat graph_img(400, 800, CV_8UC3);
    cv::Mat frame;
    
    std::cout << "Starting Fatigue Monitor (C++)..." << std::endl;
    std::cout << "Press 'q' to exit." << std::endl;

    while (true) {
        double pitch = face_mon.get_head_pose(frame);
        
        double current_time = (double)cv::getTickCount() / cv::getTickFrequency();
        if (current_time - last_update > update_interval) {
            last_update = current_time;
            
            int keys = 0, clicks = 0;
            input_mon.get_and_reset_counts(keys, clicks);
            
            if (pitch == -999.0) {
                // Face lost
                pitch = -20.0; // Penalty
            }
            
            double target_fatigue = calculate_current_fatigue(keys, clicks, pitch);
            smooth_fatigue = (smooth_fatigue * (1.0 - smoothing_factor)) + (target_fatigue * smoothing_factor);
            
            time_counter++;
            timestamps.push_back(time_counter);
            fatigue_history.push_back(smooth_fatigue);
            
            if (timestamps.size() > 200) {
                timestamps.pop_front();
                fatigue_history.pop_front();
            }
            
            if (smooth_fatigue >= 90.0) {
                if (current_time - last_alert_time > alert_cooldown) {
                    play_warning_sound();
                    last_alert_time = current_time;
                    std::cout << "WARNING: Fatigue Critical!" << std::endl;
                }
            }
        }
        
        if (!frame.empty()) {
            cv::imshow("Face Monitor", frame);
        }
        
        draw_graph(graph_img);
        cv::imshow("Fatigue Graph", graph_img);
        
        if (cv::waitKey(10) == 'q') {
            break;
        }
    }
    
    input_mon.stop();
}
