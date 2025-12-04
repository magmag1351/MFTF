#pragma once
#include "InputMonitor.hpp"
#include "FaceMonitor.hpp"
#include <deque>
#include <vector>
#include <chrono>

class FatigueApp {
public:
    FatigueApp();
    void run();

private:
    InputMonitor input_mon;
    FaceMonitor face_mon;
    
    std::deque<double> fatigue_history;
    std::deque<double> timestamps; // Relative time in seconds
    int time_counter;

    double smooth_fatigue;
    double smoothing_factor;
    
    double last_alert_time;
    double alert_cooldown;
    
    double start_time;
    double last_update;
    double update_interval;
    
    double calculate_current_fatigue(int keys, int clicks, double pitch);
    void play_warning_sound();
    void draw_graph(cv::Mat& img);
    double predict_fatigue(); // Simple linear regression
};
