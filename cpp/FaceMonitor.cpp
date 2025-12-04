#include "FaceMonitor.hpp"
#include <iostream>
#include <numeric>

FaceMonitor::FaceMonitor() : initialized(false) {
    // Load Face Detector (Haar Cascade)
    if (!face_detector.load("haarcascade_frontalface_alt2.xml")) {
        std::cerr << "Error loading haarcascade_frontalface_alt2.xml. Make sure it is in the execution directory." << std::endl;
    }

    // Load Facemark
    facemark = cv::face::FacemarkLBF::create();
    try {
        facemark->loadModel("lbfmodel.yaml");
    } catch (cv::Exception& e) {
        std::cerr << "Error loading lbfmodel.yaml: " << e.what() << ". Make sure it is in the execution directory." << std::endl;
    }

    cap.open(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video capture" << std::endl;
    }

    dist_matrix = cv::Mat::zeros(4, 1, CV_64F);
}

FaceMonitor::~FaceMonitor() {
    cap.release();
}

void FaceMonitor::init_camera_matrix(int width, int height) {
    double focal_length = width;
    cam_matrix = (cv::Mat_<double>(3, 3) << 
        focal_length, 0, width / 2.0,
        0, focal_length, height / 2.0,
        0, 0, 1);
    initialized = true;
}

double FaceMonitor::get_head_pose(cv::Mat& frame) {
    cap >> frame;
    if (frame.empty()) return -999.0;

    cv::flip(frame, frame, 1);
    int h = frame.rows;
    int w = frame.cols;

    if (!initialized) {
        init_camera_matrix(w, h);
    }

    std::vector<cv::Rect> faces;
    face_detector.detectMultiScale(frame, faces);

    if (faces.empty()) {
        pitch_history.clear();
        return -999.0;
    }

    // Find largest face
    int largest_idx = 0;
    double max_area = 0;
    for (size_t i = 0; i < faces.size(); ++i) {
        double area = faces[i].area();
        if (area > max_area) {
            max_area = area;
            largest_idx = i;
        }
    }

    std::vector<std::vector<cv::Point2f>> landmarks;
    if (facemark->fit(frame, faces, landmarks)) {
        if (largest_idx < landmarks.size()) {
            const auto& shape = landmarks[largest_idx];
            
            // 68 points mapping
            std::vector<cv::Point2d> image_points;
            image_points.push_back(shape[30]); // Nose Tip
            image_points.push_back(shape[8]);  // Chin
            image_points.push_back(shape[36]); // Left Eye Left Corner
            image_points.push_back(shape[45]); // Right Eye Right Corner
            image_points.push_back(shape[48]); // Mouth Left
            image_points.push_back(shape[54]); // Mouth Right

            // 3D model points
            std::vector<cv::Point3d> model_points;
            model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));          // Nose Tip
            model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));     // Chin
            model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));  // Left Eye Left Corner
            model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));   // Right Eye Right Corner
            model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f)); // Mouth Left
            model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));  // Mouth Right

            cv::Mat rot_vec, trans_vec;
            cv::solvePnP(model_points, image_points, cam_matrix, dist_matrix, rot_vec, trans_vec);

            cv::Mat rmat;
            cv::Rodrigues(rot_vec, rmat);
            
            cv::Mat mtxR, mtxQ;
            cv::Vec3d angles = cv::RQDecomp3x3(rmat, mtxR, mtxQ);
            double pitch = angles[0]; 

            // Smoothing
            pitch_history.push_back(pitch);
            if (pitch_history.size() > 5) pitch_history.pop_front();
            
            double avg_pitch = 0;
            for (double p : pitch_history) avg_pitch += p;
            avg_pitch /= pitch_history.size();

            // Visual Guide
            cv::Point2d p1 = shape[30];
            cv::Point2d p2;
            p2.x = p1.x + angles[1] * 10; 
            p2.y = p1.y - angles[0] * 10; 
            cv::line(frame, p1, p2, cv::Scalar(255, 0, 0), 3);
            
            // Draw rectangle around face
            cv::rectangle(frame, faces[largest_idx], cv::Scalar(0, 255, 0), 2);

            return avg_pitch;
        }
    }
    
    return -999.0;
}
