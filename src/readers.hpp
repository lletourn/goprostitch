#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct PointPair {
    bool locked = false;
    cv::Point points[2];

    PointPair() {
        points[0] = cv::Point(-1, -1);
        points[1] = cv::Point(-1, -1);
    }

    PointPair(bool l, int x1, int y1, int x2, int y2) {
        locked = l;
        points[0] = cv::Point(x1, y1);
        points[1] = cv::Point(x2, y2);
    }
};

void readSeamData(const std::string& cameras_filename, std::vector<cv::detail::CameraParams>& cameras, std::vector<cv::UMat>& masks_warped);
void readCalibration(const std::string& calibration_filename, cv::Mat& K, cv::Mat& distortion_coefficients, cv::Size& calibration_image_size);
std::vector<PointPair> readPointPairs(const std::string& pointpairs_filename);
