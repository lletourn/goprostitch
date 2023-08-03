#pragma once
#include <string>
#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/stitching/detail/camera.hpp>

void readSeamData(const std::string& cameras_filename, std::vector<cv::detail::CameraParams>& cameras, std::vector<cv::UMat>& masks_warped);
