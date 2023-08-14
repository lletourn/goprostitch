#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

void compositing(const std::vector<cv::Mat>& images, const std::vector<cv::detail::CameraParams>& cameras, const std::vector<cv::UMat>& masks_warped, cv::Mat& output_image);

float compute_warped_image_scale(const std::vector<cv::detail::CameraParams>& cameras);

