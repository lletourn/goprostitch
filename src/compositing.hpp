#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

float compute_warped_image_scale(const std::vector<cv::detail::CameraParams>& cameras);

class ImageCompositing {
  public:
    ImageCompositing(const std::vector<cv::detail::CameraParams>& cameras, const std::vector<cv::UMat>& masks_warped, const std::vector<cv::Size> images_size);
    ~ImageCompositing();

  public:
    void compose(const std::vector<cv::Mat>& images, cv::Mat& output_image);

  private:
    const std::vector<cv::detail::CameraParams> cameras_parameters_;
    std::vector<cv::Mat> blending_masks_;
    std::vector<cv::Point> corners_;
    std::vector<cv::Size> sizes_;
    cv::Ptr<cv::detail::RotationWarper> warper_;
};
