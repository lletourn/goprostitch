#pragma once
#include <mutex>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

float compute_warped_image_scale(const std::vector<cv::detail::CameraParams>& cameras);

class ImageCompositing {
  public:
    ImageCompositing(bool do_blending, bool use_gpu_, const std::vector<cv::detail::CameraParams>& cameras, const std::vector<cv::UMat>& masks_warped, const std::vector<cv::Size> images_size);
    ~ImageCompositing();

  public:
    void compose(const std::vector<cv::Mat>& images, cv::Mat& output_image, int32_t frame_idx=0);
    void buildWarpMaps(std::vector<cv::Mat>& warp_maps_x, std::vector<cv::Mat>& warp_maps_y);
    void composePreWarped(const std::vector<cv::Mat>& warped_images, cv::Mat& output_image, int32_t frame_idx=0);

  private:
    bool do_blending_;
    bool use_gpu_;
    const std::vector<cv::detail::CameraParams> cameras_parameters_;
    std::vector<cv::Mat> blending_masks_;
    std::vector<cv::Point> corners_;
    std::vector<cv::Size> sizes_;
    cv::Ptr<cv::detail::RotationWarper> warper_;
    cv::Point top_left_;
    cv::Size panoramic_image_size_;
    std::vector<cv::Mat> K_CV_32Fs_;
    std::vector<cv::Size> input_image_sizes_;

    inline static std::mutex single_warp_mutex_;
};
