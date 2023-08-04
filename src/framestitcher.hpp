#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/stitching/detail/camera.hpp>

#include "datatypes.hpp"
#include "threadsafequeue.hpp"

class FrameStitcher {
 public:
    static void BuildReferenceHistogram(uint8_t* reference_data, uint32_t width, uint32_t height, std::vector<std::vector<uint32_t>>& bgr_value_idxs, std::vector<std::vector<double>>& bgr_cumsum);
    static void MatchHistograms(cv::Mat& image, const std::vector<std::vector<double>>& reference_bgr_cumsum, const std::vector<std::vector<uint32_t>>& reference_bgr_value_idxs);
    static void interp(const std::vector<double>& x, const std::vector<double>& xp, const std::vector<uint32_t>& yp, std::vector<uint8_t>& y);

    FrameStitcher(uint32_t crop_offset_x, uint32_t crop_offset_y, uint32_t crop_width, uint32_t crop_height, ThreadSafeQueue<LeftRightPacket>& stitcher_queue, ThreadSafeQueue<PanoramicPacket>& output_queue, std::vector<cv::detail::CameraParams> camera_params, const std::vector<cv::UMat>& image_masks, const std::vector<std::vector<uint32_t>>& reference_bgr_value_idxs, const std::vector<std::vector<double>>& reference_bgr_cumsum);
    ~FrameStitcher();


    void start();
    void stop();
    void run();
    bool is_done();

    void close();
 private:
    void stitch(const std::vector<cv::Mat>& images, cv::Mat& panoramic_image);

 private:
    bool match_histogram_;
    uint32_t crop_offset_x_;
    uint32_t crop_offset_y_;
    uint32_t crop_width_;
    uint32_t crop_height_;
    ThreadSafeQueue<LeftRightPacket>& stitcher_queue_;
    ThreadSafeQueue<PanoramicPacket>& output_queue_;
    std::vector<cv::detail::CameraParams> camera_params_;
    const std::vector<cv::UMat>& image_masks_;
    const std::vector<std::vector<uint32_t>>& reference_bgr_value_idxs_;
    const std::vector<std::vector<double>>& reference_bgr_cumsum_;
    std::atomic<bool> running_;
    std::atomic<bool> done_;

    std::thread thread_;

    float warped_image_scale_;
};
