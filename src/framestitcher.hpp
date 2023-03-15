#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/stitching/detail/camera.hpp>

#include "threadsafequeue.hpp"

struct LeftRightPacket {
    uint64_t left_data_size;
    std::unique_ptr<uint8_t> left_data;
    uint64_t right_data_size;
    std::unique_ptr<uint8_t> right_data;
    uint32_t width;
    uint32_t height;
    uint32_t pts;
    double pts_time;
    uint32_t idx;
};

struct PanoramicPacket {
    uint32_t width;
    uint32_t height;
    uint32_t pts;
    double pts_time;
    uint32_t idx;
    uint64_t data_size;
    std::unique_ptr<uint8_t> data;
};

class FrameStitcher {
 public:
    static void BuildReferenceHistogram(const cv::Mat& reference_image, std::vector<std::vector<uint32_t>>& bgr_value_idxs, std::vector<std::vector<double>>& bgr_cumsum);

    FrameStitcher(ThreadSafeQueue<LeftRightPacket>& stitcher_queue, ThreadSafeQueue<PanoramicPacket>& output_queue, std::vector<cv::detail::CameraParams> camera_params, const std::vector<std::vector<uint32_t>>& reference_bgr_value_idxs, const std::vector<std::vector<double>>& reference_bgr_cumsum, uint32_t max_queue_size);
    ~FrameStitcher();

    void MatchHistograms(cv::Mat& image);

    void start();
    void stop();
    void run();
    bool is_done();

    ThreadSafeQueue<PanoramicPacket>& getOutPanoramicQueue();

    void close();
 private:
    void interp(const std::vector<double>& x, const std::vector<double>& xp, const std::vector<uint32_t>& yp, std::vector<uint8_t>& y);

 private:
    ThreadSafeQueue<LeftRightPacket>& stitcher_queue_;
    ThreadSafeQueue<PanoramicPacket>& output_queue_;
    std::vector<cv::detail::CameraParams> camera_params_;
    const std::vector<std::vector<uint32_t>>& reference_bgr_value_idxs_;
    const std::vector<std::vector<double>>& reference_bgr_cumsum_;
    ThreadSafeQueue<PanoramicPacket> output_video_packets_;
    std::atomic<bool> running_;
    std::atomic<bool> done_;

    std::thread thread_;

};
