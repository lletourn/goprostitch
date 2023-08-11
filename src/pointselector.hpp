#pragma once

#include <opencv2/opencv.hpp>
#include "readers.hpp"

struct ClickData {
    uint32_t point_idx;
    std::chrono::steady_clock::time_point change_time;
    std::vector<PointPair>* point_pairs_;
};

class PointSelector {
  public:
    PointSelector();
    PointSelector(const std::vector<PointPair>& point_pair);
    ~PointSelector();

    cv::Mat homography();

    std::vector<PointPair> select_points(const cv::Mat& left, const cv::Mat& right);
  private:
    static void mouseHandler(int event, int x, int y, int flags, void* userdata);

    bool processKeyboardEvent(bool& force_change, const cv::Mat& left, const cv::Mat& right);

  private:
    std::vector<PointPair> point_pairs_;
};
