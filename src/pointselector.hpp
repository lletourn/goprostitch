#pragma once

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

    std::vector<PointPair> select_points(const cv::Mat& left, const cv::Mat& right);
  private:
    static void mouseHandler(int event, int x, int y, int flags, void* userdata);

    bool processKeyboardEvent(bool& force_change, const cv::Mat& left, const cv::Mat& right);

  private:
    std::vector<PointPair> point_pairs_;
};
