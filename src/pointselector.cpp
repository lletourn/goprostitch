#include "pointselector.hpp"

#include <spdlog/spdlog.h>

using namespace std;
using namespace cv;

PointSelector::PointSelector()
: resize_ratio_num_(3), resize_ratio_den_(8) {
    point_pairs_.push_back(PointPair());
}

PointSelector::PointSelector(const vector<PointPair>& point_pair)
: PointSelector() {
    for(PointPair pp : point_pair) {
        point_pairs_.push_back(pp);
    }
    point_pairs_.push_back(PointPair());
}

PointSelector::~PointSelector() {
}

void PointSelector::mouseHandler(int e, int x, int y, int flags, void *ptr) {
    ClickData* click_data = (ClickData*)ptr;

    uint32_t idx = click_data->point_idx;
    switch (e) {
        case EVENT_LBUTTONDOWN:
            spdlog::info("Point set {}", idx);
            click_data->change_time = chrono::steady_clock::now();
            click_data->point_pairs_->back().points[idx] = Point(x,y);
            break;

        //case EVENT_MOUSEMOVE:
        //    if ((flags & EVENT_FLAG_SHIFTKEY) && (flags & EVENT_FLAG_LBUTTON)) {
        //        ((ClickData*) ptr)->refresh_wnd = true;
        //        updateActiveClick((ClickData*) ptr, x, y);
        //    } else if (flags & EVENT_FLAG_LBUTTON) {
        //        ((ClickData*) ptr)->refresh_wnd = true;
        //        updateClickData((ClickData*) ptr, x, y);
        //    }
        //    break;

        default:
            break;
    }
}


vector<PointPair> PointSelector::select_points(const Mat& left, const Mat& right) {
    Mat rz_left_orig;
    resize(left, rz_left_orig, Size(left.rows, left.cols*resize_ratio_num_/resize_ratio_den_));
    Mat rz_right_orig;
    resize(right, rz_right_orig, Size(right.rows, right.cols*resize_ratio_num_/resize_ratio_den_));
    imshow("Left", rz_left_orig);
    imshow("Right", rz_right_orig);

    Mat rz_left(rz_left_orig.clone());
    Mat rz_right(rz_right_orig.clone());
    Mat warped(rz_right.rows, rz_right.cols+rz_left.cols, rz_right.type());

    chrono::steady_clock::time_point last_change = chrono::steady_clock::now();

    ClickData click_data_left = {0, last_change, &point_pairs_};
    ClickData click_data_right = {1, last_change, &point_pairs_};

    setMouseCallback("Left", mouseHandler, &click_data_left);
    setMouseCallback("Right", mouseHandler, &click_data_right);

    const int radius = 25;
    bool stop = false;
    bool changed = false;
    while(!stop) {
        if(click_data_left.change_time > last_change) {
            last_change = click_data_left.change_time;
            changed = true;
        }
        if(click_data_right.change_time > last_change) {
            last_change = click_data_right.change_time;
            changed = true;
        }

        if(changed) {
            spdlog::info("Points changed updating");

            vector<Point2f> pts_left;
            vector<Point2f> pts_right;
            for(PointPair& pp : point_pairs_) {
                if(pp.points[0].x != -1) {
                    Scalar color = Scalar(0,0,255);
                    if(pp.locked) {
                        color = Scalar(255,0,0);
                    }
                    circle(rz_left, pp.points[0], radius, color);
                }
                if(pp.points[1].x != -1) {
                    Scalar color = Scalar(0,0,255);
                    if(pp.locked) {
                        color = Scalar(255,0,0);
                    }
                    circle(rz_right, pp.points[1], radius, color);
                }
            }

            imshow("Left", rz_left);
            imshow("Right", rz_right);
        }
        changed = false;
        stop = processKeyboardEvent(changed, rz_left, rz_right);
    }

    return point_pairs_;
}

bool PointSelector::processKeyboardEvent(bool& force_change, const Mat&  left, const Mat& right) {
    
    // Can't be 0 we need to check for mouse clicks...
    int key = waitKey(30);

    // modifier keys are flags starting at 0x10000
    bool shiftPressed = key & 1 << 16;
    bool ctrlPressed = key & 1 << 18;
    key = key & 0xffff;
    // Arrows == values > 60k...so for keybord, use <128 from the ASCII table
    if (key != -1 && key != 65535) {
        if (key == 13 || key == 27 || key == 10) { // CR, ESC, or LF
            return true;
        }
        else if (key == 'a' || key == 'A') {
            PointPair& pp = point_pairs_.back();
            if(pp.points[0].x != -1 && pp.points[1].x != -1) {
                pp.locked = true;
                point_pairs_.push_back(PointPair());
            }
        }
        else if (key == 'r' || key == 'R') {
            force_change = true;
        }
        else if (key == 'h' || key == 'H') {
            force_change = true;
            vector<Point2f> pts_left;
            vector<Point2f> pts_right;
            for(PointPair& pp : point_pairs_) {
                if(pp.points[0].x != -1 && pp.points[1].x != -1) {
                    pts_left.push_back(Point2f(pp.points[0].x, pp.points[0].y));
                    pts_right.push_back(Point2f(pp.points[1].x, pp.points[1].y));
                    cout << pp.points[0].x << ',' << pp.points[0].y << ' ' << pp.points[1].x << ',' << pp.points[1].y << endl;
                }
            }

            if(pts_left.size() >= 4) {
                Mat warped(Size(left.cols*2, left.rows));
                spdlog::info("More than 4 points, hom time");
                Mat homography = findHomography(pts_right, pts_left, RANSAC, 3, noArray(), 5000);
                // Mat homography = findHomography(pts_right, pts_left, LMEDS);
                warpPerspective(right, warped, homography, warped.size());
                Mat insetImage(warped, cv::Rect(0, 0, left.cols, left.rows));
                left.copyTo(insetImage);
                imshow("Warp", warped);
            }
        }
    }
    return false;
}
