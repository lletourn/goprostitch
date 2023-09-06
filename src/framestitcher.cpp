#include "framestitcher.hpp"

#include <stdexcept>
#include <iostream>
#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include "opencv2/stitching/detail/blenders.hpp"
#include <csignal>

#include "compositing.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

void FrameStitcher::BuildReferenceHistogram(uint8_t* reference_data, uint32_t width, uint32_t height, vector<vector<uint32_t>>& bgr_value_idxs, vector<vector<double>>& bgr_cumsum) {
    vector<Mat> bgr_planes;
    Mat ref_yuv(height * 3/2, width, CV_8UC1, reference_data);
    Mat reference_image;
    cvtColor(ref_yuv, reference_image, COLOR_YUV2BGR_I420);

    split(reference_image, bgr_planes);

    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };

    vector<Mat> bgr_hists(3);
    bgr_value_idxs.clear();
    bgr_cumsum.clear();
    for(uint32_t i=0; i < bgr_planes.size(); ++i) {
        bgr_value_idxs.push_back(vector<uint32_t>());
        bgr_cumsum.push_back(vector<double>());

        calcHist(&bgr_planes[i], 1, 0, Mat(), bgr_hists[i], 1, &histSize, histRange, true, false);
        assert(bgr_hists[i].type() == CV_32F);

        uint32_t cumulative_sum = 0;
        for(uint32_t j=0; j < histSize; ++j) {
            uint32_t value = (uint32_t)bgr_hists[i].at<float>(j);
            if(value > 0) {
                bgr_value_idxs[i].push_back(j);
                cumulative_sum += value;
                double quantile = ((double)cumulative_sum) / ((double)bgr_planes[i].total());
                bgr_cumsum[i].push_back(quantile);
            }
        }
    }
}

void FrameStitcher::interp(const vector<double>& x, const vector<double>& xp, const vector<uint32_t>& yp, vector<uint8_t>& y) {
    y.resize(x.size());
	
	int ip = 0;
	int ip_next = 1;
	int i = 0;

	while(i < x.size()){
        // compute slope between the 2 cumsum points where ip == x values(pixel value) quantiles == y values
		double m = (yp[ip_next]-yp[ip])/(xp[ip_next]-xp[ip]);
		double q = yp[ip] - m * xp[ip];
		while(x[i]<xp[ip_next]){
			if(x[i]>=xp[ip])
				y[i] = m*x[i]+q;
			i +=1;
			if (i >= x.size())
                break;
		}

		ip +=1;
		ip_next +=1;
		if(ip_next == xp.size()){
			while(i < x.size()){
				y[i] = yp.back();
				i++;
			}
			break;
		}
	}
}


void FrameStitcher::MatchHistograms(Mat& image, const std::vector<std::vector<double>>& reference_bgr_cumsum, const std::vector<std::vector<uint32_t>>& reference_bgr_value_idxs) {
    vector<Mat> bgr_planes;
    split(image, bgr_planes);

    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };

    
    vector<Mat> bgr_hists(3);
    vector<uint8_t> interp_a_values;
    for(uint32_t i=0; i < bgr_planes.size(); ++i) {
        calcHist(&bgr_planes[i], 1, 0, Mat(), bgr_hists[i], 1, &histSize, histRange, true, false);

        vector<uint32_t> bgr_value_idxs;
        vector<double> bgr_cumsum;
        uint32_t cumulative_sum = 0;
        for(uint32_t j=0; j < histSize; ++j) {
            uint32_t value = (uint32_t)bgr_hists[i].at<float>(j);
            bgr_value_idxs.push_back(j);
            cumulative_sum += value;
            double quantile = ((double)cumulative_sum) / ((double)bgr_planes[i].total());
            bgr_cumsum.push_back(quantile);
        }

        // Interpolate cumulative sum curve from image to reference curve to pixel value
        // https://en.wikipedia.org/wiki/Histogram_matching
        // https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_histogram_matching.html
        interp(bgr_cumsum, reference_bgr_cumsum[i], reference_bgr_value_idxs[i], interp_a_values);
        for(uint32_t y=0; y < image.rows; ++y) {
            for(uint32_t x=0; x < image.cols; ++x) {
                uint8_t pixel = bgr_planes[i].at<uint8_t>(y,x);
                bgr_planes[i].at<uint8_t>(y,x) = interp_a_values[pixel];
            }
        }
    }
    merge(bgr_planes, image);
}


FrameStitcher::FrameStitcher(
    uint32_t crop_offset_x,
    uint32_t crop_offset_y,
    uint32_t crop_width,
    uint32_t crop_height,
    ThreadSafeQueue<LeftRightPacket>& stitcher_queue,
    ThreadSafeQueue<PanoramicPacket>& output_queue,
    vector<detail::CameraParams> camera_params,
    const Mat camera_intrinsic_K,
    const Mat camera_intrinsic_distortion_coefficients,
    const Size calibration_image_size,
    const vector<UMat>& image_masks,
    const vector<vector<uint32_t>>& reference_bgr_value_idxs,
    const vector<vector<double>>& reference_bgr_cumsum)
: match_histogram_(false),
  crop_offset_x_(crop_offset_x),
  crop_offset_y_(crop_offset_y),
  crop_width_(crop_width),
  crop_height_(crop_height),
  stitcher_queue_(stitcher_queue),
  output_queue_(output_queue),
  camera_params_(camera_params),
  camera_intrinsic_K_(camera_intrinsic_K),
  camera_intrinsic_distortion_coefficients_(camera_intrinsic_distortion_coefficients),
  calibration_image_size_(calibration_image_size),
  image_masks_(image_masks),
  reference_bgr_value_idxs_(reference_bgr_value_idxs),
  reference_bgr_cumsum_(reference_bgr_cumsum),
  running_(false),
  done_(false) {

    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < camera_params_.size(); ++i) {
        focals.push_back(camera_params_[i].focal);
    }

    sort(focals.begin(), focals.end());
    if (focals.size() % 2 == 1)
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
}


FrameStitcher::~FrameStitcher() {
}


void FrameStitcher::start() {
    if (thread_.joinable()) {
        return;
    }
    running_ = true;
    done_ = false;
    thread_ = move(thread(&FrameStitcher::run, this));
}


void FrameStitcher::stop() {
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

bool FrameStitcher::is_done() {
    return done_.load();
} 

void FrameStitcher::run() {
    #ifdef _GNU_SOURCE
    pthread_setname_np(pthread_self(), "FrameStitcher");
    #endif
   
    vector<Mat> images(2);
    Mat panoramic_image;
    Mat panoramic_image_yuv;
    Mat rs;

    Mat view, rview, map1, map2;
    Mat optimized_camera_matrix = getOptimalNewCameraMatrix(camera_intrinsic_K_, camera_intrinsic_distortion_coefficients_, calibration_image_size_, 0);
    initUndistortRectifyMap(camera_intrinsic_K_, camera_intrinsic_distortion_coefficients_, Mat(), optimized_camera_matrix, calibration_image_size_,  CV_16SC2, map1, map2);

    vector<Size> image_sizes = {calibration_image_size_, calibration_image_size_};
    ImageCompositing compositor(camera_params_, image_masks_, image_sizes);

    while(running_) {
        unique_ptr<LeftRightPacket> left_right_packet(stitcher_queue_.pop(chrono::seconds(1)));
        if(left_right_packet) {
            Mat left(left_right_packet->height * 3/2, left_right_packet->width, CV_8UC1, left_right_packet->left_data.get());
            Mat right(left_right_packet->height * 3/2, left_right_packet->width, CV_8UC1, left_right_packet->right_data.get());

            cvtColor(left, images[0], COLOR_YUV2BGR_I420);
            cvtColor(right, images[1], COLOR_YUV2BGR_I420);
            remap(images[0], images[0], map1, map2, INTER_LINEAR);
            remap(images[1], images[1], map1, map2, INTER_LINEAR);

            if(match_histogram_)
                MatchHistograms(images[1], reference_bgr_cumsum_, reference_bgr_value_idxs_);

            compositor.compose(images, panoramic_image);

            Mat cropped_image(panoramic_image, Range(crop_offset_y_, crop_offset_y_+crop_height_), Range(crop_offset_x_, crop_offset_x_+crop_width_));

            cvtColor(cropped_image, panoramic_image_yuv, COLOR_BGR2YUV_I420);

            unique_ptr<PanoramicPacket> pano_packet(new PanoramicPacket);
            pano_packet->data_size = panoramic_image_yuv.total() * panoramic_image_yuv.elemSize();
            pano_packet->data = unique_ptr<uint8_t[]>(new uint8_t[pano_packet->data_size]);

            memcpy(pano_packet->data.get(), panoramic_image_yuv.data, pano_packet->data_size);

            // Don't use W H from YUV frame. Opencv rows cols represent data row col, not image row col. So it's wrong for sampled data like YUV4XX not 444.
            pano_packet->width = cropped_image.cols;
            pano_packet->height = cropped_image.rows;
            pano_packet->pts = left_right_packet->pts;
            pano_packet->pts_time = left_right_packet->pts_time;
            pano_packet->idx = left_right_packet->idx;
            left_right_packet.reset();

            output_queue_.push(move(pano_packet));
        }
    }
    done_ = true;

}
