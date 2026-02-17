#include "compositing.hpp"

#include <spdlog/spdlog.h>
#include <csignal>

using namespace std;
using namespace cv;
using namespace cv::detail;

float compute_warped_image_scale(const vector<CameraParams>& cameras) {
    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i) {
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    return warped_image_scale;
}

ImageCompositing::ImageCompositing(bool do_blending, bool use_gpu, const vector<cv::detail::CameraParams>& cameras, const std::vector<cv::UMat>& masks_warped, const vector<Size> images_size)
: do_blending_(do_blending), use_gpu_(use_gpu), cameras_parameters_(cameras), blending_masks_(masks_warped.size()), corners_(cameras.size()), sizes_(cameras.size()), top_left_(numeric_limits<int32_t>::max(), numeric_limits<int32_t>::  max()) {

    if (do_blending_)
        spdlog::info("[compositing] Using multiband blending");
    else
        spdlog::info("[compositing] Blending disabled");

    float warped_image_scale = compute_warped_image_scale(cameras);
    Ptr<WarperCreator> warper_creator;

    if (use_gpu_)
        warper_creator = makePtr<cv::CylindricalWarperGpu>();
    else
        warper_creator = makePtr<cv::CylindricalWarper>();
    if (!warper_creator) {
        throw runtime_error("Can't create the Cylindrical warper");
    }

    warper_ = warper_creator->create(warped_image_scale);

    uint32_t num_images = cameras.size();
    for (int i = 0; i < num_images; ++i) {
        // Update corner and size
        Size sz = images_size[i];

        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        K_CV_32Fs_.push_back(K);
        Rect roi = warper_->warpRoi(sz, K_CV_32Fs_[i], cameras[i].R);
        corners_[i] = roi.tl();
        sizes_[i] = roi.size();
    }

    Mat mask;
    Mat mask_warped;
    Mat dilated_mask;
    Mat seam_mask;

    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        if(!do_blending_) {
            masks_warped[img_idx].copyTo(blending_masks_[img_idx]);
            Point br(0, 0);

            for (int img_idx = 0; img_idx < num_images; ++img_idx) {
                if(corners_[img_idx].x < top_left_.x)
                    top_left_.x = corners_[img_idx].x;
                if(corners_[img_idx].y < top_left_.y)
                    top_left_.y = corners_[img_idx].y;

                int32_t right = sizes_[img_idx].width + corners_[img_idx].x;
                if(right > br.x)
                    br.x = right;

                int32_t bottom = sizes_[img_idx].height + corners_[img_idx].y;
                if(bottom > br.y)
                    br.y = bottom;
            }
            panoramic_image_size_.width = br.x - top_left_.x;
            panoramic_image_size_.height = br.y - top_left_.y;
        } else {
            // Warp the current image mask
            mask.create(images_size[img_idx], CV_8U);
            mask.setTo(Scalar::all(255));
            warper_->warp(mask, K_CV_32Fs_[img_idx], cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
            dilate(masks_warped[img_idx], dilated_mask, Mat());
            resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
            blending_masks_[img_idx] = seam_mask & mask_warped;
            mask.release();
        }
    }
}

ImageCompositing::~ImageCompositing() {
}
  
void ImageCompositing::compose(const vector<Mat>& images, Mat& output_image, int32_t frame_idx) {
    spdlog::trace("[compositing] Start");
    int num_images = static_cast<int>(images.size());
    vector<Mat> full_imgs(num_images);
    spdlog::trace("[compositing] Clone images");
    full_imgs[0] = images[0];
    full_imgs[1] = images[1];
    
    spdlog::trace("[compositing] Create pano placeholder");
    Mat pano_img(panoramic_image_size_, CV_8UC3, Scalar(0,0,0));

    Mat img;
    //int blend_type = Blender::NO;
    //int blend_type = Blender::FEATHER;
    int blend_type = Blender::MULTI_BAND;
    float blend_strength = 5;

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask_warped;
    Ptr<Blender> blender;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;
    bool is_compose_scale_set = false;
    double compose_scale = 1;

    for (int img_idx = num_images-1; img_idx >= 0; --img_idx) {
        spdlog::trace("[compositing] [{}] Loop start", img_idx);
        // Read image and resize it if necessary
        img = full_imgs[img_idx];
        Size img_size = img.size();

        spdlog::trace("[compositing] [{}] Converted to float", img_idx);
        if (use_gpu_) {
            // Need a lock guard because warp is NOT thread safe on GPU.
            lock_guard<mutex> lock(ImageCompositing::single_warp_mutex_);
            // Warp the current image
            warper_->warp(img, K_CV_32Fs_[img_idx], cameras_parameters_[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
        } else {
            warper_->warp(img, K_CV_32Fs_[img_idx], cameras_parameters_[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
        }

        spdlog::trace("[compositing] [{}] Image warped", img_idx);
        if(!do_blending_) {
            Mat roi_to_fill(pano_img, cv::Rect(corners_[img_idx].x-top_left_.x, corners_[img_idx].y-top_left_.y, img_warped.cols, img_warped.rows));
            img_warped.copyTo(roi_to_fill, blending_masks_[img_idx]);
            spdlog::trace("[compositing] [{}] Filled ROI with current masked image", img_idx);
        } else {
            spdlog::trace("[compositing] [{}] Warped", img_idx);
            img_warped.convertTo(img_warped_s, CV_16S);
            spdlog::trace("[compositing] [{}] converted to short", img_idx);
            img_warped.release();
            spdlog::trace("[compositing] [{}] Released", img_idx);

            if (!blender) {
                spdlog::trace("[compositing] [{}] Blending", img_idx);
                blender = Blender::createDefault(blend_type, false);
                spdlog::trace("[compositing] [{}] Create blender", img_idx);
                Rect roi = resultRoi(corners_, sizes_);
                Size dst_sz = roi.size();
                float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
                spdlog::trace("[compositing] [{}] Config", img_idx);
                if (blend_width < 1.f)
                    blender = Blender::createDefault(Blender::NO, false);
                else if (blend_type == Blender::MULTI_BAND)
                {
                    MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                    mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                }
                else if (blend_type == Blender::FEATHER)
                {
                    FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
                    fb->setSharpness(1.f/blend_width);
                }
                spdlog::trace("[compositing] [{}] Created", img_idx);
                blender->prepare(corners_, sizes_);
                spdlog::trace("[compositing] [{}] prepared", img_idx);
            }

            spdlog::trace("[compositing] [{}] Prefeed", img_idx);
            blender->feed(img_warped_s, blending_masks_[img_idx], corners_[img_idx]);
            spdlog::trace("[compositing] [{}] Fed", img_idx);
        }
    }

    if(!do_blending_) {
        spdlog::trace("[compositing] Write to output");
        // pano_img.convertTo(output_image, CV_8UC3);
        pano_img.copyTo(output_image);
    } else {
        spdlog::trace("[compositing] pre-blend");
        Mat result, result_mask;
        blender->blend(result, result_mask);
        spdlog::trace("[compositing] blended");

        spdlog::trace("[compositing] Back to 3 channel");
        result.convertTo(output_image, CV_8UC3);
    }
    spdlog::trace("[compositing] done");
}
