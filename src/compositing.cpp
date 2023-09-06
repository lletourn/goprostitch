#include "compositing.hpp"

#include <spdlog/spdlog.h>

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

ImageCompositing::ImageCompositing(const vector<cv::detail::CameraParams>& cameras, const std::vector<cv::UMat>& masks_warped, const vector<Size> images_size)
: cameras_parameters_(cameras), blending_masks_(masks_warped.size()), corners_(cameras.size()), sizes_(cameras.size()) {

    float warped_image_scale = compute_warped_image_scale(cameras);
    Ptr<WarperCreator> warper_creator = makePtr<cv::CylindricalWarper>();
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
        Rect roi = warper_->warpRoi(sz, K, cameras[i].R);
        corners_[i] = roi.tl();
        sizes_[i] = roi.size();
    }

    Mat mask;
    Mat mask_warped;
    Mat dilated_mask;
    Mat seam_mask;
    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);
        // Warp the current image mask
        mask.create(images_size[img_idx], CV_8U);
        mask.setTo(Scalar::all(255));
        warper_->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        blending_masks_[img_idx] = seam_mask & mask_warped;

        mask.release();
    }
}

ImageCompositing::~ImageCompositing() {
}
  
void ImageCompositing::compose(const vector<Mat>& images, Mat& output_image) {
    int num_images = static_cast<int>(images.size());
    vector<Mat> full_imgs(num_images);
    full_imgs[0] = images[0].clone();
    full_imgs[1] = images[1].clone();
    
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

    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        // Read image and resize it if necessary
        img = full_imgs[img_idx];
        Size img_size = img.size();

        Mat K;
        cameras_parameters_[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper_->warp(img, K, cameras_parameters_[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();

        if (!blender) {
            blender = Blender::createDefault(blend_type, false);
            Rect roi = resultRoi(corners_, sizes_);
            Size dst_sz = roi.size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
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
            blender->prepare(corners_, sizes_);
        }

        //imshow("0 iwar", img_warped_s);
        //imshow("0 mwar", mask_warped);
        //waitKey(0);
        blender->feed(img_warped_s, blending_masks_[img_idx], corners_[img_idx]);
    }

    Mat result, result_mask;
    blender->blend(result, result_mask);

    result.convertTo(output_image, CV_8UC3);
}
