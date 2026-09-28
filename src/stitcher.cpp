#include "stitcher.hpp"

#include <opencv2/highgui.hpp>

#include <spdlog/spdlog.h>
#include <iostream>

using namespace std;
using namespace cv;

const double NStitcher::ORIG_RESOL = -1.0;

Ptr<NStitcher> NStitcher::create(Mode mode)
{
    Ptr<NStitcher> stitcher = makePtr<NStitcher>();

    stitcher->setRegistrationResol(0.6);
    stitcher->setSeamEstimationResol(0.1);
    stitcher->setCompositingResol(ORIG_RESOL);
    stitcher->setPanoConfidenceThresh(1);
    stitcher->setSeamFinder(makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR));
    stitcher->setBlender(makePtr<detail::MultiBandBlender>(false));
    stitcher->setFeaturesFinder(ORB::create());
    stitcher->setInterpolationFlags(INTER_LINEAR);

    stitcher->work_scale_ = 1;
    stitcher->seam_scale_ = 1;
    stitcher->seam_work_aspect_ = 1;
    stitcher->warped_image_scale_ = 1;

    switch (mode)
    {
    case PANORAMA: // PANORAMA is the default
        // mostly already setup
        stitcher->setEstimator(makePtr<detail::HomographyBasedEstimator>());
        stitcher->setWaveCorrection(true);
        stitcher->setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);
        stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(false));
        stitcher->setBundleAdjuster(makePtr<detail::BundleAdjusterRay>());
        stitcher->setWarper(makePtr<SphericalWarper>());
        stitcher->setExposureCompensator(makePtr<detail::BlocksGainCompensator>());
    break;

    case SCANS:
        stitcher->setEstimator(makePtr<detail::AffineBasedEstimator>());
        stitcher->setWaveCorrection(false);
        stitcher->setFeaturesMatcher(makePtr<detail::AffineBestOf2NearestMatcher>(false, false));
        stitcher->setBundleAdjuster(makePtr<detail::BundleAdjusterAffinePartial>());
        stitcher->setWarper(makePtr<AffineWarper>());
        stitcher->setExposureCompensator(makePtr<detail::NoExposureCompensator>());
    break;

    default:
        CV_Error(Error::StsBadArg, "Invalid stitching mode. Must be one of NStitcher::Mode");
    break;
    }

    return stitcher;
}


NStitcher::Status NStitcher::estimateTransform(InputArrayOfArrays images, InputArrayOfArrays masks)
{

    images.getUMatVector(imgs_);
    masks.getUMatVector(masks_);

    Status status;

    if ((status = matchImages()) != OK)
        return status;

    if ((status = estimateCameraParams()) != OK)
        return status;

    return OK;
}


NStitcher::Status NStitcher::composePanorama(OutputArray pano)
{

    return composePanorama(std::vector<UMat>(), pano);
}


NStitcher::Status NStitcher::composePanorama(InputArrayOfArrays images, OutputArray pano)
{


    std::vector<UMat> imgs;
    images.getUMatVector(imgs);
    spdlog::info("Pre imgs");
    if (!imgs.empty())
    {
        spdlog::info("Pre imgs yes");
        CV_Assert(imgs.size() == imgs_.size());

        UMat img;
        seam_est_imgs_.resize(imgs.size());

        for (size_t i = 0; i < imgs.size(); ++i)
        {
            imgs_[i] = imgs[i];
            resize(imgs[i], img, Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
            seam_est_imgs_[i] = img.clone();
        }

        std::vector<UMat> seam_est_imgs_subset;
        std::vector<UMat> imgs_subset;

        for (size_t i = 0; i < indices_.size(); ++i)
        {
            imgs_subset.push_back(imgs_[indices_[i]]);
            seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
        }

        seam_est_imgs_ = seam_est_imgs_subset;
        imgs_ = imgs_subset;
    }

    UMat pano_;


    spdlog::info("Create vects");
    std::vector<Point> corners(imgs_.size());
    std::vector<UMat> masks_warped(imgs_.size());
    std::vector<UMat> images_warped(imgs_.size());
    std::vector<Size> sizes(imgs_.size());
    std::vector<UMat> masks(imgs_.size());

    // Prepare image masks
    spdlog::info("Prepare image masks");
    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        masks[i].create(seam_est_imgs_[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks
    spdlog::info("Warp images and their masks");
    Ptr<detail::RotationWarper> w = warper_->create(float(warped_image_scale_ * seam_work_aspect_));
    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        Mat_<float> K;
        cameras_[i].K().convertTo(K, CV_32F);
        K(0,0) *= (float)seam_work_aspect_;
        K(0,2) *= (float)seam_work_aspect_;
        K(1,1) *= (float)seam_work_aspect_;
        K(1,2) *= (float)seam_work_aspect_;

        corners[i] = w->warp(seam_est_imgs_[i], K, cameras_[i].R, interp_flags_, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        w->warp(masks[i], K, cameras_[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }



    // Compensate exposure before finding seams
    spdlog::info("Compensate exposure before finding seams");
    exposure_comp_->feed(corners, images_warped, masks_warped);
    for (size_t i = 0; i < imgs_.size(); ++i)
        exposure_comp_->apply(int(i), corners[i], images_warped[i], masks_warped[i]);

    // Find seams
    spdlog::info("Find seams");
    std::vector<UMat> images_warped_f(imgs_.size());
    for (size_t i = 0; i < imgs_.size(); ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    seam_finder_->find(images_warped_f, corners, masks_warped);

    // Release unused memory
    spdlog::info("Release mem");
    seam_est_imgs_.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    for(int ii=0; ii < corners.size(); ii++) {
        std::cout << '[' << ii << "] " << corners[ii] << std::endl;
    }

    UMat img_warped, img_warped_s;
    UMat dilated_mask, seam_mask, mask, mask_warped;

    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;
    bool is_blender_prepared = false;

    double compose_scale = 1;
    bool is_compose_scale_set = false;

    std::vector<detail::CameraParams> cameras_scaled(cameras_);

    UMat full_img, img;
    spdlog::info("Work on images");
    for (size_t img_idx = 0; img_idx < imgs_.size(); ++img_idx)
    {
        // Read image and resize it if necessary
        full_img = imgs_[img_idx];
        if (!is_compose_scale_set)
        {
            if (compose_resol_ > 0)
                compose_scale = std::min(1.0, std::sqrt(compose_resol_ * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale_;
            compose_work_aspect = compose_scale / work_scale_;

            // Update warped image scale
            float warp_scale = static_cast<float>(warped_image_scale_ * compose_work_aspect);
            w = warper_->create(warp_scale);

            // Update corners and sizes
            for (size_t i = 0; i < imgs_.size(); ++i)
            {
                // Update intrinsics
                cameras_scaled[i].ppx *= compose_work_aspect;
                cameras_scaled[i].ppy *= compose_work_aspect;
                cameras_scaled[i].focal *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes_[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes_[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes_[i].height * compose_scale);
                }

                Mat K;
                cameras_scaled[i].K().convertTo(K, CV_32F);
                Rect roi = w->warpRoi(sz, K, cameras_scaled[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (std::abs(compose_scale - 1) > 1e-1)
        {
            resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
        }
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();


        Mat K;
        cameras_scaled[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        w->warp(img, K, cameras_[img_idx].R, interp_flags_, BORDER_REFLECT, img_warped);
        //    Mat dst;
        //    resize(img_warped, dst, Size(1280, 720), 0, 0, INTER_CUBIC);
        //    cv::imshow("Left", dst);
        //    cv::waitKey();
        //    cv::destroyWindow("Left");

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        w->warp(mask, K, cameras_[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // Compensate exposure
        exposure_comp_->apply((int)img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        // Make sure seam mask has proper size
        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);

        bitwise_and(seam_mask, mask_warped, mask_warped);


        if (!is_blender_prepared)
        {
            blender_->prepare(corners, sizes);
            is_blender_prepared = true;
    for(int ii=0; ii < corners.size(); ii++) {
        std::cout << '[' << ii << "] prep " << corners[ii] << std::endl;
    }
        }


        // Blend the current image
        blender_->feed(img_warped_s, mask_warped, corners[img_idx]);
    for(int ii=0; ii < corners.size(); ii++) {
        std::cout << '[' << ii << "] blend " << corners[ii] << std::endl;
    }
    }

    UMat result;

    spdlog::info("Blend");
    blender_->blend(result, result_mask_);


    // Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
    // so convert it to avoid user confusing
    spdlog::info("Convert to 8U");
    result.convertTo(pano, CV_8U);

    for(int ii=0; ii < corners.size(); ii++) {
        std::cout << '[' << ii << "] final " << corners[ii] << std::endl;
    }
    return OK;
}


NStitcher::Status NStitcher::stitch(InputArrayOfArrays images, OutputArray pano)
{
    return stitch(images, noArray(), pano);
}


NStitcher::Status NStitcher::stitch(InputArrayOfArrays images, InputArrayOfArrays masks, OutputArray pano)
{

    Status status = estimateTransform(images, masks);
    if (status != OK)
        return status;
    return composePanorama(pano);
}


NStitcher::Status NStitcher::matchImages()
{
    if ((int)imgs_.size() < 2)
    {
        return ERR_NEED_MORE_IMGS;
    }

    work_scale_ = 1;
    seam_work_aspect_ = 1;
    seam_scale_ = 1;
    bool is_work_scale_set = false;
    bool is_seam_scale_set = false;
    features_.resize(imgs_.size());
    seam_est_imgs_.resize(imgs_.size());
    full_img_sizes_.resize(imgs_.size());


    std::vector<UMat> feature_find_imgs(imgs_.size());
    std::vector<UMat> feature_find_masks(masks_.size());

    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        full_img_sizes_[i] = imgs_[i].size();
        if (registr_resol_ < 0)
        {
            feature_find_imgs[i] = imgs_[i];
            work_scale_ = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale_ = std::min(1.0, std::sqrt(registr_resol_ * 1e6 / full_img_sizes_[i].area()));
                is_work_scale_set = true;
            }
            resize(imgs_[i], feature_find_imgs[i], Size(), work_scale_, work_scale_, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale_ = std::min(1.0, std::sqrt(seam_est_resol_ * 1e6 / full_img_sizes_[i].area()));
            seam_work_aspect_ = seam_scale_ / work_scale_;
            is_seam_scale_set = true;
        }

        if (!masks_.empty())
        {
            resize(masks_[i], feature_find_masks[i], Size(), work_scale_, work_scale_, INTER_NEAREST);
        }
        features_[i].img_idx = (int)i;

        resize(imgs_[i], seam_est_imgs_[i], Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
    }

    // find features possibly in parallel
    detail::computeImageFeatures(features_finder_, feature_find_imgs, features_, feature_find_masks);

    // Do it to save memory
    feature_find_imgs.clear();
    feature_find_masks.clear();


    (*features_matcher_)(features_, pairwise_matches_, matching_mask_);
    features_matcher_->collectGarbage();

    // Leave only images we are sure are from the same panorama
    indices_ = detail::leaveBiggestComponent(features_, pairwise_matches_, (float)conf_thresh_);
    std::vector<UMat> seam_est_imgs_subset;
    std::vector<UMat> imgs_subset;
    std::vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices_.size(); ++i)
    {
        imgs_subset.push_back(imgs_[indices_[i]]);
        seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
        full_img_sizes_subset.push_back(full_img_sizes_[indices_[i]]);
    }
    seam_est_imgs_ = seam_est_imgs_subset;
    imgs_ = imgs_subset;
    full_img_sizes_ = full_img_sizes_subset;

    if ((int)imgs_.size() < 2)
    {
        return ERR_NEED_MORE_IMGS;
    }

    return OK;
}


NStitcher::Status NStitcher::estimateCameraParams()
{
    // estimate homography in global frame
    if (!(*estimator_)(features_, pairwise_matches_, cameras_))
        return ERR_HOMOGRAPHY_EST_FAIL;

    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        Mat R;
        cameras_[i].R.convertTo(R, CV_32F);
        cameras_[i].R = R;
    }

    bundle_adjuster_->setConfThresh(conf_thresh_);
    if (!(*bundle_adjuster_)(features_, pairwise_matches_, cameras_))
        return ERR_CAMERA_PARAMS_ADJUST_FAIL;

    // Find median focal length and use it as final image scale
    std::vector<double> focals;
    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        focals.push_back(cameras_[i].focal);
    }

    std::sort(focals.begin(), focals.end());
    if (focals.size() % 2 == 1)
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    if (do_wave_correct_)
    {
        std::vector<Mat> rmats;
        for (size_t i = 0; i < cameras_.size(); ++i)
            rmats.push_back(cameras_[i].R.clone());
        detail::waveCorrect(rmats, wave_correct_kind_);
        for (size_t i = 0; i < cameras_.size(); ++i)
            cameras_[i].R = rmats[i];
    }

    return OK;
}

NStitcher::Status NStitcher::setTransform(InputArrayOfArrays images, const std::vector<detail::CameraParams> &cameras)
{
    std::vector<int> component;
    for (int i = 0; i < (int)images.total(); i++)
        component.push_back(i);

    return setTransform(images, cameras, component);
}


NStitcher::Status NStitcher::setTransformCams(InputArrayOfArrays images, const detail::CameraParams &cameras1, const detail::CameraParams &cameras2) {

    std::vector<detail::CameraParams> cameras;
    cameras.push_back(cameras1);
    cameras.push_back(cameras2);
    return setTransform(images, cameras);
}

NStitcher::Status NStitcher::setTransform(
        InputArrayOfArrays images, const std::vector<detail::CameraParams> &cameras, const std::vector<int> &component)
{
//    CV_Assert(images.size() == cameras.size());

    images.getUMatVector(imgs_);
    masks_.clear();

    if ((int)imgs_.size() < 2)
    {
        return ERR_NEED_MORE_IMGS;
    }

    work_scale_ = 1;
    seam_work_aspect_ = 1;
    seam_scale_ = 1;
    bool is_work_scale_set = false;
    bool is_seam_scale_set = false;
    seam_est_imgs_.resize(imgs_.size());
    full_img_sizes_.resize(imgs_.size());


    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        full_img_sizes_[i] = imgs_[i].size();
        if (registr_resol_ < 0)
        {
            std::cout << "registr_resol_ = " << registr_resol_ << std::endl;
            work_scale_ = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale_ = std::min(1.0, std::sqrt(registr_resol_ * 1e6 / full_img_sizes_[i].area()));
                is_work_scale_set = true;
            }
        }
        if (!is_seam_scale_set)
        {
            seam_scale_ = std::min(1.0, std::sqrt(seam_est_resol_ * 1e6 / full_img_sizes_[i].area()));
            seam_work_aspect_ = seam_scale_ / work_scale_;
            is_seam_scale_set = true;
        }

        resize(imgs_[i], seam_est_imgs_[i], Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
    }

    features_.clear();
    pairwise_matches_.clear();

    indices_ = component;
    std::vector<UMat> seam_est_imgs_subset;
    std::vector<UMat> imgs_subset;
    std::vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices_.size(); ++i)
    {
        imgs_subset.push_back(imgs_[indices_[i]]);
        seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
        full_img_sizes_subset.push_back(full_img_sizes_[indices_[i]]);
    }
    seam_est_imgs_ = seam_est_imgs_subset;
    imgs_ = imgs_subset;
    full_img_sizes_ = full_img_sizes_subset;

    if ((int)imgs_.size() < 2)
    {
        return ERR_NEED_MORE_IMGS;
    }

    cameras_ = cameras;

    std::vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
        focals.push_back(cameras_[i].focal);

    std::sort(focals.begin(), focals.end());
    if (focals.size() % 2 == 1)
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    return Status::OK;
}
