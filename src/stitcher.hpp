#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>


class NStitcher {
public:
    static const double ORIG_RESOL; // Initialized in stitcher.cpp

    enum Status
    {
        OK = 0,
        ERR_NEED_MORE_IMGS = 1,
        ERR_HOMOGRAPHY_EST_FAIL = 2,
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
    };

    enum Mode
    {
        /** Mode for creating photo panoramas. Expects images under perspective
        transformation and projects resulting pano to sphere.

        @sa cv::detail::BestOf2NearestMatcher SphericalWarper
        */
        PANORAMA = 0,
        /** Mode for composing scans. Expects images under affine transformation does
        not compensate exposure by default.

        @sa cv::detail::AffineBestOf2NearestMatcher AffineWarper
        */
        SCANS = 1,

    };

    /** @brief Creates a Stitcher configured in one of the stitching modes.

    @param mode Scenario for stitcher operation. This is usually determined by source of images
    to stitch and their transformation. Default parameters will be chosen for operation in given
    scenario.
    @return Stitcher class instance.
     */
    static cv::Ptr<NStitcher> create(Mode mode = NStitcher::PANORAMA);

    double registrationResol() const { return registr_resol_; }
    void setRegistrationResol(double resol_mpx) { registr_resol_ = resol_mpx; }

    double seamEstimationResol() const { return seam_est_resol_; }
    void setSeamEstimationResol(double resol_mpx) { seam_est_resol_ = resol_mpx; }

    double compositingResol() const { return compose_resol_; }
    void setCompositingResol(double resol_mpx) { compose_resol_ = resol_mpx; }

    double panoConfidenceThresh() const { return conf_thresh_; }
    void setPanoConfidenceThresh(double conf_thresh) { conf_thresh_ = conf_thresh; }

    bool waveCorrection() const { return do_wave_correct_; }
    void setWaveCorrection(bool flag) { do_wave_correct_ = flag; }

    cv::InterpolationFlags interpolationFlags() const { return interp_flags_; }
    void setInterpolationFlags(cv::InterpolationFlags interp_flags) { interp_flags_ = interp_flags; }

    cv::detail::WaveCorrectKind waveCorrectKind() const { return wave_correct_kind_; }
    void setWaveCorrectKind(cv::detail::WaveCorrectKind kind) { wave_correct_kind_ = kind; }

    cv::Ptr<cv::Feature2D> featuresFinder() { return features_finder_; }
    cv::Ptr<cv::Feature2D> featuresFinder() const { return features_finder_; }
    void setFeaturesFinder(cv::Ptr<cv::Feature2D> features_finder)
        { features_finder_ = features_finder; }

    cv::Ptr<cv::detail::FeaturesMatcher> featuresMatcher() { return features_matcher_; }
    cv::Ptr<cv::detail::FeaturesMatcher> featuresMatcher() const { return features_matcher_; }
    void setFeaturesMatcher(cv::Ptr<cv::detail::FeaturesMatcher> features_matcher)
        { features_matcher_ = features_matcher; }

    const cv::UMat& matchingMask() const { return matching_mask_; }
    void setMatchingMask(const cv::UMat &mask)
    {
        CV_Assert(mask.type() == CV_8U && mask.cols == mask.rows);
        matching_mask_ = mask.clone();
    }

    cv::Ptr<cv::detail::BundleAdjusterBase> bundleAdjuster() { return bundle_adjuster_; }
    const cv::Ptr<cv::detail::BundleAdjusterBase> bundleAdjuster() const { return bundle_adjuster_; }
    void setBundleAdjuster(cv::Ptr<cv::detail::BundleAdjusterBase> bundle_adjuster)
        { bundle_adjuster_ = bundle_adjuster; }

    cv::Ptr<cv::detail::Estimator> estimator() { return estimator_; }
    const cv::Ptr<cv::detail::Estimator> estimator() const { return estimator_; }
    void setEstimator(cv::Ptr<cv::detail::Estimator> estimator)
        { estimator_ = estimator; }

    cv::Ptr<cv::WarperCreator> warper() { return warper_; }
    const cv::Ptr<cv::WarperCreator> warper() const { return warper_; }
    void setWarper(cv::Ptr<cv::WarperCreator> creator) { warper_ = creator; }

    cv::Ptr<cv::detail::ExposureCompensator> exposureCompensator() { return exposure_comp_; }
    const cv::Ptr<cv::detail::ExposureCompensator> exposureCompensator() const { return exposure_comp_; }
    void setExposureCompensator(cv::Ptr<cv::detail::ExposureCompensator> exposure_comp)
        { exposure_comp_ = exposure_comp; }

    cv::Ptr<cv::detail::SeamFinder> seamFinder() { return seam_finder_; }
    const cv::Ptr<cv::detail::SeamFinder> seamFinder() const { return seam_finder_; }
    void setSeamFinder(cv::Ptr<cv::detail::SeamFinder> seam_finder) { seam_finder_ = seam_finder; }

    cv::Ptr<cv::detail::Blender> blender() { return blender_; }
    const cv::Ptr<cv::detail::Blender> blender() const { return blender_; }
    void setBlender(cv::Ptr<cv::detail::Blender> b) { blender_ = b; }

    /** @brief These functions try to match the given images and to estimate rotations of each camera.

    @note Use the functions only if you're aware of the stitching pipeline, otherwise use
    Stitcher::stitch.

    @param images Input images.
    @param masks Masks for each input image specifying where to look for keypoints (optional).
    @return Status code.
     */
    Status estimateTransform(cv::InputArrayOfArrays images, cv::InputArrayOfArrays masks = cv::noArray());

    /** @brief These function restors camera rotation and camera intrinsics of each camera
     *  that can be got with @ref Stitcher::cameras call

    @param images Input images.
    @param cameras Estimated rotation of cameras for each of the input images.
    @param component Indices (0-based) of images constituting the final panorama (optional).
    @return Status code.
     */
    Status setTransform(cv::InputArrayOfArrays images,
                        const std::vector<cv::detail::CameraParams> &cameras,
                        const std::vector<int> &component);
    /** @overload */
    Status setTransform(cv::InputArrayOfArrays images, const std::vector<cv::detail::CameraParams> &cameras);
    Status setTransformCams(cv::InputArrayOfArrays images, const cv::detail::CameraParams &cameras1, const cv::detail::CameraParams &cameras2);

    /** @overload */
    Status composePanorama(cv::OutputArray pano);
    /** @brief These functions try to compose the given images (or images stored internally from the other function
    calls) into the final pano under the assumption that the image transformations were estimated
    before.

    @note Use the functions only if you're aware of the stitching pipeline, otherwise use
    Stitcher::stitch.

    @param images Input images.
    @param pano Final pano.
    @return Status code.
     */
    Status composePanorama(cv::InputArrayOfArrays images, cv::OutputArray pano);

    /** @overload */
    Status stitch(cv::InputArrayOfArrays images, cv::OutputArray pano);
    /** @brief These functions try to stitch the given images.

    @param images Input images.
    @param masks Masks for each input image specifying where to look for keypoints (optional).
    @param pano Final pano.
    @return Status code.
     */
    Status stitch(cv::InputArrayOfArrays images, cv::InputArrayOfArrays masks, cv::OutputArray pano);

    std::vector<int> component() const { return indices_; }
    std::vector<cv::detail::CameraParams> cameras() const { return cameras_; }
    cv::detail::CameraParams cameras(int i) const { return cameras_[i]; }
    int camerascnt() const { return cameras_.size(); }
    double workScale() const { return work_scale_; }

    /** @brief Return the mask of the panorama.

    The mask is a 8U cv::UMat with the values: 0xFF (white) for pixels filled by the input images,
    0 (black) for unused pixels. It can be used as the mask for inpaint.

    @return The mask.
     */
    cv::UMat resultMask() const { return result_mask_; }

private:
    Status matchImages();
    Status estimateCameraParams();

    double registr_resol_;
    double seam_est_resol_;
    double compose_resol_;
    double conf_thresh_;
    cv::InterpolationFlags interp_flags_;
    cv::Ptr<cv::Feature2D> features_finder_;
    cv::Ptr<cv::detail::FeaturesMatcher> features_matcher_;
    cv::UMat matching_mask_;
    cv::Ptr<cv::detail::BundleAdjusterBase> bundle_adjuster_;
    cv::Ptr<cv::detail::Estimator> estimator_;
    bool do_wave_correct_;
    cv::detail::WaveCorrectKind wave_correct_kind_;
    cv::Ptr<cv::WarperCreator> warper_;
    cv::Ptr<cv::detail::ExposureCompensator> exposure_comp_;
    cv::Ptr<cv::detail::SeamFinder> seam_finder_;
    cv::Ptr<cv::detail::Blender> blender_;

    std::vector<cv::UMat> imgs_;
    std::vector<cv::UMat> masks_;
    std::vector<cv::Size> full_img_sizes_;
    std::vector<cv::detail::ImageFeatures> features_;
    std::vector<cv::detail::MatchesInfo> pairwise_matches_;
    std::vector<cv::UMat> seam_est_imgs_;
    std::vector<int> indices_;
    std::vector<cv::detail::CameraParams> cameras_;
    cv::UMat result_mask_;
    double work_scale_;
    double seam_scale_;
    double seam_work_aspect_;
    double warped_image_scale_;
};
