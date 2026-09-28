#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <opencv2/core/ocl.hpp>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include <spdlog/spdlog.h>

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

#include "compositing.hpp"
#include "readers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace rapidjson;

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void rotateImage(const Mat& src, Mat& dst, int width, int height, double angle) {
    Point2f center = Point2f(width / 2.0, height / 2.0);
    Mat M = getRotationMatrix2D(center, angle, 1.0);
    warpAffine(src, dst, M, Size(width, height), INTER_LINEAR, BORDER_CONSTANT, (0, 0, 0));
}

void finding(const string& features_type, const vector<Mat>& src_images, float overlap_fraction, const vector<PointPair>& point_pairs, vector<CameraParams>& cameras, vector<UMat>& masks_warped);

Rect cropPano(const Mat& panorama, const Rect& initial_rect);

void writeStitchingData(const string& cameras_params_filename, const vector<CameraParams>& cameras, const vector<UMat>& masks_warped, const Rect& crop_rect, double left_rotation, double right_rotation) {
    rapidjson::Document stitching_doc(kObjectType);

    Value cameras_doc(kArrayType);
    for(uint32_t cam_idx=0; cam_idx < cameras.size(); ++cam_idx){
        stringstream ss;
        ss << "warped_seam_mask_" << cam_idx << ".png";
        imwrite(ss.str(), masks_warped[cam_idx]);
        spdlog::info("Mask type: {}", type2str(masks_warped[cam_idx].type()));

        const CameraParams& camera_params = cameras[cam_idx];
        Value camera(kObjectType);
        camera.AddMember("aspect", Value(camera_params.aspect), stitching_doc.GetAllocator());
        camera.AddMember("focal", Value(camera_params.focal), stitching_doc.GetAllocator());
        camera.AddMember("ppx", Value(camera_params.ppx), stitching_doc.GetAllocator());
        camera.AddMember("ppy", Value(camera_params.ppy), stitching_doc.GetAllocator());

        if(camera_params.R.depth() == CV_32F)
            spdlog::info("R is 32bit float");
        else if(camera_params.R.depth() == CV_64F)
            spdlog::info("R is 64bit double");
        else
            spdlog::error("R type is unknown");
            
        Value R(kArrayType);
        for(uint32_t i = 0; i < camera_params.R.rows; ++i) {
            Value R_row(kArrayType);
            for(uint32_t j = 0; j < camera_params.R.cols; ++j ) {
                R_row.PushBack(Value(camera_params.R.at<float>(i,j)), stitching_doc.GetAllocator());
            }
            R.PushBack(R_row, stitching_doc.GetAllocator());
        }
        camera.AddMember("R", R, stitching_doc.GetAllocator());

        if(camera_params.t.depth() == CV_32F)
            spdlog::error("t is 32bit float");
        else if(camera_params.t.depth() == CV_64F)
            spdlog::error("t is 64bit double");
        else
            spdlog::error("t type is unknown");
        Value t(kArrayType);
        for(uint32_t i = 0; i < camera_params.t.rows; ++i) {
            t.PushBack(Value(camera_params.t.at<double>(i)), stitching_doc.GetAllocator());
        }
        camera.AddMember("t", t, stitching_doc.GetAllocator());

        cameras_doc.PushBack(camera, stitching_doc.GetAllocator());
    }
    stitching_doc.AddMember("cameras_params", cameras_doc, stitching_doc.GetAllocator());

    Value pano_crop(kObjectType);
    pano_crop.AddMember("x", crop_rect.x, stitching_doc.GetAllocator());
    pano_crop.AddMember("y", crop_rect.y, stitching_doc.GetAllocator());
    pano_crop.AddMember("w", crop_rect.width, stitching_doc.GetAllocator());
    pano_crop.AddMember("h", crop_rect.height, stitching_doc.GetAllocator());
    stitching_doc.AddMember("crop", pano_crop, stitching_doc.GetAllocator());

    stitching_doc.AddMember("left_rotation", left_rotation, stitching_doc.GetAllocator());
    stitching_doc.AddMember("right_rotation", right_rotation, stitching_doc.GetAllocator());

    ofstream ofs(cameras_params_filename);
    OStreamWrapper osw(ofs);
 
    Writer<OStreamWrapper> writer(osw);
    stitching_doc.Accept(writer);
}

int main(int argc, char* argv[]) {
    spdlog::set_pattern("%Y%m%dT%H:%M:%S.%e [%^%l%$] -%n- -%t- : %v");
    spdlog::set_level(spdlog::level::trace);

    cv::setBreakOnError(true);
    cv::ocl::setUseOpenCL(false);

    const String keys =
        "{help h usage ? | | print this message }"
        "{left |<none>| Left image }"
        "{right |<none>| Right image }"
        "{leftrotation |0.0| Left image rotation }"
        "{rightrotation |0.0| Right image rotation }"
        "{keypoints | | Left-Right keypoints json filename}"
        "{output |<none>| Output panorama }"
        "{camparams | | Camera parameter filename }"
        "{findcamparams | false | Generate camera parameters or read them from the file }"
        "{featuresfinder | sift | Features finder to use. sift, surf, orb}"
        "{overlapfraction | 0.25 | Fraction of each image's X range to search for features (kept strip width); left keeps the rightmost fraction, right keeps the leftmost }"
        "{cropsize | 4730x1630 | Output crop size WxH }"
    ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("Seam finder");

    if(parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    double left_rotation = parser.get<double>("leftrotation");
    double right_rotation = parser.get<double>("rightrotation");
    Mat tmp_l = imread(parser.get<string>("left"));
    Mat tmp_r = imread(parser.get<string>("right"));
    if(left_rotation != 0)
        rotateImage(tmp_l, tmp_l, tmp_l.cols, tmp_l.rows, left_rotation);
    if(right_rotation != 0)
        rotateImage(tmp_r, tmp_r, tmp_r.cols, tmp_r.rows, right_rotation);

    vector<Mat> images = {tmp_l, tmp_r};
    vector<Size> images_size = {images[0].size(), images[1].size()};
    string result_name = parser.get<string>("output");

    Size crop_size;
    char crop_sep = 0;
    istringstream crop_ss(parser.get<string>("cropsize"));
    if(!(crop_ss >> crop_size.width >> crop_sep >> crop_size.height) || (crop_sep != 'x' && crop_sep != 'X') || crop_size.width <= 0 || crop_size.height <= 0) {
        spdlog::error("Invalid cropsize '{}', expected WxH", parser.get<string>("cropsize"));
        return 1;
    }

    vector<CameraParams> cameras;
    vector<UMat> masks_warped;
    // Negative position means "centre the crop box once the panorama is known"
    Rect crop_rect(-1, -1, crop_size.width, crop_size.height);

    if(!parser.get<bool>("findcamparams")) {
        spdlog::info("From cameras params");
        Rect saved_rect;
        double left_rotation;
        double right_rotation;
        readSeamData(parser.get<string>("camparams"), cameras, masks_warped, saved_rect, left_rotation, right_rotation);
        crop_rect.x = saved_rect.x;
        crop_rect.y = saved_rect.y;
    } else {
        spdlog::info("Generate camera params");
        string features_type = parser.get<string>("featuresfinder");

        float overlap_fraction = parser.get<float>("overlapfraction");
        if(overlap_fraction <= 0.0f || overlap_fraction > 1.0f) {
            spdlog::warn("overlapfraction {} out of range (0, 1]; clamping to 0.25", overlap_fraction);
            overlap_fraction = 0.25f;
        }

        vector<PointPair> point_pairs;
        if(parser.has("keypoints"))
            point_pairs = readPointPairs(parser.get<string>("keypoints"));
        finding(features_type, images, overlap_fraction, point_pairs, cameras, masks_warped);
    }

    Mat output_image;
    ImageCompositing compositor(true, false, cameras, masks_warped, images_size);
    spdlog::info("Pre compo");
    compositor.compose(images, output_image);
    spdlog::info("Compo");

    imwrite(result_name, output_image);

    if(output_image.cols < crop_size.width || output_image.rows < crop_size.height) {
        spdlog::error("Panorama {}x{} is smaller than the crop size {}x{}", output_image.cols, output_image.rows, crop_size.width, crop_size.height);
        return 1;
    }
    if(crop_rect.x < 0 || crop_rect.y < 0) {
        crop_rect.x = (output_image.cols - crop_size.width) / 2;
        crop_rect.y = (output_image.rows - crop_size.height) / 2;
    }

    Rect rect = cropPano(output_image, crop_rect);
    writeStitchingData(parser.get<string>("camparams"), cameras, masks_warped, rect, left_rotation, right_rotation);
}

// Fixed-size crop box; the arrows move the whole box 1px at a time, clamped to the panorama.
Rect cropPano(const Mat& panorama, const Rect& initial_rect) {
    const int max_x = panorama.cols - initial_rect.width;
    const int max_y = panorama.rows - initial_rect.height;
    Rect rect = initial_rect;
    rect.x = std::clamp(rect.x, 0, max_x);
    rect.y = std::clamp(rect.y, 0, max_y);

    namedWindow("Pano", WINDOW_NORMAL);
    resizeWindow("Pano", 1920, 720);

    bool redraw = true;
    while(true) {
        if(redraw) {
            cout << "Rect: " << rect << endl;
            Mat tmp = panorama.clone();
            rectangle(tmp, rect, Scalar(0,255,0), 5);
            imshow("Pano", tmp);
            redraw = false;
        }

        int key = waitKeyEx(30);
        if(key == -1 || key == 65535)
            continue;

        if(key == 13 || key == 27 || key == 10) { // CR, ESC, or LF
            break;
        } else if(key == 65361) { // Left
            rect.x = std::max(rect.x - 1, 0);
        } else if(key == 65362) { // Up
            rect.y = std::max(rect.y - 1, 0);
        } else if(key == 65363) { // Right
            rect.x = std::min(rect.x + 1, max_x);
        } else if(key == 65364) { // Down
            rect.y = std::min(rect.y + 1, max_y);
        } else {
            continue;
        }
        redraw = true;
    }
    destroyAllWindows();

    return rect;
}

void featurePairAuto(const string& features_type, const vector<Mat>& images, float conf_thresh, float overlap_fraction, vector<ImageFeatures>& features, vector<MatchesInfo>& pairwise_matches) {
    float match_conf = 0.50f;

    // Only the overlap region of each image carries useful correspondences, so
    // feature detection is restricted to that strip (see overlap_fraction; passed
    // in via the --overlapfraction CLI arg). We never feed keypoints from the
    // non-overlapping part of the frame into the homography estimate.
    // Left image (index 0): keep the rightmost fraction of the X range.
    // Right image (others): keep the leftmost fraction of the X range.

    spdlog::info("Finding features (overlap fraction {})...", overlap_fraction);
    Ptr<Feature2D> finder;
    if (features_type == "orb") {
        finder = ORB::create();
    }
    else if (features_type == "akaze") {
        finder = AKAZE::create();
    }
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf") {
        finder = xfeatures2d::SURF::create();
    }
#endif
    else if (features_type == "sift") {
        finder = SIFT::create();
    }
    else {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
    }

    for (int i = 0; i < images.size(); ++i) {
        // Build a detection mask covering only the overlap strip. Keypoint
        // coordinates and features[i].img_size stay in full-image space, so the
        // downstream estimator's principal-point assumption is preserved.
        Mat mask(images[i].size(), CV_8U, Scalar(0));
        const int w = images[i].cols;
        if (i == 0) {
            const int x0 = cvRound((1.0f - overlap_fraction) * w);
            mask(Rect(x0, 0, w - x0, images[i].rows)).setTo(255);
        } else {
            const int x1 = cvRound(overlap_fraction * w);
            mask(Rect(0, 0, x1, images[i].rows)).setTo(255);
        }

        computeImageFeatures(finder, images[i], features[i], mask);
        features[i].img_idx = i;
        spdlog::info("Features in image #{}: {}", i+1, features[i].keypoints.size());
    }

    cout << "Features: " << features.size() << endl;

    spdlog::info("Pairwise matching");
    Ptr<FeaturesMatcher> matcher;
    matcher = makePtr<BestOf2NearestMatcher>(false, match_conf);

    (*matcher)(features, pairwise_matches);
    cout << "Post matcher: Features: " << features.size() << " PW: " << pairwise_matches.size() << endl;
    matcher->collectGarbage();

    spdlog::info("Draw match");
    cout << "FEatures: " << features.size() << endl;
    cout << "PW: " << pairwise_matches.size() << endl;
    Mat pair_img;
    drawMatches(images[0], features[0].getKeypoints(), images[1], features[1].getKeypoints(), pairwise_matches[1].getMatches(), pair_img);

    // Draw the overlap-strip boundary on each side. drawMatches lays image 0 on
    // the left and image 1 on the right (offset by image 0's width), so the
    // right-image boundary is shifted by images[0].cols.
    const int left_boundary_x = cvRound((1.0f - overlap_fraction) * images[0].cols);
    const int right_boundary_x = images[0].cols + cvRound(overlap_fraction * images[1].cols);
    line(pair_img, Point(left_boundary_x, 0), Point(left_boundary_x, pair_img.rows), Scalar(0, 0, 255), 3);
    line(pair_img, Point(right_boundary_x, 0), Point(right_boundary_x, pair_img.rows), Scalar(0, 0, 255), 3);

    imwrite("pair.png", pair_img);

    spdlog::info("Find matching images");
    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    cout << "Features big: " << features.size() << endl;
    cout << "PW big: " << pairwise_matches.size() << endl;
    spdlog::info("Biggest component indices");
}

void featurePairManual(const vector<PointPair>& point_pairs, const vector<Size>& image_sizes, vector<ImageFeatures>& features, vector<MatchesInfo>& pairwise_matches) {

    ImageFeatures feature_left;
    feature_left.img_idx = 0;
    feature_left.img_size = image_sizes[0];
    ImageFeatures feature_right;
    feature_right.img_idx = 1;
    feature_right.img_size = image_sizes[1];

    vector<Point2f> pts_left;
    vector<Point2f> pts_right;
    for(const PointPair& pp : point_pairs) {
        pts_left.push_back(Point2f(pp.points[0].x, pp.points[0].y));
        pts_right.push_back(Point2f(pp.points[1].x, pp.points[1].y));

        feature_left.keypoints.push_back(KeyPoint((float)pp.points[0].x, (float)pp.points[0].y, 5.0));
        feature_right.keypoints.push_back(KeyPoint((float)pp.points[1].x, (float)pp.points[1].y, 5.0));
    }

    MatchesInfo match_info;
    match_info.src_img_idx = 0;
    match_info.dst_img_idx = 1;
    match_info.H = findHomography(pts_right, pts_left, RANSAC, 3, noArray(), 5000);
    match_info.confidence = numeric_limits<double>::max();
    for(uint32_t i=0; i < point_pairs.size(); ++i) {
        DMatch match(i, i, 0.0);
        match_info.matches.push_back(match);
    }

    //std::vector<uchar> inliers_mask;    //!< Geometrically consistent matches mask
    //int num_inliers;                    //!< Number of geometrically consistent matches
    pairwise_matches.push_back(match_info);
}

void finding(const string& features_type, const vector<Mat>& src_images, float overlap_fraction, const vector<PointPair>& point_pairs, vector<CameraParams>& cameras, vector<UMat>& masks_warped) {
float conf_thresh = 0.1f;
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string seam_find_type = "voronoi";

    // Check if have enough images
    int num_images = static_cast<int>(src_images.size());
    if (num_images < 2) {
        spdlog::info("Need more images");
    }

    vector<ImageFeatures> features(num_images);
    vector<MatchesInfo> pairwise_matches;
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);

    spdlog::info("Built rectifying maps");
    for (int i = 0; i < num_images; ++i) {
        images[i] = src_images[i].clone();
        full_img_sizes[i] = images[i].size();
    }

    if(point_pairs.empty())
        featurePairAuto(features_type, images, conf_thresh, overlap_fraction, features, pairwise_matches);
    else
        featurePairManual(point_pairs, full_img_sizes, features, pairwise_matches);
    
    spdlog::info("Build estimators");
    Ptr<Estimator> estimator = makePtr<HomographyBasedEstimator>();

    spdlog::info("Generate estimates");
    if (!(*estimator)(features, pairwise_matches, cameras)) {
        cout << "Homography estimation failed.\n";
    }

    spdlog::info("Adjust cameras");
    for (size_t i = 0; i < cameras.size(); ++i) {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        cout << "Initial camera intrinsics #" << i << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R << endl;
    }

    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
    else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
    else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
    else {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
    }

    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras)) {
        cout << "Camera parameters adjusting failed.\n";
    }

    float warped_image_scale = compute_warped_image_scale(cameras);

    if (do_wave_correct)
    {
        spdlog::info("Wave correcting");
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    spdlog::info("Warping images (auxiliary)... ");

    vector<Mat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<Mat> masks(num_images);
    masks_warped.resize(num_images);

    // Prepare images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks
    Ptr<WarperCreator> warper_creator = makePtr<cv::CylindricalWarper>();
    if (!warper_creator) {
        spdlog::error("Can't create the Cylindrical warper");
    }

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale));
    spdlog::info("Warp aspect: {0:f}", static_cast<float>(warped_image_scale));

    vector<Point> corners(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = 1.0;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }
    spdlog::info("Warped");

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    spdlog::info("Finding seams...");

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = makePtr<detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        seam_finder = makePtr<detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color") {
        seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad") {
        seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
    if (!seam_finder) {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    for(const auto& cp : cameras)
        cout << "Initial camera intrinsics:\nK:\n" << cp.K() << "\nR:\n" << cp.R << endl;
    spdlog::info("GC");
    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();
}
