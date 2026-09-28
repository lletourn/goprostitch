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

bool estimateCameras(const string& features_type, const vector<Mat>& images, float overlap_fraction, const vector<PointPair>& point_pairs, vector<CameraParams>& cameras);
void findSeams(const vector<Mat>& images, const vector<CameraParams>& cameras, vector<UMat>& masks_warped);

struct Tilt {
    Mat G;                          // Global rotation applied to every camera: R <- G * R
    Vec3d normal;                   // Plane (through the camera centre) containing the far boards base, pre-tilt world frame
    vector<Point2f> board_points;   // Clicked points, in untilted panorama pixels
    double rms_px;                  // RMS distance of the clicked rays to the fitted plane, in panorama pixels
};

bool pickTilt(const vector<Mat>& images, vector<CameraParams>& cameras, Tilt& tilt);
Point panoramaTopLeft(const vector<Mat>& images, const vector<CameraParams>& cameras);

// Where the far boards base lands in the initial crop box, as a fraction of the box height from its top
constexpr double kCropBoardsRowFraction = 0.15;

Rect cropPano(const Mat& panorama, const Rect& initial_rect);

// Reusing an existing camparams file: only the crop changes, everything else (tilt included) is kept as is
void updateCrop(const string& cameras_params_filename, const Rect& crop_rect) {
    rapidjson::Document stitching_doc;
    {
        ifstream ifs(cameras_params_filename);
        IStreamWrapper isw(ifs);
        stitching_doc.ParseStream(isw);
    }

    Value& pano_crop = stitching_doc["crop"];
    pano_crop["x"].SetInt(crop_rect.x);
    pano_crop["y"].SetInt(crop_rect.y);
    pano_crop["w"].SetInt(crop_rect.width);
    pano_crop["h"].SetInt(crop_rect.height);

    ofstream ofs(cameras_params_filename);
    OStreamWrapper osw(ofs);
    Writer<OStreamWrapper> writer(osw);
    stitching_doc.Accept(writer);
}

void writeStitchingData(const string& cameras_params_filename, const vector<CameraParams>& cameras, const vector<UMat>& masks_warped, const Rect& crop_rect, const Tilt& tilt) {
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

    // Informational only: the tilt is already baked into each camera's R
    Value tilt_doc(kObjectType);
    Value G(kArrayType);
    for(int i = 0; i < 3; ++i) {
        Value G_row(kArrayType);
        for(int j = 0; j < 3; ++j)
            G_row.PushBack(Value(tilt.G.at<double>(i, j)), stitching_doc.GetAllocator());
        G.PushBack(G_row, stitching_doc.GetAllocator());
    }
    tilt_doc.AddMember("G", G, stitching_doc.GetAllocator());
    Value normal(kArrayType);
    for(int i = 0; i < 3; ++i)
        normal.PushBack(Value(tilt.normal[i]), stitching_doc.GetAllocator());
    tilt_doc.AddMember("normal", normal, stitching_doc.GetAllocator());
    Value board_points(kArrayType);
    for(const Point2f& pt : tilt.board_points) {
        Value point(kObjectType);
        point.AddMember("x", Value(pt.x), stitching_doc.GetAllocator());
        point.AddMember("y", Value(pt.y), stitching_doc.GetAllocator());
        board_points.PushBack(point, stitching_doc.GetAllocator());
    }
    tilt_doc.AddMember("board_points", board_points, stitching_doc.GetAllocator());
    tilt_doc.AddMember("rms_px", Value(tilt.rms_px), stitching_doc.GetAllocator());
    stitching_doc.AddMember("tilt", tilt_doc, stitching_doc.GetAllocator());

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

    vector<Mat> images = {imread(parser.get<string>("left")), imread(parser.get<string>("right"))};
    vector<Size> images_size = {images[0].size(), images[1].size()};
    string result_name = parser.get<string>("output");

    Size crop_size;
    char crop_sep = 0;
    istringstream crop_ss(parser.get<string>("cropsize"));
    if(!(crop_ss >> crop_size.width >> crop_sep >> crop_size.height) || (crop_sep != 'x' && crop_sep != 'X') || crop_size.width <= 0 || crop_size.height <= 0) {
        spdlog::error("Invalid cropsize '{}', expected WxH", parser.get<string>("cropsize"));
        return 1;
    }

    const bool find_cam_params = parser.get<bool>("findcamparams");
    vector<CameraParams> cameras;
    vector<UMat> masks_warped;
    Tilt tilt;
    Rect crop_rect(0, 0, crop_size.width, crop_size.height);

    if(!find_cam_params) {
        spdlog::info("From cameras params");
        Rect saved_rect;
        readSeamData(parser.get<string>("camparams"), cameras, masks_warped, saved_rect);
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
        if(!estimateCameras(features_type, images, overlap_fraction, point_pairs, cameras))
            return 1;
        if(!pickTilt(images, cameras, tilt))
            return 1;
        findSeams(images, cameras, masks_warped);
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
    if(find_cam_params) {
        // The far boards base now sits on the cylinder's equator (v = 0)
        const int boards_row = -panoramaTopLeft(images, cameras).y;
        crop_rect.x = (output_image.cols - crop_size.width) / 2;
        crop_rect.y = boards_row - cvRound(kCropBoardsRowFraction * crop_size.height);
    }

    Rect rect = cropPano(output_image, crop_rect);
    if(find_cam_params)
        writeStitchingData(parser.get<string>("camparams"), cameras, masks_warped, rect, tilt);
    else
        updateCrop(parser.get<string>("camparams"), rect);
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

bool estimateCameras(const string& features_type, const vector<Mat>& images, float overlap_fraction, const vector<PointPair>& point_pairs, vector<CameraParams>& cameras) {
float conf_thresh = 0.1f;
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;

    // Check if have enough images
    int num_images = static_cast<int>(images.size());
    if (num_images < 2) {
        spdlog::error("Need more images");
        return false;
    }

    vector<ImageFeatures> features(num_images);
    vector<MatchesInfo> pairwise_matches;
    vector<Size> full_img_sizes(num_images);
    for (int i = 0; i < num_images; ++i)
        full_img_sizes[i] = images[i].size();

    if(point_pairs.empty())
        featurePairAuto(features_type, images, conf_thresh, overlap_fraction, features, pairwise_matches);
    else
        featurePairManual(point_pairs, full_img_sizes, features, pairwise_matches);
    
    spdlog::info("Build estimators");
    Ptr<Estimator> estimator = makePtr<HomographyBasedEstimator>();

    spdlog::info("Generate estimates");
    if (!(*estimator)(features, pairwise_matches, cameras)) {
        spdlog::error("Homography estimation failed");
        return false;
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
        spdlog::error("Camera parameters adjusting failed");
        return false;
    }

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

    return true;
}

void findSeams(const vector<Mat>& images, const vector<CameraParams>& cameras, vector<UMat>& masks_warped) {
string seam_find_type = "voronoi";

    int num_images = static_cast<int>(images.size());
    float warped_image_scale = compute_warped_image_scale(cameras);

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
}

// Top-left of the cylindrical panorama in warped (u, v) coordinates, same as ImageCompositing's output origin
Point panoramaTopLeft(const vector<Mat>& images, const vector<CameraParams>& cameras) {
    Ptr<RotationWarper> warper = makePtr<cv::CylindricalWarper>()->create(compute_warped_image_scale(cameras));
    Point top_left(numeric_limits<int>::max(), numeric_limits<int>::max());
    for(size_t i = 0; i < images.size(); ++i) {
        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        Rect roi = warper->warpRoi(images[i].size(), K, cameras[i].R);
        top_left.x = std::min(top_left.x, roi.x);
        top_left.y = std::min(top_left.y, roi.y);
    }
    return top_left;
}

// Quick unblended panorama, only used to pick the far boards and check the tilt
Mat renderPreview(const vector<Mat>& images, const vector<CameraParams>& cameras, Point& top_left) {
    Ptr<RotationWarper> warper = makePtr<cv::CylindricalWarper>()->create(compute_warped_image_scale(cameras));
    vector<Mat> warped(images.size());
    vector<Mat> warped_masks(images.size());
    vector<Point> corners(images.size());
    for(size_t i = 0; i < images.size(); ++i) {
        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_CONSTANT, warped[i]);
        Mat mask(images[i].size(), CV_8U, Scalar(255));
        warper->warp(mask, K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, warped_masks[i]);
    }

    vector<Size> sizes(images.size());
    for(size_t i = 0; i < images.size(); ++i)
        sizes[i] = warped[i].size();
    Rect pano_roi = resultRoi(corners, sizes);
    top_left = pano_roi.tl();

    Mat pano(pano_roi.size(), CV_8UC3, Scalar::all(0));
    for(size_t i = 0; i < images.size(); ++i)
        warped[i].copyTo(pano(Rect(corners[i] - top_left, sizes[i])), warped_masks[i]);
    return pano;
}

// Viewing ray (pre-tilt world frame) of a panorama pixel: inverse of the cylindrical projection
Vec3d panoPixelToRay(const Point2f& pt, const Point& top_left, float scale) {
    const double theta = (pt.x + top_left.x) / scale;
    const double h = (pt.y + top_left.y) / scale;
    return normalize(Vec3d(sin(theta), h, cos(theta)));
}

// Plane through the camera centre that best contains the rays: normal is the smallest eigenvector of sum(d d^T)
void fitBoardsPlane(const vector<Point2f>& points, const Point& top_left, float scale, Vec3d& normal, double& rms_px) {
    Matx33d scatter = Matx33d::zeros();
    vector<Vec3d> rays;
    for(const Point2f& pt : points) {
        Vec3d d = panoPixelToRay(pt, top_left, scale);
        scatter += d * d.t();
        rays.push_back(d);
    }

    Mat eigenvalues, eigenvectors;
    eigen(Mat(scatter), eigenvalues, eigenvectors);
    normal = Vec3d(eigenvectors.at<double>(2, 0), eigenvectors.at<double>(2, 1), eigenvectors.at<double>(2, 2));
    // Keep the cylinder axis pointing the same way as the current one, otherwise the panorama flips upside down
    if(normal[1] < 0)
        normal = -normal;

    double sum_sq = 0.0;
    for(const Vec3d& d : rays) {
        const double err_px = asin(std::min(1.0, std::abs(normal.dot(d)))) * scale;
        sum_sq += err_px * err_px;
    }
    rms_px = sqrt(sum_sq / rays.size());
}

// Rotation that maps the boards plane normal onto the cylinder axis (y). The spin around that axis is free:
// pick it so the bisector of the cameras' viewing directions lands at theta = 0, keeping the panorama centred.
Mat buildTilt(const Vec3d& normal, const vector<CameraParams>& cameras) {
    Vec3d forward(0, 0, 0);
    for(const CameraParams& camera : cameras) {
        Mat R;
        camera.R.convertTo(R, CV_64F);
        forward += normalize(Vec3d(R.at<double>(0, 2), R.at<double>(1, 2), R.at<double>(2, 2)));
    }
    Vec3d z = normalize(forward - forward.dot(normal) * normal);
    Vec3d x = normal.cross(z);

    Mat G(3, 3, CV_64F);
    for(int j = 0; j < 3; ++j) {
        G.at<double>(0, j) = x[j];
        G.at<double>(1, j) = normal[j];
        G.at<double>(2, j) = z[j];
    }
    return G;
}

struct BoardPicker {
    vector<Point2f> points;
    bool changed = true;
};

void onBoardClick(int event, int x, int y, int, void* userdata) {
    if(event != EVENT_LBUTTONDOWN)
        return;
    BoardPicker* picker = static_cast<BoardPicker*>(userdata);
    picker->points.emplace_back(static_cast<float>(x), static_cast<float>(y));
    picker->changed = true;
}

bool isEnterKey(int key) {
    return key == 13 || key == 10 || key == 65293 || key == 65421;
}

void drawBanner(Mat& image, const string& text) {
    const double font_scale = image.cols / 2000.0;
    const int thickness = std::max(1, cvRound(font_scale * 2));
    putText(image, text, Point(20, cvRound(60 * font_scale)), FONT_HERSHEY_SIMPLEX, font_scale, Scalar(0, 0, 0), thickness * 4);
    putText(image, text, Point(20, cvRound(60 * font_scale)), FONT_HERSHEY_SIMPLEX, font_scale, Scalar(255, 255, 255), thickness);
}

// Clicks on the untilted preview; returns false if aborted with ESC
bool clickBoardPoints(const Mat& preview, const Point& top_left, float scale, vector<Point2f>& points, Vec3d& normal, double& rms_px) {
    BoardPicker picker;
    picker.points = points;
    namedWindow("Tilt", WINDOW_NORMAL);
    resizeWindow("Tilt", 1920, 720);
    setMouseCallback("Tilt", onBoardClick, &picker);

    const int radius = std::max(4, preview.cols / 600);
    while(true) {
        if(picker.changed) {
            picker.changed = false;
            Mat display = preview.clone();
            string status = "Click the BASE of the far boards, in any order. Backspace: undo last, Enter: compute (min 3), Esc: abort.";

            if(picker.points.size() >= 2) {
                fitBoardsPlane(picker.points, top_left, scale, normal, rms_px);
                // Projection of the fitted plane on the current cylinder
                vector<Point> curve;
                if(std::abs(normal[1]) > 1e-9) {
                    for(int px = 0; px < display.cols; px += 4) {
                        const double theta = (px + top_left.x) / scale;
                        const double h = -(normal[0] * sin(theta) + normal[2] * cos(theta)) / normal[1];
                        curve.emplace_back(px, cvRound(h * scale - top_left.y));
                    }
                }
                // Semi-transparent so the boards stay visible under the curve
                Mat overlay = display.clone();
                polylines(overlay, curve, false, Scalar(0, 255, 255), std::max(2, radius / 2));
                addWeighted(overlay, 0.4, display, 0.6, 0.0, display);
                status += cv::format("  Points: %zu  RMS: %.1f px", picker.points.size(), rms_px);
                spdlog::info("Boards fit: {} points, RMS {:.2f} px", picker.points.size(), rms_px);
            }
            for(const Point2f& pt : picker.points)
                circle(display, pt, radius, Scalar(0, 0, 255), FILLED);

            drawBanner(display, status);
            imshow("Tilt", display);
        }

        int key = waitKeyEx(30);
        if(key == -1 || key == 65535)
            continue;
        if(key == 27) {
            destroyWindow("Tilt");
            return false;
        } else if(key == 8 || key == 65288) { // Backspace
            if(!picker.points.empty()) {
                picker.points.pop_back();
                picker.changed = true;
            }
        } else if(isEnterKey(key)) {
            if(picker.points.size() >= 3)
                break;
            spdlog::warn("Need at least 3 points on the far boards, have {}", picker.points.size());
        }
    }

    setMouseCallback("Tilt", nullptr);
    // The plane fit doesn't depend on order; sort so the saved points read left to right
    points = picker.points;
    std::sort(points.begin(), points.end(), [](const Point2f& a, const Point2f& b) { return a.x < b.x; });
    return true;
}

bool pickTilt(const vector<Mat>& images, vector<CameraParams>& cameras, Tilt& tilt) {
    const float scale = compute_warped_image_scale(cameras);
    vector<Mat> untilted_R;
    for(const CameraParams& camera : cameras)
        untilted_R.push_back(camera.R.clone());

    Point untilted_top_left;
    Mat untilted_preview = renderPreview(images, cameras, untilted_top_left);

    vector<Point2f> points;
    while(true) {
        Vec3d normal;
        double rms_px = 0.0;
        if(!clickBoardPoints(untilted_preview, untilted_top_left, scale, points, normal, rms_px))
            return false;

        Mat G = buildTilt(normal, cameras);
        for(size_t i = 0; i < cameras.size(); ++i) {
            Mat R;
            untilted_R[i].convertTo(R, CV_64F);
            Mat tilted = G * R;
            tilted.convertTo(cameras[i].R, CV_32F);
        }
        spdlog::info("Tilt normal: [{:.5f}, {:.5f}, {:.5f}], RMS {:.2f} px", normal[0], normal[1], normal[2], rms_px);

        Point tilted_top_left;
        Mat tilted_preview = renderPreview(images, cameras, tilted_top_left);
        const int boards_row = -tilted_top_left.y;
        line(tilted_preview, Point(0, boards_row), Point(tilted_preview.cols - 1, boards_row), Scalar(0, 255, 255), std::max(2, tilted_preview.cols / 2000));
        drawBanner(tilted_preview, "Far boards base should follow the line. Enter: accept, R: re-pick, Esc: abort.");
        imshow("Tilt", tilted_preview);

        int key;
        do {
            key = waitKeyEx(30);
        } while(!isEnterKey(key) && key != 27 && key != 'r' && key != 'R');

        if(isEnterKey(key)) {
            destroyWindow("Tilt");
            tilt.G = G;
            tilt.normal = normal;
            tilt.board_points = points;
            tilt.rms_px = rms_px;
            return true;
        }

        for(size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = untilted_R[i].clone();
        if(key == 27) {
            destroyWindow("Tilt");
            return false;
        }
    }
}
