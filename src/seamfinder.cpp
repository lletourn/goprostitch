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


Rect compositing(const vector<string>& img_names, const string& result_name, const vector<CameraParams>& cameras, const vector<UMat>& masks_warped);

void finding(const vector<string>& img_names, const vector<PointPair>& point_pairs, vector<CameraParams>& cameras, vector<UMat>& masks_warped);

void writeStitchingData(const string& cameras_params_filename, const vector<CameraParams>& cameras, const vector<UMat>& masks_warped, const Rect& crop_rect) {
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

    ofstream ofs(cameras_params_filename);
    OStreamWrapper osw(ofs);
 
    Writer<OStreamWrapper> writer(osw);
    stitching_doc.Accept(writer);
}

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


int main(int argc, char* argv[]) {
    spdlog::set_pattern("%Y%m%dT%H:%M:%S.%e [%^%l%$] -%n- -%t- : %v");
    spdlog::set_level(spdlog::level::debug);

    cv::setBreakOnError(true);
    cv::ocl::setUseOpenCL(false);

    const String keys =
        "{help h usage ? | | print this message }"
        "{left |<none>| Left image }"
        "{right |<none>| Right image }"
        "{keypoints | | Left-Right keypoints json filename}"
        "{output |<none>| Output panorama }"
        "{camparams | | Camera parameter filename }"
        "{findcamparams | true | Generate camera parameters or read them from the file }"
    ;

    CommandLineParser parser(argc, argv, keys);
    parser.about("Seam finder");

    if(parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    vector<string> img_names;
    img_names.push_back(parser.get<string>("left"));
    img_names.push_back(parser.get<string>("right"));
    string result_name = parser.get<string>("output");;

    int num_images = static_cast<int>(img_names.size());

    vector<CameraParams> cameras;
    vector<UMat> masks_warped;

    if(!parser.get<bool>("findcamparams")) {
        spdlog::info("From cameras params");
        Rect rect;
        readSeamData(parser.get<string>("camparams"), cameras, masks_warped, rect);
    } else {
        spdlog::info("Generate canera params");
        vector<PointPair> point_pairs;
        if(parser.has("keypoints"))
            point_pairs = readPointPairs(parser.get<string>("keypoints"));
        finding(img_names, point_pairs, cameras, masks_warped);
    }

    Rect rect = compositing(img_names, result_name, cameras, masks_warped);
    writeStitchingData(parser.get<string>("camparams"), cameras, masks_warped, rect);
}

Rect cropPano(const Mat& panorama) {
    int x1 = 2;
    int y1 = 228;
    int x2 = 4654;
    int y2 = 2148;

    namedWindow("Pano", WINDOW_NORMAL);

    cout << "Rect: " << Rect(x1, y1, x2-x1, y2-y1) << endl;
    Mat tmp = panorama.clone();
    rectangle(tmp, Rect(x1, y1, x2-x1, y2-y1), Scalar(0,255,0), 5);
    imshow("Pano", tmp);
    char ver = 't';
    char hor = 'l';
    while(true) {
        int key = waitKeyEx(30);

        // modifier keys are flags starting at 0x10000
        bool redraw = false;
        bool shiftPressed = key & 1 << 16;
        bool ctrlPressed = key & 1 << 18;

        //key = key & 0xffff;
        // Arrows == values > 60k...so for keybord, use <128 from the ASCII table
        if (key != -1 && key != 65535) {
            cout << "Key: " << key << endl;
            if (key == 13 || key == 27 || key == 10) { // CR, ESC, or LF
                break;
            } else if(key == 't' || key == 'T') {
                ver = 't';
                cout << "Ver: " << ver << endl;
            } else if(key == 'b' || key == 'B') {
                ver = 'b';
                cout << "Ver: " << ver << endl;
            } else if(key == 'l' || key == 'L') {
                hor = 'l';
                cout << "Hor: " << hor << endl;
            } else if(key == 'r' || key == 'R') {
                hor = 'r';
                cout << "Hor: " << hor << endl;
            } else if(key == 65361) { // LEFT
                cout << " Left" << endl;
                cout << "Hor: " << hor << endl;
                if(hor == 'l')
                    x1--;
                else
                    x2--;

                if(x1 < 0)
                    x1 = 0;
                if(x2 < 0)
                    x2 = x1;
                if(x2 <= x1)
                    x2 = x1+1;
            } else if(key == 65362) {
                cout << "UP" << endl;
                if(ver == 't')
                    y1--;
                else
                    y2--;

                if(y1 < 0)
                    y1 = 0;
                if(y2 < 0)
                    y2 = y1;
                if(y2 <= y1)
                    y2 = y1+1;
            } else if(key == 65363) { // Right
                cout << " Right" << endl;
                cout << "Hor: " << hor << endl;
                if(hor == 'l')
                    x1++;
                else
                    x2++;

                if(x2 >= panorama.size().width)
                    x2 = panorama.size().width-1;
                if(x1 >= panorama.size().width)
                    x1 = x2;
                if(x1 >= x2)
                    x1 = x2-1;
            } else if(key == 65364) {
                cout << "Down" << endl;
                if(ver == 't')
                    y1++;
                else
                    y2++;

                if(y2 >= panorama.size().height)
                    y2 = panorama.size().height-1;
                if(y1 >= panorama.size().height)
                    y1 = y2;
                if(y1 >= y2)
                    y1 = y2-1;
            }
            redraw = true;
        }
        if(redraw) {
            Rect rect(x1, y1, x2-x1, y2-y1);
            cout << "Rect: " << rect << endl;
            Mat tmp = panorama.clone();
            rectangle(tmp, rect, Scalar(0,255,0), 5);
            imshow("Pano", tmp);
        }
    }
    destroyAllWindows();

    return Rect(x1, y1, x2-x1, y2-y1);
}

Rect compositing(const vector<string>& img_names, const string& result_name, const vector<CameraParams>& cameras, const vector<UMat>& masks_warped) {
    int num_images = static_cast<int>(img_names.size());
    vector<Mat> full_imgs(num_images);
    vector<Size> sizes(num_images);
    full_imgs[0] = imread(samples::findFile(img_names[0]));
    full_imgs[1] = imread(samples::findFile(img_names[1]));
    
    Mat img;
    float warped_image_scale = compute_warped_image_scale(cameras);
    Ptr<WarperCreator> warper_creator = makePtr<cv::CylindricalWarper>();
    if (!warper_creator) {
        spdlog::error("Can't create the Cylindrical warper");
    }

    Ptr<RotationWarper> warper = warper_creator->create(warped_image_scale);
    spdlog::info("Compositing...");
int blend_type = Blender::NO;
float blend_strength = 5;

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;
    bool is_compose_scale_set = false;
    double compose_scale = 1;

    spdlog::info("Update corners");
    vector<Point> corners(num_images);
    for (int i = 0; i < num_images; ++i) {
        // Update corner and size
        Size sz = full_imgs[i].size();

        Mat K;
        cameras[i].K().convertTo(K, CV_32F);
        Rect roi = warper->warpRoi(sz, K, cameras[i].R);
        corners[i] = roi.tl();
        sizes[i] = roi.size();
        cout << "Size: " << sizes[i] << endl;
    }
    for(auto& corner : corners)
        cout << "Corner: " << corner << endl;

    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        stringstream ss;
        ss << "Compositing image #" <<img_idx+1;
        spdlog::info(ss.str()); ss.str(string()); ss.clear();

        // Read image and resize it if necessary
        ss << "Reading: [" << img_idx << "] " << img_names[img_idx];
        spdlog::info(ss.str()); ss.str(string()); ss.clear();
        img = full_imgs[img_idx];
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        spdlog::info("Warp image");
        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        spdlog::info("Create mask");
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        spdlog::info("Create seam mask");
        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        mask_warped = seam_mask & mask_warped;

        if (!blender) {
            blender = Blender::createDefault(blend_type, false);
            Rect roi = resultRoi(corners, sizes);
            Size dst_sz = roi.size();
            cout << "ROI: " << roi << " Sz: " << dst_sz << endl;
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, false);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                stringstream ssb("Multi-band blender, number of bands: ");
                ssb << mb->numBands();
                spdlog::info(ssb.str());
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
                fb->setSharpness(1.f/blend_width);
                stringstream ssb("Feather blender, sharpness: ");
                ssb << fb->sharpness();
                spdlog::info(ssb.str());
            }
            blender->prepare(corners, sizes);
        }

        //imshow("0 iwar", img_warped_s);
        //imshow("0 mwar", mask_warped);
        //waitKey(0);
        cout << "Corner Idx: " << img_idx << " [" << corners[img_idx].x << "," << corners[img_idx].y << endl;
        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
    }

    Mat result, result_mask;
    blender->blend(result, result_mask);

    Mat m;
    result.convertTo(m, CV_8UC3);
    cout << "Res type: " << type2str(m.type()) << endl;
    imwrite(result_name, m);

    return cropPano(m);
}


void featurePairAuto(const vector<Mat>& images, float conf_thresh, vector<ImageFeatures>& features, vector<MatchesInfo>& pairwise_matches) {
    string features_type = "surf";
    float match_conf = 0.65f;

    spdlog::info("Finding features...");
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
        computeImageFeatures(finder, images[i], features[i]);
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

void finding(const vector<string>& img_names, const vector<PointPair>& point_pairs, vector<CameraParams>& cameras, vector<UMat>& masks_warped) {
float conf_thresh = 0.1f;
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string seam_find_type = "gc_color";

    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2) {
        spdlog::info("Need more images");
    }

    Mat full_img;
    vector<ImageFeatures> features(num_images);
    vector<MatchesInfo> pairwise_matches;
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);

    spdlog::info("Built rectifying maps");
    for (int i = 0; i < num_images; ++i) {
        full_img = imread(samples::findFile(img_names[i]));
        spdlog::info("Read image");
        full_img_sizes[i] = full_img.size();

        images[i] = full_img.clone();
    }
    full_img.release();
    if(point_pairs.empty())
        featurePairAuto(images, conf_thresh, features, pairwise_matches);
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
    spdlog::info("Warp askpect: {0:f}", static_cast<float>(warped_image_scale));

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

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    spdlog::info("Finding seams...");

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = makePtr<detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        seam_finder = makePtr<detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color")
    {
        seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
        seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
    if (!seam_finder)
    {
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
