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

void compositing(const vector<string>& img_names, const string& result_name, const vector<CameraParams>& cameras, const vector<UMat>& masks_warped);

void finding(const vector<string>& img_names, vector<CameraParams>& cameras, vector<UMat>& masks_warped);

void writeStitchingData(const vector<CameraParams>& cameras, const vector<UMat>& masks_warped) {
    rapidjson::Document cameras_doc(kArrayType);

    for(uint32_t cam_idx=0; cam_idx < cameras.size(); ++cam_idx){
        stringstream ss;
        ss << "warped_seam_mask_" << cam_idx << ".png";
        imwrite(ss.str(), masks_warped[cam_idx]);
        spdlog::info("Mask type: {}", type2str(masks_warped[cam_idx].type()));

        const CameraParams& camera_params = cameras[cam_idx];
        Value camera(kObjectType);
        camera.AddMember("aspect", Value(camera_params.aspect), cameras_doc.GetAllocator());
        camera.AddMember("focal", Value(camera_params.focal), cameras_doc.GetAllocator());
        camera.AddMember("ppx", Value(camera_params.ppx), cameras_doc.GetAllocator());
        camera.AddMember("ppy", Value(camera_params.ppy), cameras_doc.GetAllocator());


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
                R_row.PushBack(Value(camera_params.R.at<float>(i,j)), cameras_doc.GetAllocator());
            }
            R.PushBack(R_row, cameras_doc.GetAllocator());
        }
        camera.AddMember("R", R, cameras_doc.GetAllocator());

        if(camera_params.t.depth() == CV_32F)
            spdlog::error("t is 32bit float");
        else if(camera_params.t.depth() == CV_64F)
            spdlog::error("t is 64bit double");
        else
            spdlog::error("t type is unknown");
        Value t(kArrayType);
        for(uint32_t i = 0; i < camera_params.t.rows; ++i) {
            t.PushBack(Value(camera_params.t.at<double>(i)), cameras_doc.GetAllocator());
        }
        camera.AddMember("t", t, cameras_doc.GetAllocator());

        cameras_doc.PushBack(camera, cameras_doc.GetAllocator());
    }
    ofstream ofs("cameras.json");
    OStreamWrapper osw(ofs);
 
    Writer<OStreamWrapper> writer(osw);
    cameras_doc.Accept(writer);
}

void readStitchingData(const string& cameras_filename, vector<CameraParams>& cameras, vector<UMat>& masks_warped) {
    rapidjson::Document cameras_doc;
    ifstream ifs(cameras_filename);
    rapidjson::IStreamWrapper isw(ifs);
    cameras_doc.ParseStream(isw);

    filesystem::path file_path = filesystem::canonical(cameras_filename);
    uint32_t cam_idx = 0;
    Mat tmp;
    for(const auto& camera : cameras_doc.GetArray()) {
        stringstream ss;
        ss << "warped_seam_mask_" << cam_idx << ".png";
        string image_name = file_path.replace_filename(ss.str());
        cout << image_name << endl;

        tmp = imread(image_name, IMREAD_GRAYSCALE);
        UMat img;
        tmp.copyTo(img);
        spdlog::info("Mask type: {}", type2str(img.type()));
        masks_warped.push_back(img);

        CameraParams cp;
        cp.R = Mat::eye(3, 3, CV_32F); // Seems it needs to be floats not doubles
        cp.aspect = camera["aspect"].GetDouble();
        cp.focal = camera["focal"].GetDouble();
        cp.ppx = camera["ppx"].GetDouble();
        cp.ppy = camera["ppy"].GetDouble();

        const Value& R = camera["R"];
        for(uint32_t i=0; i < R.Size(); ++i) {
            for(uint32_t j=0; j < R[i].Size(); ++j) {
                cp.R.at<float>(i, j) = R[i][j].GetFloat();
            }
        }

        const Value& t = camera["t"];
        for(uint32_t i=0; i < t.Size(); ++i) {
            cp.t.at<double>(i) = t[i].GetDouble();
        }

        cameras.push_back(cp);
        cout << "Initial camera intrinsics #" << cam_idx+1 << ":\nK:\n" << cp.K() << "\nR:\n" << cp.R << endl;
        cam_idx++;
    }
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

    vector<string> img_names;
    img_names.push_back(argv[1]);
    img_names.push_back(argv[2]);
    string result_name = argv[3];

    int num_images = static_cast<int>(img_names.size());

    vector<CameraParams> cameras;
    vector<UMat> masks_warped;

    if(argc > 4) {
        readStitchingData(argv[4], cameras, masks_warped);
    } else {
        finding(img_names, cameras, masks_warped);
        writeStitchingData(cameras, masks_warped);
    }

    compositing(img_names, result_name, cameras, masks_warped);
}

void compositing(const vector<string>& img_names, const string& result_name, const vector<CameraParams>& cameras, const vector<UMat>& masks_warped) {
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
        img = imread(samples::findFile(img_names[img_idx]));
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

    imwrite(result_name, result);
}


void finding(const vector<string>& img_names, vector<CameraParams>& cameras, vector<UMat>& masks_warped) {
// Default command line args
double work_megapix = -1;
double seam_megapix = -1;
double compose_megapix = -1;
float conf_thresh = 0.1f;
string features_type = "sift";
float match_conf = 0.65f;
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string seam_find_type = "gc_color";
int range_width = -1;

    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2) {
        spdlog::info("Need more images");
    }

    double work_scale = 1, seam_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false;

    spdlog::info("Finding features...");

    Ptr<Feature2D> finder;
    if (features_type == "orb")
    {
        finder = ORB::create();
    }
    else if (features_type == "akaze")
    {
        finder = AKAZE::create();
    }
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf")
    {
        finder = xfeatures2d::SURF::create();
    }
#endif
    else if (features_type == "sift")
    {
        finder = SIFT::create();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
    }

    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(samples::findFile(img_names[i]));
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            spdlog::info("Can't open image ");
            spdlog::info(img_names[i]);
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }
        cout << "SS: " << seam_scale << " WS: " << work_scale << " SWA: " << seam_work_aspect << endl;

        computeImageFeatures(finder, img, features[i]);
        features[i].img_idx = i;
        spdlog::info("Features in image #");
        spdlog::info(i+1);
        spdlog::info(": ");
        spdlog::info(features[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();

    spdlog::info("Pairwise matching");
    vector<MatchesInfo> pairwise_matches;
    Ptr<FeaturesMatcher> matcher;
    if (matcher_type == "affine")
        matcher = makePtr<AffineBestOf2NearestMatcher>(false, false, match_conf);
    else if (range_width==-1)
        matcher = makePtr<BestOf2NearestMatcher>(false, match_conf);
    else
        matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, false, match_conf);

    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();

    spdlog::info("Find matching images");
    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<String> img_names_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    full_img_sizes = full_img_sizes_subset;

    // Check if we still have enough images
    num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        spdlog::info("Need more images: ");
        spdlog::info(num_images);
    }

    Ptr<Estimator> estimator;
    if (estimator_type == "affine")
        estimator = makePtr<AffineBasedEstimator>();
    else
        estimator = makePtr<HomographyBasedEstimator>();

    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        cout << "Homography estimation failed.\n";
    }

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        cout << "Initial camera intrinsics #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R << endl;
    }

    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
    else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
    else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
    else
    {
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
    if (!(*adjuster)(features, pairwise_matches, cameras))
    {
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

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
    spdlog::info("Warp askpect: {0:f}", static_cast<float>(warped_image_scale * seam_work_aspect));

    vector<Point> corners(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
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
