#include "readers.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <spdlog/spdlog.h>

using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace rapidjson;

void readSeamData(const string& cameras_filename, vector<CameraParams>& cameras, vector<UMat>& masks_warped, Rect& rect) {
    rapidjson::Document stitching_doc;
    ifstream ifs(cameras_filename);
    rapidjson::IStreamWrapper isw(ifs);
    stitching_doc.ParseStream(isw);

    filesystem::path file_path = filesystem::canonical(cameras_filename);
    uint32_t cam_idx = 0;
    Mat tmp;
    for(const auto& camera : stitching_doc["cameras_params"].GetArray()) {
        stringstream ss;
        ss << "warped_seam_mask_" << cam_idx << ".png";
        string image_name = file_path.replace_filename(ss.str());
        cout << image_name << endl;

        tmp = imread(image_name, IMREAD_GRAYSCALE);
        UMat img;
        tmp.copyTo(img);
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

    int x = stitching_doc["crop"]["x"].GetInt();
    int y = stitching_doc["crop"]["y"].GetInt();
    int w = stitching_doc["crop"]["w"].GetInt();
    int h = stitching_doc["crop"]["h"].GetInt();
    rect = Rect(x, y, w, h);
}

void readCalibration(const string& calibration_filename, Mat& K, Mat& distortion_coefficients, Size& calibration_image_size) {
    rapidjson::Document camera_params_doc;
    ifstream ifs(calibration_filename);
    rapidjson::IStreamWrapper isw(ifs);
    camera_params_doc.ParseStream(isw);

    K = Mat::zeros(3, 3, CV_64F);
    distortion_coefficients = Mat::zeros(3, 3, CV_64F);

    for (SizeType i = 0; i < camera_params_doc["K"].Size(); ++i) {
        for (SizeType j=0; j < camera_params_doc["K"][i].Size(); ++j) {
            K.at<double>(i,j) = camera_params_doc["K"][i][j].GetDouble();
        }
    }

    distortion_coefficients = Mat::zeros(1, camera_params_doc["D"].Size(), CV_64F);
    for (SizeType i = 0; i < camera_params_doc["D"].Size(); ++i) {
        distortion_coefficients.at<double>(i) = camera_params_doc["D"][i].GetDouble();
    }

    calibration_image_size.height = camera_params_doc["height"].GetInt();
    calibration_image_size.width = camera_params_doc["width"].GetInt();
}

vector<PointPair> readPointPairs(const string& pointpairs_filename) {
    rapidjson::Document pairs_doc;
    ifstream ifs(pointpairs_filename);
    rapidjson::IStreamWrapper isw(ifs);
    pairs_doc.ParseStream(isw);

    vector<PointPair> pps;
    for(const auto& pair : pairs_doc.GetArray()) {
        PointPair pp;
        pp.locked = true;
        pp.points[0].x = pair["left"]["x"].GetInt();
        pp.points[0].y = pair["left"]["y"].GetInt();
        pp.points[1].x = pair["right"]["x"].GetInt();
        pp.points[1].y = pair["right"]["y"].GetInt();

        pps.push_back(pp);
    }

    return pps;
}

