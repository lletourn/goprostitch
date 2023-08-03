#include "seamreader.hpp"

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

void readSeamData(const string& cameras_filename, vector<CameraParams>& cameras, vector<UMat>& masks_warped) {
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
