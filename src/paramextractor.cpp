#include <fstream>
#include <iostream>
#include <vector>
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

#include "framestitcher.hpp"

using namespace std;
using namespace cv;

void write_camera_params(ofstream& writer, const detail::CameraParams camera_params, const string& label) {
    writer << "Mat " << label << "_R(3, 3, CV_32F);" << endl;
    writer << label << "_R.at<float>(0,0) = " << camera_params.R.at<float>(0,0) << ';' << endl;
    writer << label << "_R.at<float>(0,1) = " << camera_params.R.at<float>(0,1) << ';' << endl; 
    writer << label << "_R.at<float>(0,2) = " << camera_params.R.at<float>(0,2) << ';' << endl; 
    writer << label << "_R.at<float>(1,0) = " << camera_params.R.at<float>(1,0) << ';' << endl; 
    writer << label << "_R.at<float>(1,1) = " << camera_params.R.at<float>(1,1) << ';' << endl; 
    writer << label << "_R.at<float>(1,2) = " << camera_params.R.at<float>(1,2) << ';' << endl; 
    writer << label << "_R.at<float>(2,0) = " << camera_params.R.at<float>(2,0) << ';' << endl; 
    writer << label << "_R.at<float>(2,1) = " << camera_params.R.at<float>(2,1) << ';' << endl; 
    writer << label << "_R.at<float>(2,2) = " << camera_params.R.at<float>(2,2) << ';' << endl; 
    writer << "detail::CameraParams camera_" << label << ';' << endl;
    writer << "camera_" << label << ".focal = " << camera_params.focal << ';' << endl;
    writer << "camera_" << label << ".aspect = 1.0;"<< camera_params.aspect << ';' << endl; 
    writer << "camera_" << label << ".ppx = 516.5;" << camera_params.ppx << ';' << endl; 
    writer << "camera_" << label << ".ppy = 290.5;" << camera_params.ppy << ';' << endl; 
    writer << "camera_" << label << ".R = right_R;";
    writer << "camera_" << label << ".t = Mat::zeros(3, 1, CV_64F);" << endl;

    cout << "Is " << label << " t zero: " << camera_params.t << endl;
}


int main(int argc, const char* argv[]) {
    spdlog::set_pattern("%Y%m%dT%H:%M:%S.%e [%^%l%$] -%n- -%t- : %v");
    spdlog::set_level(spdlog::level::debug);

    string left_filename(argv[1]);
    string right_filename(argv[2]);

    spdlog::info("Opening files {} {}", left_filename, right_filename);
    VideoCapture left_video(left_filename);
    VideoCapture right_video(right_filename);

    const uint32_t offset = 10;
    spdlog::info("Skiping first {} frames", offset);
    for(uint32_t i=0; i < offset; ++i) {
        left_video.grab();
        right_video.grab();
    }
    spdlog::info("Reading left and right frame");
    Mat left_image;
    left_video.read(left_image);
    Mat right_image;
    right_video.read(right_image);

    spdlog::info("Writing original images");
    imwrite("left.png", left_image);
    imwrite("right.png", right_image);

    vector<vector<uint32_t>> bgr_value_idxs;
    vector<vector<double>> bgr_cumsum;
    Mat left_yuv;
    cvtColor(left_image, left_yuv, COLOR_BGR2YUV_I420);
    spdlog::info("Building ref image");
    FrameStitcher::BuildReferenceHistogram(left_yuv.data, left_image.cols, left_image.rows, bgr_value_idxs, bgr_cumsum);
    spdlog::info("Matching histo");
    FrameStitcher::MatchHistograms(right_image, bgr_cumsum, bgr_value_idxs);
    spdlog::info("Writing match histo image");
    imwrite("right-corrected.png", right_image);

    spdlog::info("Create stitcher");
    vector<Mat> images;
    images.push_back(left_image);
    images.push_back(right_image);
    Ptr<Stitcher> stitcher(Stitcher::create(Stitcher::PANORAMA));

    spdlog::info("Stitch");
    Mat panoramic_image;
    Stitcher::Status status = stitcher->stitch(images, panoramic_image);

    spdlog::info("Write outputs");
    imwrite("left.png", left_image);
    imwrite("right.png", right_image);
    imwrite("panoramic.png", panoramic_image);

    ofstream stitchparams("stitchcameraparams.txt");
    write_camera_params(stitchparams, stitcher->cameras()[0], "left");
    write_camera_params(stitchparams, stitcher->cameras()[1], "right");
    spdlog::info("Done");
    
    return 0;
}
