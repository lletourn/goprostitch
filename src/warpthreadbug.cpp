#include <atomic>
#include <iostream>
#include <limits>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

atomic<int> atomic_count{0};

void work(const int nb_images, const string image_filename, const double focal, const Mat& K, const Mat& R) {
    Mat img_warped(4000,4000, CV_8UC3);
    Ptr<WarperCreator> warper_creator = makePtr<cv::CylindricalWarperGpu>();  // Can't use GPU causes weird warping artifacts
    Ptr<detail::RotationWarper> warper = warper_creator->create(focal);

    Mat img;
    imread(image_filename, img);

    for(int i=0; i < nb_images; ++i) {
        warper->warp(img, K, R, INTER_LINEAR, BORDER_REFLECT, img_warped);
    // namedWindow("src", WINDOW_NORMAL);
    // imshow("src", img);
    // namedWindow("dst", WINDOW_NORMAL);
    // imshow("dst", img_warped);
    // waitKey();
    // destroyAllWindows();

        int count = ++atomic_count;

        stringstream warped_img;
        warped_img << "warped_" << std::setfill('0') << std::setw(6) << count << ".jpg";

        imwrite(warped_img.str().c_str(), img_warped);
    }
}

int main(int argc, const char ** argv) {
    cv::setNumThreads(0);

    const String keys =
        "{help h usage ? | | print this message }"
        "{usegpu | false | Use NVIDIA GPU to encode }"
        "{workers | 2 | Nb of worker threads }" 
        "{image1 |<none>| Image1 }"
        "{image2 |<none>| Image2 }"
    ;
    CommandLineParser parser(argc, argv, keys);
    parser.about("Seam finder");

    int nb_workers(parser.get<int>("workers"));
    if(parser.has("help") || nb_workers < 2) {
        parser.printMessage();
        return 0;
    }
    bool use_gpu(parser.get<bool>("usegpu"));
    // string mask_filename(parser.get<string>("mask"));
    string image_filename1(parser.get<string>("image1"));
    string image_filename2(parser.get<string>("image2"));

    float K[3][3] = {{ 1.7688181032986872e+03, 0.0, 1.9237091208494583e+03}, {0.0, 1.7679751216421994e+03, 1.0957257960681286e+03}, {0.0, 0.0, 1.0}};
    Mat camera_intrinsics_K = Mat(3, 3, CV_32F, K);
    double D[5] = {-2.3918513751023274e-01, 6.9352326700516775e-02, -5.8377437741893799e-05, 1.9806067043683098e-04, -1.0478418302033846e-02};
    Mat camera_intrinsics_distortion_coefficients(1, 5, CV_64F, D);
    Size camera_intrinsics_image_size_used(3840, 2160);

    float r1[3][3] = {{0.8138929605484009,0.024791114032268525,-0.5804857015609741},{-7.205906427110165e-10,0.9990893006324768,0.04266864061355591},{0.58101487159729,-0.034727707505226138,0.8131517767906189}}; 
    float r2[3][3] = {{0.8142296075820923,-0.00815536454319954,0.5804857611656189},{9.267237288668184e-8,0.999901294708252,0.014047694392502308},{-0.5805431008338928,-0.011437991634011269,0.8141492009162903}};
    Mat R1(3, 3, CV_32F, r1);
    Mat R2(3, 3, CV_32F, r2);
    double focal = 1719.23;

    vector<thread> work_threads;
    for(int i=0; i< nb_workers; ++i) {
        string f = image_filename1;
        if(i%2 == 0)
            work_threads.push_back(move(thread(work, 250, image_filename1, focal, camera_intrinsics_K, R1)));
        else
            work_threads.push_back(move(thread(work, 250, image_filename2, focal, camera_intrinsics_K, R2)));
    }

    for(thread& t : work_threads) {
        t.join();
    }
}
