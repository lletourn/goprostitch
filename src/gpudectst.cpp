#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

#include "framestitcher.hpp"
#include "inputprocessor.hpp"
#include "inputsyncer.hpp"
#include "outputencoder.hpp"
#include "readers.hpp"

using namespace std;
using namespace cv;

void process_videos(const bool use_gpu, const string& filename) {
    spdlog::info("Loading file: {}", filename);

    const uint32_t input_queue_size = 1;
    InputProcessor processor(filename, true, 0, input_queue_size);
    processor.initialize();
    processor.start();

    chrono::milliseconds wait_period(15);
    while(true) {
        while(true) {
            unique_ptr<AVPacket, PacketDeleter> audio_packet(processor.getOutAudioQueue().pop(wait_period));
            if(!audio_packet)
                break;
        }
        unique_ptr<VideoPacket> left_packet = processor.getOutVideoQueue().pop(wait_period);

        if (left_packet) {
            // Mat tst;
            // Mat tst_yuv(left_packet->height * 3/2, left_packet->width, CV_8UC1, left_packet->data.get());
            Mat tst(left_packet->height, left_packet->width, CV_8UC3, left_packet->data.get());
            // cvtColor(tst_yuv, tst, COLOR_YUV2BGR_I420);
            // cvtColor(tst_yuv, tst, COLOR_YUV2BGR_NV12);
            namedWindow("Main", WINDOW_NORMAL);
            imshow("Main", tst);
            waitKey();
            destroyAllWindows();
        }
    }
    processor.stop();
}

int main(int argc, const char ** argv) {
    cv::setNumThreads(0);
    spdlog::set_pattern("%Y%m%dT%H:%M:%S.%e [%^%l%$] -%n- -%t- : %v");
    spdlog::set_level(spdlog::level::info);

    const String keys =
        "{help h usage ? | | print this message }"
        "{usegpu | false | Use NVIDIA GPU to encode }"
        "{video |<none>| video }"
    ;
    CommandLineParser parser(argc, argv, keys);
    parser.about("Gpu Decoder Test");

    if(parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    bool use_gpu(parser.get<bool>("usegpu"));
    string filename(parser.get<string>("video"));

    process_videos(use_gpu, filename);
    spdlog::info("Done");
}
