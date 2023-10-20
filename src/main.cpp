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

void process_videos(
    const string& left_filename,
    const string& right_filename,
    const string& output_filename,
    const uint32_t left_video_offset,
    const uint32_t right_video_offset,
    const string& camera_params_filename,
    const string& camera_intrinsics_filename,
    const uint16_t nb_stitch_workers,
    const uint16_t nb_encoding_threads,
    const bool fix_exposure
) {

    Mat camera_intrinsics_K;
    Mat camera_intrinsics_distortion_coefficients;
    Size camera_intrinsics_image_size_used;

    readCalibration(camera_intrinsics_filename, camera_intrinsics_K, camera_intrinsics_distortion_coefficients, camera_intrinsics_image_size_used);

    bool use_left_audio = true;
    if(right_video_offset < left_video_offset) {
        use_left_audio = false;
    }
    
    spdlog::info("Loading Seam data: {}", camera_params_filename);
    vector<detail::CameraParams> cameras_params;
    vector<UMat> masks_warped;
    Rect rect;
    readSeamData(camera_params_filename, cameras_params, masks_warped, rect);
    uint32_t pano_offset_x = rect.x;
    uint32_t pano_offset_y = rect.y;
    uint32_t pano_width = rect.width;
    uint32_t pano_height = rect.height;

    spdlog::info("Loading file: {}", left_filename);
    spdlog::info("Loading file: {}", right_filename);

    const uint32_t input_queue_size = nb_stitch_workers;
    InputProcessor left_processor(left_filename, left_video_offset, input_queue_size);
    left_processor.initialize();
    InputProcessor right_processor(right_filename, right_video_offset, input_queue_size);
    right_processor.initialize();

    InputSyncer input_syncer(left_processor.getOutVideoQueue(), right_processor.getOutVideoQueue());

    spdlog::info("Left duratiaon: {}", left_processor.duration());
    spdlog::info("Right duratiaon: {}", right_processor.duration());
    double duration = left_processor.duration();
    if(duration > right_processor.duration())
        duration = right_processor.duration();

    ThreadSafeQueue<LeftRightPacket> stitcher_queue(input_queue_size);

    spdlog::info("Writting file: {}", output_filename);
    OutputEncoder output_encoder(output_filename, left_processor.getOutAudioQueue(), right_processor.getOutAudioQueue(), pano_width, pano_height, use_left_audio, left_processor.video_time_base(), left_processor.video_frame_rate(), left_processor.audio_time_base(), input_queue_size, nb_encoding_threads);
    output_encoder.initialize(left_processor.audio_codec_parameters(), right_processor.audio_codec_parameters(), duration);

    // Start IO Threads
    output_encoder.start();
    left_processor.start();
    right_processor.start();

    vector<vector<uint32_t>> reference_bgr_value_idxs;
    vector<vector<double>> reference_bgr_cumsum;
//    if(fix_exposure) {
//        spdlog::info("Find Ref frame for white balance");
//        spdlog::info("Fixing exposure");
//        while(true) {
//            unique_ptr<VideoPacket> left_input_packet(left_processor.getOutVideoQueue().pop(chrono::seconds(1)));
//            if(left_input_packet) {
//                FrameStitcher::BuildReferenceHistogram(left_input_packet->data.get(), left_input_packet->width, left_input_packet->height, reference_bgr_value_idxs, reference_bgr_cumsum);
//
//                uint32_t idx = left_input_packet->idx;
//                left_video_packets.insert(make_pair(idx, move(left_input_packet)));
//                break;
//            }
//        }
//    }


    vector<unique_ptr<FrameStitcher>> frame_stitchers;
    for(uint16_t i=0; i < nb_stitch_workers; ++i) {
        unique_ptr<FrameStitcher> fs(new FrameStitcher(pano_offset_x, pano_offset_y, pano_width, pano_height, stitcher_queue, output_encoder.getInPanoramicQueue(), cameras_params, camera_intrinsics_K, camera_intrinsics_distortion_coefficients, camera_intrinsics_image_size_used, masks_warped, reference_bgr_value_idxs, reference_bgr_cumsum));
        fs->start();
        frame_stitchers.push_back(move(fs));
    }

    spdlog::info("Processing all frames");
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    double prev_delta = 0;
    while(true) {
        bool read_packet = false;

        unique_ptr<LeftRightPacket> lr_packet(input_syncer.next_pair());
        if(lr_packet) {
            read_packet = true;
            stitcher_queue.push(move(lr_packet));

            double delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();
            // double fps = ((double)idx_to_process/delta) * 1000.0;

            if((delta - prev_delta) > 60000.0) {
                prev_delta = delta;
                spdlog::info("L: {} R: {} S: {} oP:{}", 
                                left_processor.getOutVideoQueue().estimated_size(),
                                right_processor.getOutVideoQueue().estimated_size(),
                                stitcher_queue.estimated_size(),
                                output_encoder.getInPanoramicQueue().estimated_size());
            }
        } else {
            if(left_processor.is_done() || right_processor.is_done()) {
                input_syncer.set_reader_done();
            }
        }

        if(!read_packet && left_processor.is_done() && right_processor.is_done()) {
            spdlog::info("Inputs done, stopping loop...");
            break;
        }
    }

    // Cleanup
    spdlog::info("Stopping left right processors...");
    left_processor.stop();
    right_processor.stop();

    spdlog::info("Waiting for stitchers to be done...");
    while(stitcher_queue.size() > 0)
        this_thread::sleep_for(chrono::seconds(1));

    spdlog::info("Stopping stitchers...");
    for(auto& frame_stitcher : frame_stitchers) {
        frame_stitcher->stop();
    }

    spdlog::info("Stopping output encoder...");
    output_encoder.stop();
    camera_intrinsics_K.release();
    camera_intrinsics_distortion_coefficients.release();
}

int main(int argc, const char ** argv) {
    cv::setNumThreads(0);
    spdlog::set_pattern("%Y%m%dT%H:%M:%S.%e [%^%l%$] -%n- -%t- : %v");
    spdlog::set_level(spdlog::level::info);

    const String keys =
        "{help h usage ? | | print this message }"
        "{left |<none>| Left video }"
        "{right |<none>| Right video }"
        "{leftoffset | 0 | Left video frame offset.}"
        "{rightoffset | 0 | Right video frame offset.}"
        "{output |<none>| Output video }"
        "{camparams |<none>| Camera parameter filename }"
        "{camintrinsics |<none>| Camera instrinsics filename }"
        "{stitchthreads | 1 | Nb of stitching threads. There are 2 threads by default for video reading. }"
        "{encthreads | 1 | Nb of encoding threads. There are 2 threads by default for video reading. }"
        "{fixexposure | false | Fix whitebalance between cameras }"
    ;
    CommandLineParser parser(argc, argv, keys);
    parser.about("Seam finder");

    if(parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    string left_filename(parser.get<string>("left"));
    string right_filename(parser.get<string>("right"));
    string output_filename(parser.get<string>("output"));
    uint32_t left_video_offset(parser.get<uint32_t>("leftoffset"));
    uint32_t right_video_offset(parser.get<uint32_t>("rightoffset"));
    string camera_params_filename(parser.get<string>("camparams"));
    string camera_intrinsics_filename(parser.get<string>("camintrinsics"));
    uint16_t nb_stitch_workers(parser.get<uint32_t>("stitchthreads"));
    uint16_t nb_encoding_threads(parser.get<uint32_t>("encthreads"));
    bool fix_exposure(parser.get<bool>("fixexposure"));

    process_videos(left_filename, right_filename, output_filename, left_video_offset, right_video_offset, camera_params_filename, camera_intrinsics_filename, nb_stitch_workers, nb_encoding_threads, fix_exposure);
    spdlog::info("Done");
}
