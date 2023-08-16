#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

#include "framestitcher.hpp"
#include "inputprocessor.hpp"
#include "outputencoder.hpp"
#include "readers.hpp"

using namespace std;
using namespace cv;


int main(int argc, const char ** argv) {
    spdlog::set_pattern("%Y%m%dT%H:%M:%S.%e [%^%l%$] -%n- -%t- : %v");
    spdlog::set_level(spdlog::level::debug);

    const String keys =
        "{help h usage ? | | print this message }"
        "{left |<none>| Left video }"
        "{right |<none>| Right video }"
        "{offset | 0 | Video frame offset. Positive offsets shifts right and takes left audio. Negative is the opposite }"
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
    int32_t video_offset(parser.get<int32_t>("offset"));
    string camera_params_filename(parser.get<string>("camparams"));
    string camera_intrinsics_filename(parser.get<string>("camintrinsics"));
    uint16_t nb_stitch_workers(parser.get<uint32_t>("stitchthreads"));
    uint16_t nb_encoding_threads(parser.get<uint32_t>("encthreads"));

    Mat camera_intrinsics_K;
    Mat camera_intrinsics_distortion_coefficients;
    Size camera_intrinsics_image_size_used;
    readCalibration(camera_intrinsics_filename, camera_intrinsics_K, camera_intrinsics_distortion_coefficients, camera_intrinsics_image_size_used);

    uint32_t left_offset;
    uint32_t right_offset;
    bool use_left_audio = true;
    if(video_offset >= 0) {
        left_offset = 0;
        right_offset = video_offset;
    } else {
        left_offset = -1*video_offset;
        right_offset = 0;
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

    const uint32_t input_queue_size = static_cast<uint32_t>(std::ceil((float)nb_stitch_workers/2.0));
    InputProcessor left_processor(left_filename, left_offset, input_queue_size);
    left_processor.initialize();
    InputProcessor right_processor(right_filename, right_offset, input_queue_size);
    right_processor.initialize();

    spdlog::info("Left duratiaon: {}", left_processor.duration());
    spdlog::info("Right duratiaon: {}", right_processor.duration());
    double duration = left_processor.duration();
    if(duration > right_processor.duration())
        duration = right_processor.duration();
    ThreadSafeQueue<LeftRightPacket> stitcher_queue(input_queue_size*3);

    spdlog::info("Writting file: {}", output_filename);
    OutputEncoder output_encoder(output_filename, left_processor.getOutAudioQueue(), right_processor.getOutAudioQueue(), pano_width, pano_height, use_left_audio, left_processor.video_time_base(), left_processor.audio_time_base(), input_queue_size, nb_encoding_threads);
    output_encoder.initialize(left_processor.audio_codec_parameters(), right_processor.audio_codec_parameters(), duration);

    // Start IO Threads
    output_encoder.start();
    left_processor.start();
    right_processor.start();

    unordered_map<uint32_t, unique_ptr<VideoPacket>> left_video_packets;
    unordered_map<uint32_t, unique_ptr<VideoPacket>> right_video_packets;

    spdlog::info("Find Ref frame for white balance");
    vector<vector<uint32_t>> reference_bgr_value_idxs;
    vector<vector<double>> reference_bgr_cumsum;
    if(parser.get<bool>("fixexposure")) {
        spdlog::info("Fixing exposure");
        while(true) {
            unique_ptr<VideoPacket> left_input_packet(left_processor.getOutVideoQueue().pop(chrono::seconds(1)));
            if(left_input_packet) {
                FrameStitcher::BuildReferenceHistogram(left_input_packet->data.get(), left_input_packet->width, left_input_packet->height, reference_bgr_value_idxs, reference_bgr_cumsum);

                uint32_t idx = left_input_packet->idx;
                left_video_packets.insert(make_pair(idx, move(left_input_packet)));
                break;
            }
        }
    }


    vector<unique_ptr<FrameStitcher>> frame_stitchers;
    for(uint16_t i=0; i < nb_stitch_workers; ++i) {
        unique_ptr<FrameStitcher> fs(new FrameStitcher(pano_offset_x, pano_offset_y, pano_width, pano_height, stitcher_queue, output_encoder.getInPanoramicQueue(), cameras_params, camera_intrinsics_K, camera_intrinsics_distortion_coefficients, camera_intrinsics_image_size_used, masks_warped, reference_bgr_value_idxs, reference_bgr_cumsum));
        fs->start();
        frame_stitchers.push_back(move(fs));
    }

    int32_t last_video_idx = numeric_limits<int32_t>::max();
    double last_video_frame_time = 0;
    uint32_t last_left_video_idx = -1;
    double last_left_video_time = 0.0;
    uint32_t last_right_video_idx = -1;
    double last_right_video_time = 0.0;
    uint32_t idx_to_process = 0;
    spdlog::info("Processing all frames");
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    double prev_delta = 0;
    while(true) {
        bool read_left = false;
        bool read_right = false;

        unique_ptr<VideoPacket> left_video_packet(left_processor.getOutVideoQueue().pop(chrono::seconds(1)));
        if(left_video_packet) {
            read_left = true;
            last_left_video_idx = left_video_packet->idx;
            last_left_video_time = left_video_packet->pts_time;
            left_video_packets.insert(make_pair(last_left_video_idx, move(left_video_packet)));
        }

        unique_ptr<VideoPacket> right_video_packet(right_processor.getOutVideoQueue().pop(chrono::seconds(1)));
        if(right_video_packet) {
            read_right = true;
            last_right_video_idx = right_video_packet->idx;
            last_right_video_time = right_video_packet->pts_time;
            right_video_packets.insert(make_pair(last_right_video_idx, move(right_video_packet)));
        }

        if(left_processor.is_done() && left_processor.getOutVideoQueue().size() == 0 && (last_left_video_idx < last_video_idx)) {
            spdlog::info("Left input is done");
            last_video_idx = last_left_video_idx;
            last_video_frame_time = last_left_video_time;
        }
        if(right_processor.is_done() && right_processor.getOutVideoQueue().size() == 0 && (last_right_video_idx < last_video_idx)) {
            spdlog::info("Right input is done");
            last_video_idx = last_right_video_idx;
            last_video_frame_time = last_right_video_time;
        }

        if(last_video_idx != numeric_limits<int32_t>::max()) {
            bool found_one = true;
            for(int32_t i=last_video_idx+1; found_one; i++) {
                found_one = false;
                if(left_video_packets.find((uint32_t)i) != left_video_packets.end() ) {
                    found_one = true;
                    left_video_packets.erase((uint32_t)i);
                }
                if(right_video_packets.find((uint32_t)i) != right_video_packets.end()) {
                    found_one = true;
                    right_video_packets.erase((uint32_t)i);
                }
            }
        }

        while(left_video_packets.find(idx_to_process) != left_video_packets.end() && right_video_packets.find(idx_to_process) != right_video_packets.end()) {
            unique_ptr<LeftRightPacket> lr_packet(new LeftRightPacket());
            
            lr_packet->left_data_size = left_video_packets[idx_to_process]->data_size;
            lr_packet->left_data = move(left_video_packets[idx_to_process]->data);
            lr_packet->right_data_size = right_video_packets[idx_to_process]->data_size;
            lr_packet->right_data = move(right_video_packets[idx_to_process]->data);
            lr_packet->width = left_video_packets[idx_to_process]->width;
            lr_packet->height = left_video_packets[idx_to_process]->height;
            lr_packet->pts = left_video_packets[idx_to_process]->pts;
            lr_packet->pts_time = left_video_packets[idx_to_process]->pts_time;
            lr_packet->idx = left_video_packets[idx_to_process]->idx;

            stitcher_queue.push(move(lr_packet));

            left_video_packets.erase(idx_to_process);
            right_video_packets.erase(idx_to_process);
            ++idx_to_process;

            double delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();
            // double fps = ((double)idx_to_process/delta) * 1000.0;

            if((delta - prev_delta) > 60000.0) {
                prev_delta = delta;
                spdlog::debug("L: {} R: {} S: {} oP:{}", 
                                left_processor.getOutVideoQueue().estimated_size(),
                                right_processor.getOutVideoQueue().estimated_size(),
                                stitcher_queue.estimated_size(),
                                output_encoder.getInPanoramicQueue().estimated_size());
            }
            // spdlog::debug("Main FPS: {}", fps);
        }

        if(!read_left && !read_right && left_processor.is_done() && right_processor.is_done()) {
            break;
        }
    }
    left_processor.stop();
    right_processor.stop();

    output_encoder.stop();
    spdlog::info("Done");
}
