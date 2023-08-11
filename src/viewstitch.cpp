#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

#include "framestitcher.hpp"
#include "inputprocessor.hpp"
#include "readers.hpp"

using namespace std;
using namespace cv;


int main(int argc, const char ** argv) {
    spdlog::set_pattern("%Y%m%dT%H:%M:%S.%e [%^%l%$] -%n- -%t- : %v");
    spdlog::set_level(spdlog::level::debug);

    string left_filename(argv[1]);
    string right_filename(argv[2]);
    int32_t video_offset(atoi(argv[3])); // Positive offsets shifts right and takes left audio. Negative is the opposite
    string camera_params_filename(argv[4]);

    uint32_t left_offset;
    uint32_t right_offset;
    if(video_offset >= 0) {
        left_offset = 0;
        right_offset = video_offset;
    } else {
        left_offset = -1*video_offset;
        right_offset = 0;
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

    const uint32_t input_queue_size = 2;
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

    // Start IO Threads
    left_processor.start();
    right_processor.start();

    unordered_map<uint32_t, unique_ptr<VideoPacket>> left_video_packets;
    unordered_map<uint32_t, unique_ptr<VideoPacket>> right_video_packets;

    spdlog::info("Find Ref frame for white balance");
    vector<vector<uint32_t>> reference_bgr_value_idxs;
    vector<vector<double>> reference_bgr_cumsum;
    chrono::milliseconds audio_wait(chrono::milliseconds(1));
    while(true) {
        while(true) {
            unique_ptr<AVPacket, PacketDeleter> audio_packet(left_processor.getOutAudioQueue().pop(audio_wait));
            if(!audio_packet)
                break;
        }
        while(true) {
            unique_ptr<AVPacket, PacketDeleter> audio_packet(right_processor.getOutAudioQueue().pop(audio_wait));
            if(!audio_packet)
                break;
        }
        unique_ptr<VideoPacket> left_input_packet(left_processor.getOutVideoQueue().pop(chrono::seconds(1)));
        if(left_input_packet) {
            FrameStitcher::BuildReferenceHistogram(left_input_packet->data.get(), left_input_packet->width, left_input_packet->height, reference_bgr_value_idxs, reference_bgr_cumsum);

            uint32_t idx = left_input_packet->idx;
            left_video_packets.insert(make_pair(idx, move(left_input_packet)));
            break;
        }
    }
    spdlog::info("Found Ref frame for white balance");

    ThreadSafeQueue<PanoramicPacket> panoramic_packet_queue(input_queue_size);
    vector<unique_ptr<FrameStitcher>> frame_stitchers;
    unique_ptr<FrameStitcher> fs(new FrameStitcher(pano_offset_x, pano_offset_y, pano_width, pano_height, stitcher_queue, panoramic_packet_queue, cameras_params, masks_warped, reference_bgr_value_idxs, reference_bgr_cumsum));
    fs->start();
    frame_stitchers.push_back(move(fs));

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
    chrono::milliseconds video_wait(chrono::milliseconds(10));
    Mat img, img_small;
    spdlog::info("Process loop...");
    while(true) {
        while(true) {
            unique_ptr<AVPacket, PacketDeleter> audio_packet(left_processor.getOutAudioQueue().pop(audio_wait));
            if(!audio_packet)
                break;
        }
        while(true) {
            unique_ptr<AVPacket, PacketDeleter> audio_packet(right_processor.getOutAudioQueue().pop(audio_wait));
            if(!audio_packet)
                break;
        }

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

        spdlog::info("Reading video packets");
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

            unique_ptr<PanoramicPacket> new_panoramic_packet(panoramic_packet_queue.pop(video_wait));
            if(new_panoramic_packet) {
                spdlog::info("Displaying pano");
                Mat pano(new_panoramic_packet->height * 3/2, new_panoramic_packet->width, CV_8UC1, new_panoramic_packet->data.get());
                cvtColor(pano, img, COLOR_YUV2BGR_I420);
                resize(img, img_small, Size(1920, 1080), 0, 0, INTER_CUBIC);
                
                imshow("Pano", img_small);
                waitKey(1);
            }
        }

        if(!read_left && !read_right && left_processor.is_done() && right_processor.is_done()) {
            break;
        }
    }
    left_processor.stop();
    right_processor.stop();

    spdlog::info("Done");
}
