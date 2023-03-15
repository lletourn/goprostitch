#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

#include "framestitcher.hpp"
#include "inputprocessor.hpp"
#include "outputencoder.hpp"

using namespace std;
using namespace cv;

vector<detail::CameraParams> buildCamParams() {
    spdlog::info("Setting up camera params");
    Mat left_R(3, 3, CV_32F);
    left_R.at<float>(0,0) = 0.98906273;
    left_R.at<float>(0,1) = -0.076125443;
    left_R.at<float>(0,2) = -0.12633191;
    left_R.at<float>(1,0) = -5.9456329e-10;
    left_R.at<float>(1,1) = 0.85651541;
    left_R.at<float>(1,2) = -0.51612151;
    left_R.at<float>(2,0) = 0.1474952;
    left_R.at<float>(2,1) = 0.51047659;
    left_R.at<float>(2,2) = 0.84714752;

    detail::CameraParams  camera_left;
    camera_left.focal = 2137.88;
    camera_left.aspect = 1.0;
    camera_left.ppx = 516.5;
    camera_left.ppy = 290.5;
    camera_left.R = left_R;
    camera_left.t = Mat::zeros(3, 1, CV_64F);

    Mat right_R(3, 3, CV_32F);
    right_R.at<float>(0,0) = 0.98918295;
    right_R.at<float>(0,1) = 0.074548103;
    right_R.at<float>(0,2) = 0.12633191;
    right_R.at<float>(1,0) = -1.2479756e-08;
    right_R.at<float>(1,1) = 0.86123264;
    right_R.at<float>(1,2) = -0.50821096;
    right_R.at<float>(2,0) = -0.1466873;
    right_R.at<float>(2,1) = 0.50271362;
    right_R.at<float>(2,2) = 0.85191655;

    detail::CameraParams camera_right;
    camera_right.focal = 2119.94;
    camera_right.aspect = 1.0;
    camera_right.ppx = 516.5;
    camera_right.ppy = 290.5;
    camera_right.R = right_R;
    camera_right.t = Mat::zeros(3, 1, CV_64F);

    vector<detail::CameraParams> camera_params;
    camera_params.push_back(camera_left);
    camera_params.push_back(camera_right);
    return camera_params;
}


void handle_packet(unique_ptr<InputPacket> packet, ThreadSafeQueue<InputPacket>& audio_queue, unordered_map<uint32_t, unique_ptr<InputPacket>>& packets) {
    if (packet->type == InputPacket::InputPacketType::AUDIO) {
        audio_queue.push(move(packet));
    } else {
        //packets.emplace(packet->idx, packet);
        packets.insert(make_pair(packet->idx, move(packet)));
    }
}


int main(int argc, const char ** argv) {
    spdlog::set_pattern("%Y%m%dT%H:%M:%S.%e [%^%l%$] -%n- -%t- : %v");
    spdlog::set_level(spdlog::level::debug);

    string left_filename(argv[1]);
    string right_filename(argv[2]);
    string output_filename(argv[3]);
    uint16_t nb_workers(atoi(argv[4]));

    spdlog::info("Loading file: {}", left_filename);
    spdlog::info("Loading file: {}", right_filename);
    InputProcessor left_processor(left_filename, 60);
    InputProcessor right_processor(right_filename, 60);
    left_processor.start();
    right_processor.start();

    spdlog::info("Writting file: {}", output_filename);
    OutputEncoder output_encoder(output_filename, 60);
    output_encoder.start();

    ThreadSafeQueue<LeftRightPacket> stitcher_queue(40);

    vector<detail::CameraParams> camera_params = buildCamParams();

    unordered_map<uint32_t, unique_ptr<InputPacket>> left_video_packets;
    unordered_map<uint32_t, unique_ptr<InputPacket>> right_video_packets;

    spdlog::info("Find Ref frame for white balance");
    Mat reference_image;
    while(reference_image.data == nullptr) {
        unique_ptr<InputPacket> left_input_packet(left_processor.getOutQueue().pop(chrono::seconds(1)));
        if(left_input_packet) {
            handle_packet(move(left_input_packet), output_encoder.getInLeftAudioQueue(), left_video_packets);
            if(left_video_packets.size() > 0) {
                auto iter = left_video_packets.cbegin();
                spdlog::debug("Img sizes: {} vs {}", (iter->second->height * iter->second->width * 3), iter->second->data_size);
                Mat temp_image(iter->second->height, iter->second->width, CV_8UC3, iter->second->data.get());
                reference_image = temp_image.clone();
            }
        }
    }
    vector<vector<uint32_t>> reference_bgr_value_idxs;
    vector<vector<double>> reference_bgr_cumsum;
    FrameStitcher::BuildReferenceHistogram(reference_image, reference_bgr_value_idxs, reference_bgr_cumsum);

    vector<unique_ptr<FrameStitcher>> frame_stitchers;
    for(uint16_t i=0; i < nb_workers; ++i) {
        unique_ptr<FrameStitcher> fs(new FrameStitcher(stitcher_queue, output_encoder.getInPanoramicQueue(), camera_params, reference_bgr_value_idxs, reference_bgr_cumsum, 10));
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
    while(true) {
        bool read_left = false;
        bool read_right = false;

        unique_ptr<InputPacket> left_input_packet(left_processor.getOutQueue().pop(chrono::seconds(1)));
        if(left_input_packet) {
            read_left = true;
            last_left_video_idx = left_input_packet->idx;
            last_left_video_time = left_input_packet->pts_time;
            handle_packet(move(left_input_packet), output_encoder.getInLeftAudioQueue(), left_video_packets);
        }

        unique_ptr<InputPacket> right_input_packet(right_processor.getOutQueue().pop(chrono::seconds(1)));
        if(right_input_packet) {
            read_right = true;
            last_right_video_idx = right_input_packet->idx;
            last_right_video_time = right_input_packet->pts_time;
            handle_packet(move(right_input_packet), output_encoder.getInRightAudioQueue(), right_video_packets);
        }

        if(left_processor.is_done() && left_processor.getOutQueue().size() == 0 && (last_left_video_idx < last_video_idx)) {
            last_video_idx = last_left_video_idx;
            last_video_frame_time = last_left_video_time;
        }
        if(right_processor.is_done() && right_processor.getOutQueue().size() == 0 && (last_right_video_idx < last_video_idx)) {
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
        }

        if(!read_left && !read_right && left_processor.is_done() && right_processor.is_done()) {
            break;
        }
    }
    left_processor.stop();
    right_processor.stop();

    for(unique_ptr<FrameStitcher>& fs : frame_stitchers) {
        fs->stop();
    }

    output_encoder.stop();
    spdlog::info("Done");
}
