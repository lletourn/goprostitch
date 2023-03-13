#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

#include "inputprocessor.hpp"

using namespace std;
using namespace cv;

vector<detail::CameraParams> buildCamParams() {
    spdlog::info("Setting up camera params");
    detail::CameraParams  camera_left;
    camera_left.focal = 1380.9097896938035;
    camera_left.aspect = 1.0;
    camera_left.ppx = 516.5;
    camera_left.ppy = 290.5;
    camera_left.R = Mat_<double>(3,3) << 9.7980595e-01, -5.8920074e-02, -1.9107230e-01, -9.6140185e-10,  9.5559806e-01, -2.9467332e-01, 1.9995050e-01, 2.8872266e-01, 9.3630064e-01;
    camera_left.t = Mat_<double>(3,1) << 0., 0., 0.;

    detail::CameraParams camera_right;
    camera_right.focal = 1384.4108312478572;
    camera_right.aspect = 1.0;
    camera_right.ppx = 516.5;
    camera_right.ppy = 290.5;
    camera_right.R = Mat_<double>(3,3) << 9.7996306e-01, 5.6246426e-02, 1.9107229e-01, 2.2138309e-08, 9.5929933e-01, -2.8239137e-01, -1.9917902e-01, 2.7673310e-01, 9.4007784e-01;
    camera_right.t = Mat_<double>(3,1) << 0., 0., 0.;

    vector<detail::CameraParams> camera_params;
    camera_params.push_back(camera_left);
    camera_params.push_back(camera_right);
    return camera_params;
}


void handle_packet(unique_ptr<InputPacket> packet, unordered_map<uint32_t, unique_ptr<InputPacket>>& packets) {
    if (packet->type == InputPacket::InputPacketType::AUDIO) {
        // push to encoder
        packet.reset();
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
    spdlog::info("Loading file: {}", left_filename);
    spdlog::info("Loading file: {}", right_filename);
    InputProcessor left_processor(left_filename, 60);
    InputProcessor right_processor(right_filename, 60);
    left_processor.start();
    right_processor.start();

    unordered_map<uint32_t, unique_ptr<InputPacket>> left_video_packets;
    unordered_map<uint32_t, unique_ptr<InputPacket>> right_video_packets;

    spdlog::info("Find Ref frame for white balance");
    Mat reference_image;
    while(reference_image.data == nullptr) {
        unique_ptr<InputPacket> left_input_packet(left_processor.getOutQueue().pop(chrono::seconds(1)));
        if(left_input_packet) {
            handle_packet(move(left_input_packet), left_video_packets);
            if(left_video_packets.size() > 0) {
                auto iter = left_video_packets.cbegin();
                spdlog::debug("Img sizes: {} vs {}", (iter->second->height * iter->second->width * 3), iter->second->data_size);
                Mat temp_image(iter->second->height, iter->second->width, CV_8UC3, iter->second->data.get());
                reference_image = temp_image.clone();
            }
        }
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
            handle_packet(move(left_input_packet), left_video_packets);
            /*
            cv::Mat left_mat = cv::Mat(left_input_packet->height, left_input_packet->width, CV_8UC3, left_input_packet->data.get());
            Mat dst;
            resize(left_mat, dst, Size(1280, 720), 0, 0, INTER_CUBIC);
            cv::imshow("Left", dst);
            cv::waitKey();
            cv::destroyWindow("Left");
            left_input_packet.reset();
            */
        }

        unique_ptr<InputPacket> right_input_packet(right_processor.getOutQueue().pop(chrono::seconds(1)));
        if(right_input_packet) {
            read_right = true;
            last_right_video_idx = right_input_packet->idx;
            last_right_video_time = right_input_packet->pts_time;
            handle_packet(move(right_input_packet), right_video_packets);
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
            spdlog::info("Found frame match: {} vs {}", left_video_packets.find(idx_to_process)->second->pts_time, right_video_packets.find(idx_to_process)->second->pts_time);
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
    spdlog::info("Done");
}
