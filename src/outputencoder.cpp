#include "outputencoder.hpp"

#include <stdexcept>
#include <iostream>
#include <csignal>
#include <spdlog/spdlog.h>

extern "C" {
  #include <libavutil/error.h>
  #include <libavutil/imgutils.h>
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
  #include <libswscale/swscale.h>
}

using namespace std;

OutputEncoder::OutputEncoder(const string& filename, uint32_t queue_size)
: filename_(filename), running_(false), done_(false), left_audio_packet_queue_(queue_size), right_audio_packet_queue_(queue_size), panoramic_packet_queue_(queue_size) {
};


OutputEncoder::~OutputEncoder() {
}


void OutputEncoder::start() {
    if (thread_.joinable()) {
        return;
    }
    running_ = true;
    done_ = false;
    thread_ = move(thread(&OutputEncoder::run, this));
}


void OutputEncoder::stop() {
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

bool OutputEncoder::is_done() {
    return done_.load();
} 

ThreadSafeQueue<InputPacket>& OutputEncoder::getInLeftAudioQueue() {
    return left_audio_packet_queue_;
}

ThreadSafeQueue<InputPacket>& OutputEncoder::getInRightAudioQueue() {
    return right_audio_packet_queue_;
}

ThreadSafeQueue<PanoramicPacket>& OutputEncoder::getInPanoramicQueue() {
    return panoramic_packet_queue_;
}

void OutputEncoder::run() {
    #ifdef _GNU_SOURCE
    pthread_setname_np(pthread_self(), "OutputEncoder");
    #endif

    chrono::duration audio_wait(chrono::milliseconds(1));
    chrono::duration video_wait(chrono::milliseconds(100));

    uint32_t frame_idx=0;
    double prev_fps = -1;
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    while(running_) {
        while(true) {
            unique_ptr<InputPacket> audio_packet(left_audio_packet_queue_.pop(audio_wait));
            if(audio_packet) {
                left_audio_packets_.push(move(audio_packet));
            } else {
                break;
            }
        }
        while(true) {
            unique_ptr<InputPacket> audio_packet(right_audio_packet_queue_.pop(audio_wait));
            if(audio_packet) {
                right_audio_packets_.push(move(audio_packet));
            } else {
                break;
            }
        }
        
        unique_ptr<PanoramicPacket> panoramic_packet(panoramic_packet_queue_.pop(video_wait));
        if(panoramic_packet) {
            while(!left_audio_packets_.empty() && left_audio_packets_.front()->pts_time <= panoramic_packet->pts_time) {
                // Encode audio
                unique_ptr<InputPacket> audio_packet(move(left_audio_packets_.front()));
                left_audio_packets_.pop();
                audio_packet.reset();
            }
            while(!right_audio_packets_.empty() && right_audio_packets_.front()->pts_time <= panoramic_packet->pts_time) {
                // Encode audio
                unique_ptr<InputPacket> audio_packet(move(right_audio_packets_.front()));
                right_audio_packets_.pop();
                audio_packet.reset();
            }
            panoramic_packet.reset();
            ++frame_idx;

            auto delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();
            double fps = ((double)frame_idx/delta) * 1000.0; 
            if(fps != prev_fps) {
                spdlog::debug("FPS: {}", fps);
            }
        }
    }

    done_ = true;
}
