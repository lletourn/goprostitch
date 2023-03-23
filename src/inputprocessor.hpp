#pragma once

#include <atomic>
#include <chrono>
#include <ratio>
#include <string>
#include <thread>

extern "C" {
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
}

#include "datatypes.hpp"
#include "threadsafequeue.hpp"

class InputProcessor {
 public:
    InputProcessor(const std::string& filename, uint32_t queue_size);
    ~InputProcessor();

    void initialize();
    void start();
    void stop();
    void run();
    bool is_done();

    Rational video_time_base() {return video_time_base_;};
    Rational audio_time_base() {return audio_time_base_;};
    double duration() {return duration_;};
    // Pointer is owned by this class
    const AVCodecParameters* audio_codec_parameters() {return av_format_ctx_->streams[audio_stream_]->codecpar;};
    
    ThreadSafeQueue<VideoPacket>& getOutVideoQueue();
    ThreadSafeQueue<AVPacket, PacketDeleter>& getOutAudioQueue();

    void close();

 private:
    const std::string filename_;
    uint64_t timecode_;
    bool running_;
    std::atomic<bool> done_;
    Rational video_time_base_;
    Rational audio_time_base_;
    double duration_;

    ThreadSafeQueue<VideoPacket> video_packet_queue_;
    ThreadSafeQueue<AVPacket, PacketDeleter> audio_packet_queue_;
    std::thread thread_;

    AVFormatContext* av_format_ctx_;
    AVCodecContext* video_codec_ctx_orig_;
    AVCodecContext* video_codec_ctx_;
    const AVCodec* video_codec_;
    int video_stream_;
    int audio_stream_;

};
