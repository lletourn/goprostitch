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
    InputProcessor(const std::string& filename, bool use_gpu, uint32_t offset, uint32_t queue_size);
    ~InputProcessor();

    void initialize();
    void start();
    void stop();
    void run();
    bool is_done();

    Rational video_time_base() {return video_time_base_;};
    Rational video_frame_rate() {return video_frame_rate_;};
    Rational audio_time_base() {return audio_time_base_;};
    PixelFormat pixel_format() {return pixel_format_;};
    AVColorSpace colorspace() {return colorspace_;};
    AVColorRange color_range() {return color_range_;};
    double duration() {return duration_;};
    // Pointer is owned by this class
    const AVCodecParameters* audio_codec_parameters() {return av_format_ctx_->streams[audio_stream_idx_]->codecpar;};
    
    ThreadSafeQueue<VideoPacket>& getOutVideoQueue();
    ThreadSafeQueue<AVPacket, PacketDeleter>& getOutAudioQueue();

    void close();

 private:
    const std::string filename_;
    bool use_gpu_;
    const uint32_t offset_;
    uint64_t timecode_;
    std::atomic<bool> running_;
    std::atomic<bool> done_;
    Rational video_time_base_;
    Rational video_frame_rate_;
    Rational audio_time_base_;
    PixelFormat pixel_format_;
    AVColorSpace colorspace_;
    AVColorRange color_range_;
    double duration_;

    ThreadSafeQueue<VideoPacket> video_packet_queue_;
    ThreadSafeQueue<AVPacket, PacketDeleter> audio_packet_queue_;
    std::thread thread_;

    AVFormatContext* av_format_ctx_;
    AVCodecContext* video_codec_ctx_;
    const AVCodec* video_codec_;
    AVBufferRef *hw_device_ctx_;
    int video_stream_idx_;
    int audio_stream_idx_;

};
