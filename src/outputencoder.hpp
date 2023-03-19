#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

extern "C" {
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
}
#include "datatypes.hpp"
#include "threadsafequeue.hpp"

class OutputEncoder {
 public:
    OutputEncoder(const std::string& filename, ThreadSafeQueue<AVPacket, PacketDeleter>& left_audio_queue, ThreadSafeQueue<AVPacket, PacketDeleter>& right_audio_queue, uint32_t width, uint32_t height, Rational video_time_base, Rational audio_time_base, uint32_t queue_size);
    ~OutputEncoder();

    void initialize(const AVCodecParameters* left_audio_codec_parameters, const AVCodecParameters* right_audio_codec_parameters);
    void start();
    void stop();
    void run();
    bool is_done();

    ThreadSafeQueue<PanoramicPacket>& getInPanoramicQueue();

    void close();

 private:
    AVStream* init_audio(const AVCodecParameters* audio_codec_parameters, const char* title);
    void init_video();
    void write_audio(AVPacket* packet, AVStream* audio_stream);

 private:
    const std::string filename_;
    bool running_;
    std::atomic<bool> done_;
    uint32_t video_width_;
    uint32_t video_height_;
    Rational video_time_base_;

    ThreadSafeQueue<AVPacket, PacketDeleter>& left_audio_packet_queue_;
    ThreadSafeQueue<AVPacket, PacketDeleter>& right_audio_packet_queue_;
    ThreadSafeQueue<PanoramicPacket> panoramic_packet_queue_;
    std::thread thread_;
    std::queue<std::unique_ptr<AVPacket, PacketDeleter>> left_audio_packets_;
    std::queue<std::unique_ptr<AVPacket, PacketDeleter>> right_audio_packets_;

    AVRational audio_time_base_;
    AVFormatContext* av_format_ctx_;
    AVStream* video_stream_;
    const AVCodec* video_codec_;
    AVCodecContext* video_codec_ctx_;
    AVFrame* video_frame_;
    AVStream* left_audio_stream_;
    AVStream* right_audio_stream_;
};
