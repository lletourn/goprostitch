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
    OutputEncoder(const std::string& filename, ThreadSafeQueue<AudioPacket>& left_audio_queue, ThreadSafeQueue<AudioPacket>& right_audio_queue, uint32_t width, uint32_t height, Rational video_time_base, Rational audio_time_base, uint32_t queue_size);
    ~OutputEncoder();

    void start();
    void stop();
    void run();
    bool is_done();

    ThreadSafeQueue<PanoramicPacket>& getInPanoramicQueue();

    void close();

 private:
    void init_audio(AVFormatContext* fmt_ctx, AVStream*& audio_stream, const AVCodec*& audio_codec, AVCodecContext*& audio_codec_ctx, AVPacket*& video_pkt, AVFrame*& audio_frame, const char* title);
    void init_video(AVFormatContext* fmt_ctx, AVStream*& video_stream, const AVCodec*& video_codec, AVCodecContext*& video_codec_ctx, AVPacket*& audio_pkt, AVFrame*& video_frame);
    int encode_audio_frame(AVFrame *frame, AVFormatContext *fmt_ctx, AVCodecContext *audio_codec_ctx, int *data_present);

 private:
    const std::string filename_;
    bool running_;
    std::atomic<bool> done_;
    uint32_t video_width_;
    uint32_t video_height_;
    Rational video_time_base_;
    Rational audio_time_base_;

    ThreadSafeQueue<AudioPacket>& left_audio_packet_queue_;
    ThreadSafeQueue<AudioPacket>& right_audio_packet_queue_;
    ThreadSafeQueue<PanoramicPacket> panoramic_packet_queue_;
    std::thread thread_;
    std::queue<std::unique_ptr<AudioPacket>> left_audio_packets_;
    std::queue<std::unique_ptr<AudioPacket>> right_audio_packets_;
};
