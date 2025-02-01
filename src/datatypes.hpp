#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

extern "C" {
    #include <libavcodec/packet.h>
}

enum PixelFormat {
    PIX_FMT_NONE = 0,
    PIX_FMT_YUV420_NV12, // Semi packed
    PIX_FMT_YUV420_P, // Planar P == I420
    PIX_FMT_RGB24,
    PIX_FMT_BGR24
};

struct Rational {
    uint32_t num;
    uint32_t den;

    Rational(uint32_t n, uint32_t d) : num(n), den(d) {};
};

struct LeftRightPacket {
    uint64_t left_data_size;
    std::unique_ptr<uint8_t[]> left_data;
    uint64_t right_data_size;
    std::unique_ptr<uint8_t[]> right_data;
    uint32_t width;
    uint32_t height;
    uint32_t pts;
    double pts_time;
    uint32_t idx;
};

struct PanoramicPacket {
    uint32_t width;
    uint32_t height;
    uint32_t pts;
    double pts_time;
    uint32_t idx;
    uint64_t data_size;
    std::unique_ptr<uint8_t[]> data;
};
struct VideoPacket {
    uint32_t width;
    uint32_t height;
    uint32_t pts;
    double pts_time;
    uint32_t idx;
    uint64_t data_size;
    std::unique_ptr<uint8_t[]> data;
};

struct PacketDeleter {
    void operator()(AVPacket* packet) const {
        av_packet_free(&packet);
    }
};

using AVPacketUniquePTR = std::unique_ptr<AVPacket, PacketDeleter>;
