#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>


struct Rational {
    uint32_t num;
    uint32_t den;

    Rational(uint32_t n, uint32_t d) : num(n), den(d) {};
};

struct LeftRightPacket {
    uint64_t left_data_size;
    std::unique_ptr<uint8_t> left_data;
    uint64_t right_data_size;
    std::unique_ptr<uint8_t> right_data;
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
    std::unique_ptr<uint8_t> data;
};
struct VideoPacket {
    uint32_t width;
    uint32_t height;
    uint32_t pts;
    double pts_time;
    uint32_t idx;
    uint64_t data_size;
    std::unique_ptr<uint8_t> data;
};

struct AudioPacket {
    uint32_t pts;
    double pts_time;
    uint32_t idx;
    int format;
    uint64_t layout;
    uint32_t sample_rate;
    uint64_t data_size;
    std::unique_ptr<uint8_t> data;
};
