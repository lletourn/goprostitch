#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <thread>


#include "threadsafequeue.hpp"

struct InputPacket {
    enum InputPacketType {AUDIO, VIDEO};

    InputPacketType type;
    uint64_t data_size;
    std::unique_ptr<uint8_t> data;
    uint32_t width;
    uint32_t height;
    uint32_t pts;
    double pts_time;
    uint32_t idx;
    int format;
    uint64_t layout;
    uint32_t sample_rate;
};

class InputProcessor {
 public:
    InputProcessor(const std::string& filename, uint32_t queue_size);
    ~InputProcessor();

    void start();
    void stop();
    void run();
    bool is_done();

    ThreadSafeQueue<InputPacket>& getOutQueue();

    void close();

 private:
    const std::string filename_;
    uint64_t timecode_;
    bool running_;
    std::atomic<bool> done_;

    ThreadSafeQueue<InputPacket> packet_queue_;
    std::thread thread_;

};
