#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include "framestitcher.hpp"
#include "inputprocessor.hpp"
#include "threadsafequeue.hpp"

class OutputEncoder {
 public:
    OutputEncoder(const std::string& filename, uint32_t queue_sizes);
    ~OutputEncoder();

    void start();
    void stop();
    void run();
    bool is_done();

    ThreadSafeQueue<InputPacket>& getInLeftAudioQueue();
    ThreadSafeQueue<InputPacket>& getInRightAudioQueue();
    ThreadSafeQueue<PanoramicPacket>& getInPanoramicQueue();

    void close();

 private:
    const std::string filename_;
    bool running_;
    std::atomic<bool> done_;

    ThreadSafeQueue<InputPacket> left_audio_packet_queue_;
    ThreadSafeQueue<InputPacket> right_audio_packet_queue_;
    ThreadSafeQueue<PanoramicPacket> panoramic_packet_queue_;
    std::thread thread_;
    std::queue<std::unique_ptr<InputPacket>> left_audio_packets_;
    std::queue<std::unique_ptr<InputPacket>> right_audio_packets_;
};
