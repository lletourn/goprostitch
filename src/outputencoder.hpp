#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include "datatypes.hpp"
#include "threadsafequeue.hpp"

class OutputEncoder {
 public:
    OutputEncoder(const std::string& filename, ThreadSafeQueue<AudioPacket>& left_audio_queue, ThreadSafeQueue<AudioPacket>& right_audio_queue, Rational video_time_base, Rational audio_time_base, uint32_t queue_size);
    ~OutputEncoder();

    void start();
    void stop();
    void run();
    bool is_done();

    ThreadSafeQueue<PanoramicPacket>& getInPanoramicQueue();

    void close();

 private:
    const std::string filename_;
    bool running_;
    std::atomic<bool> done_;
    Rational video_time_base_;
    Rational audio_time_base_;

    ThreadSafeQueue<AudioPacket>& left_audio_packet_queue_;
    ThreadSafeQueue<AudioPacket>& right_audio_packet_queue_;
    ThreadSafeQueue<PanoramicPacket> panoramic_packet_queue_;
    std::thread thread_;
    std::queue<std::unique_ptr<AudioPacket>> left_audio_packets_;
    std::queue<std::unique_ptr<AudioPacket>> right_audio_packets_;
};
