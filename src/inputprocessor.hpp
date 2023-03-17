#pragma once

#include <atomic>
#include <chrono>
#include <ratio>
#include <string>
#include <thread>

#include "datatypes.hpp"
#include "threadsafequeue.hpp"

class InputProcessor {
 public:
    InputProcessor(const std::string& filename, uint32_t video_queue_size, uint32_t audio_queue_size);
    ~InputProcessor();

    void start();
    void stop();
    void run();
    bool is_done();

    Rational video_time_base() {return video_time_base_.load();};
    Rational audio_time_base() {return audio_time_base_.load();};

    ThreadSafeQueue<VideoPacket>& getOutVideoQueue();
    ThreadSafeQueue<AudioPacket>& getOutAudioQueue();

    void close();

 private:
    const std::string filename_;
    uint64_t timecode_;
    bool running_;
    std::atomic<bool> done_;
    std::atomic<Rational> video_time_base_;
    std::atomic<Rational> audio_time_base_;

    ThreadSafeQueue<VideoPacket> video_packet_queue_;
    ThreadSafeQueue<AudioPacket> audio_packet_queue_;
    std::thread thread_;

};
