#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <thread>


#include "threadsafequeue.hpp"

class FrameStitcher {
 public:
    FrameStitcher(ThreadSafeQueue<InputPacket>& input_packets, uint32_t max_queue_size);
    ~FrameStitcher();

    void start();
    void stop();
    void run();
    bool is_done();

    ThreadSafeQueue<InputPacket>& getOutAudioQueue();
    ThreadSafeQueue<InputPacket>& getOutVideoQueue();

    void close();

 private:
    ThreadSafeQueue<InputPacket>& input_packets_;
    ThreadSafeQueue<InputPacket> output_audio_packets_;
    ThreadSafeQueue<InputPacket> output_video_packets_;
    bool running_;
    std::atomic<bool> done_;

    std::thread thread_;

};
