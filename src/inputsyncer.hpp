#pragma once

#include <chrono>
#include <memory>
#include <unordered_map>

#include "datatypes.hpp"
#include "threadsafequeue.hpp"

class InputSyncer {
 public:
    InputSyncer(ThreadSafeQueue<VideoPacket>& left_queue, ThreadSafeQueue<VideoPacket>& right_queue);
    ~InputSyncer();

    void set_reader_done() {readers_are_done_ = true;};
    std::unique_ptr<LeftRightPacket> next_pair();

 private:
    ThreadSafeQueue<VideoPacket>& left_queue_;
    ThreadSafeQueue<VideoPacket>& right_queue_;

    bool readers_are_done_;
    uint32_t next_frame_idx_;
    uint32_t last_frame_idx_;
    std::unique_ptr<VideoPacket> left_packet_;
    std::unique_ptr<VideoPacket> right_packet_;
};
