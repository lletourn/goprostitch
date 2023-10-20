#include "inputsyncer.hpp"

#include <limits>
#include <spdlog/spdlog.h>

using namespace std;

InputSyncer::InputSyncer(ThreadSafeQueue<VideoPacket>& left_queue, ThreadSafeQueue<VideoPacket>& right_queue)
: left_queue_(left_queue), right_queue_(right_queue) {
    readers_are_done_ = false;
    next_frame_idx_ = 0;
    last_frame_idx_ = numeric_limits<int32_t>::max();
    left_packet_.reset(nullptr);
    right_packet_.reset(nullptr);
}

InputSyncer::~InputSyncer() {
}

unique_ptr<LeftRightPacket> InputSyncer::next_pair() {
    chrono::milliseconds wait_period(15);
    if(next_frame_idx_ >= last_frame_idx_) {
        wait_period = chrono::milliseconds(1);
    }

    if(!left_packet_) {
        left_packet_ = left_queue_.pop(wait_period);
    }

    if(!right_packet_) {
        right_packet_ = right_queue_.pop(wait_period);
    }

    if(next_frame_idx_ >= last_frame_idx_ && (right_packet_ || left_packet_)) {
        left_packet_.reset(nullptr);
        right_packet_.reset(nullptr);
    }

    spdlog::debug("NextFrame: {} Has Left frame: {} Has Right Frame: {}", next_frame_idx_, (bool)left_packet_, (bool)right_packet_);

    if((!left_packet_ || !right_packet_) && readers_are_done_ && last_frame_idx_ == numeric_limits<int32_t>::max())
        last_frame_idx_ = next_frame_idx_;

    if(left_packet_ && right_packet_ && left_packet_->idx == next_frame_idx_ && right_packet_->idx == next_frame_idx_) {
        unique_ptr<LeftRightPacket> lr_packet(new LeftRightPacket());
  
        lr_packet->left_data_size = left_packet_->data_size;
        lr_packet->left_data = move(left_packet_->data);
        lr_packet->right_data_size = right_packet_->data_size;
        lr_packet->right_data = move(right_packet_->data);
        lr_packet->width = left_packet_->width;
        lr_packet->height = left_packet_->height;
        lr_packet->pts = left_packet_->pts;
        lr_packet->pts_time = left_packet_->pts_time;
        lr_packet->idx = left_packet_->idx;
 
        left_packet_.reset(nullptr);
        right_packet_.reset(nullptr);

        next_frame_idx_++;

        return move(lr_packet);
    }

    return nullptr;
}
