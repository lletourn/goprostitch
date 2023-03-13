#pragma once

#include <chrono>
#include <condition_variable>
#include <queue>
#include <memory>
#include <mutex>

#include <iostream>

template<class T> class ThreadSafeQueue {
  public:
    ThreadSafeQueue(uint64_t max_size)
    : max_size_(max_size), q_(), m_(), read_condition_(), write_condition_() {
    }

    std::unique_ptr<T> pop(std::chrono::milliseconds wait_period=std::chrono::milliseconds(0)) {
        std::unique_lock<std::mutex> lock(m_);
        while (q_.empty()) {
            if (wait_period.count() > 0) {
                if (read_condition_.wait_for(lock, wait_period) == std::cv_status::timeout) {
                    // if timed out, put in dummy value
                    q_.push(std::unique_ptr<T>(nullptr));
                }
            } else {
                read_condition_.wait(lock);
            }
        }
        std::unique_ptr<T> val(move(q_.front()));
        q_.pop();
        write_condition_.notify_one();
        return val;
    }

    void push(std::unique_ptr<T> t) {
        std::unique_lock<std::mutex> lock(m_);
        if(q_.size() == max_size_) {
            write_condition_.wait(lock);
        }
        q_.push(move(t));
        read_condition_.notify_one();
    }

    uint64_t size() {
        std::lock_guard<std::mutex> lock(m_);
        return (uint64_t)q_.size();
    }

    uint64_t estimated_size() {
        return (uint64_t)q_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_);
        std::queue<std::unique_ptr<T>> empty;
        std::swap(q_, empty);
    }

  private:
    uint64_t max_size_;
    std::queue<std::unique_ptr<T>> q_;
    std::mutex m_;
    std::condition_variable read_condition_;
    std::condition_variable write_condition_;
};
