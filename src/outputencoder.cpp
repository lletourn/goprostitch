#include "outputencoder.hpp"

#include <iostream>
#include <csignal>
#include <stdexcept>
#include <sstream>
#include <spdlog/spdlog.h>

#include <opencv2/opencv.hpp>
extern "C" {
  #include <libavutil/error.h>
  #include <libavutil/imgutils.h>
  #include <libavutil/opt.h>
}

using namespace std;

OutputEncoder::OutputEncoder(const string& filename, ThreadSafeQueue<AVPacket, PacketDeleter>& left_audio_queue, ThreadSafeQueue<AVPacket, PacketDeleter>& right_audio_queue, uint32_t width, uint32_t height, Rational video_time_base, Rational audio_time_base, uint32_t queue_size)
: filename_(filename), running_(false), done_(false), video_width_(width), video_height_(height), video_time_base_(video_time_base), left_audio_packet_queue_(left_audio_queue), right_audio_packet_queue_(right_audio_queue), panoramic_packet_queue_(queue_size) {
    audio_time_base_ = (AVRational){(int)audio_time_base.num, (int)audio_time_base.den};
};


OutputEncoder::~OutputEncoder() {
}


void OutputEncoder::start() {
    if (thread_.joinable()) {
        return;
    }
    running_ = true;
    done_ = false;
    thread_ = move(thread(&OutputEncoder::run, this));
}


void OutputEncoder::stop() {
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

bool OutputEncoder::is_done() {
    return done_.load();
} 

ThreadSafeQueue<PanoramicPacket>& OutputEncoder::getInPanoramicQueue() {
    return panoramic_packet_queue_;
}


void OutputEncoder::initialize(const AVCodecParameters* left_audio_codec_parameters, const AVCodecParameters* right_audio_codec_parameters, double total_duration) {
    total_duration_ = total_duration;

    avformat_alloc_output_context2(&av_format_ctx_, NULL, NULL, filename_.c_str());
    if (!av_format_ctx_) {
        spdlog::error("Could not allocate format context");
        throw runtime_error("Could not allocate format context");
    }

    init_video();
    left_audio_stream_ = init_audio(left_audio_codec_parameters, "Left");
    right_audio_stream_ = init_audio(right_audio_codec_parameters, "Left");
    
    int32_t ret;
    ret = avio_open(&av_format_ctx_->pb, filename_.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        spdlog::error("Could not open output file: {}", filename_);
        throw runtime_error("Could not open output file.");
    }

    ret = avformat_write_header(av_format_ctx_, NULL);
    if (ret < 0) {
        spdlog::error("Error occurred when opening output file");
        throw runtime_error("Error occurred when opening output file");
    }

}

void OutputEncoder::init_video() {
    video_stream_ = avformat_new_stream(av_format_ctx_, NULL);
    if (!video_stream_) {
        spdlog::error("Could not allocate video stream");
        throw runtime_error("Could not allocate video stream");
    }

    video_codec_ = avcodec_find_encoder(AV_CODEC_ID_HEVC);
    // video_codec_ = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!video_codec_) {
        spdlog::error("Video codec not found");
        throw runtime_error("Video codec not found");
    }
  
    video_codec_ctx_ = avcodec_alloc_context3(video_codec_);
    if (!video_codec_ctx_) {
        spdlog::error("Could not allocate video codec context");
        throw runtime_error("Could not allocate video codec context");
    }

    AVDictionary *opt=NULL;
    av_dict_set(&opt, "x265-params", "pools=16", 0);
    av_dict_set(&opt, "crf", "24", 0);
    av_dict_set(&opt, "preset", "slow", 0);
    av_dict_set(&opt, "keyint", "600", 0);

    video_codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    AVRational tb;
    tb.num = video_time_base_.num;
    tb.den = video_time_base_.den;
    video_codec_ctx_->time_base = tb;
    video_codec_ctx_->gop_size = 600;
    video_codec_ctx_->framerate = (AVRational){60000, 1001};
    av_opt_set(video_codec_ctx_->priv_data, "preset", "slow", 0);
    av_opt_set(video_codec_ctx_->priv_data, "crf", "24", 0);
    video_codec_ctx_->thread_type = FF_THREAD_FRAME;
    video_codec_ctx_->thread_count = 1;


    if (av_format_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        video_codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    video_codec_ctx_->height = video_height_;
    video_codec_ctx_->width = video_width_;

    int32_t ret = avcodec_open2(video_codec_ctx_, video_codec_, &opt);
    if (ret < 0) {
        spdlog::error("Cannot open video encoder");
        throw runtime_error("Cannot open video encoder");;
    }
    ret = avcodec_parameters_from_context(video_stream_->codecpar, video_codec_ctx_);
    if (ret < 0) {
        spdlog::error("Failed to copy video encoder parameters to output stream");
        throw runtime_error("Failed to copy video encoder parameters to output stream");;
    }
  
    video_stream_->time_base = video_codec_ctx_->time_base;
    video_stream_->avg_frame_rate = video_codec_ctx_->framerate;
    if(spdlog::should_log(spdlog::level::debug))
        av_dump_format(av_format_ctx_, 0, filename_.c_str(), 1);

    video_frame_ = av_frame_alloc();
    if (!video_frame_) {
        spdlog::error("Could not allocate video frame");
        throw runtime_error("Could not allocate video frame");
    }
    video_frame_->format = video_codec_ctx_->pix_fmt;
    video_frame_->width  = video_codec_ctx_->width;
    video_frame_->height = video_codec_ctx_->height;
    av_image_fill_linesizes(video_frame_->linesize, video_codec_ctx_->pix_fmt, video_frame_->width);
  
    ret = av_frame_get_buffer(video_frame_, 0);
    if (ret < 0) {
        spdlog::error("Could not allocate the video frame data");
        throw runtime_error("Could not allocate the video frame data");
    }
}

AVStream* OutputEncoder::init_audio(const AVCodecParameters* audio_codec_parameters, const char* title) {
    char error_msg[AV_ERROR_MAX_STRING_SIZE];

    AVStream* audio_stream = avformat_new_stream(av_format_ctx_, NULL);
    if (!audio_stream) {
        spdlog::error("Could not allocate audio stream");
        throw runtime_error("Could not allocate audio stream");
    }
    int32_t ret = avcodec_parameters_copy(audio_stream->codecpar, audio_codec_parameters);
    if (ret < 0) {
        spdlog::error("Failed to copy codec parameters");
        throw runtime_error("Failed to copy codec parameters");
    }

    av_dict_set(&audio_stream->metadata, "title", title, 0);

    return audio_stream;
}


string sToHMS(double seconds) {
    stringstream buf;
    int hours, minutes;
    minutes = seconds / 60;
    hours = minutes / 60;

    buf << hours << ':' << int(minutes%60) << ':' << int((int)seconds%60);
    return buf.str();
}


void OutputEncoder::run() {
    #ifdef _GNU_SOURCE
    pthread_setname_np(pthread_self(), "OutputEncoder");
    #endif

    chrono::milliseconds audio_wait(chrono::milliseconds(1));
    chrono::milliseconds video_wait(chrono::milliseconds(100));

    AVPacket* video_pkt;

    video_pkt = av_packet_alloc();
    if (!video_pkt) {
        spdlog::error("Couldn't allocate video packet");
        throw runtime_error("Couldn't allocate video packet");
    }

    uint32_t frame_idx=0;
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    unordered_map<uint32_t, unique_ptr<PanoramicPacket>> video_packets;
    double prev_delta = 0;
    while(running_) {
        while(true) {
            unique_ptr<AVPacket, PacketDeleter> audio_packet(left_audio_packet_queue_.pop(audio_wait));
            if(audio_packet) {
                left_audio_packets_.push(move(audio_packet));
            } else {
                break;
            }
        }
        while(true) {
            unique_ptr<AVPacket, PacketDeleter> audio_packet(right_audio_packet_queue_.pop(audio_wait));
            if(audio_packet) {
                right_audio_packets_.push(move(audio_packet));
            } else {
                break;
            }
        }
        
        unique_ptr<PanoramicPacket> new_panoramic_packet(panoramic_packet_queue_.pop(video_wait));
        if(new_panoramic_packet) {
            uint32_t idx = new_panoramic_packet->idx;
            video_packets.insert(make_pair(idx, move(new_panoramic_packet)));
        }

        while(video_packets.find(frame_idx) != video_packets.end()) {
            unique_ptr<PanoramicPacket> current_panoramic_packet(move(video_packets[frame_idx]));
            video_packets.erase(frame_idx);
            // if(frame_idx == 0) {
            //     cv::Mat img_yuv(video_frame_->height *3/2, video_frame_->width, CV_8UC1, current_panoramic_packet->data.get());
            //     cv::Mat img;
            //     cv::cvtColor(img_yuv, img, cv::COLOR_YUV2BGR_I420);
            //     cv::imwrite("first_pano.png", img); 
            // }
            ++frame_idx;

            //raise(SIGINT);
            while(!left_audio_packets_.empty() && (left_audio_packets_.front()->pts * av_q2d(left_audio_stream_->time_base)) <= current_panoramic_packet->pts_time) {
                // Encode audio
                unique_ptr<AVPacket, PacketDeleter> audio_packet(move(left_audio_packets_.front()));
                write_audio(audio_packet.get(), left_audio_stream_);

                left_audio_packets_.pop();
                audio_packet.reset();
            }
            while(!right_audio_packets_.empty() && (right_audio_packets_.front()->pts * av_q2d(right_audio_stream_->time_base)) <= current_panoramic_packet->pts_time) {
                unique_ptr<AVPacket, PacketDeleter> audio_packet(move(right_audio_packets_.front()));
                write_audio(audio_packet.get(), right_audio_stream_);

                right_audio_packets_.pop();
                audio_packet.reset();
            }

            int32_t ret = av_frame_make_writable(video_frame_);
            if (ret < 0) {
                spdlog::error("Error making video frame writable");
                throw runtime_error("Error making video frame writable");
            }

            video_frame_->pts = current_panoramic_packet->pts;
            ret = av_image_fill_arrays(video_frame_->data, video_frame_->linesize, current_panoramic_packet->data.get(), video_codec_ctx_->pix_fmt, video_frame_->width, video_frame_->height, 1);
            if (ret < 0) {
                spdlog::error("Error filling image array");
                throw runtime_error("Error filling image array");
            }
            ret = avcodec_send_frame(video_codec_ctx_, video_frame_);
            if (ret < 0) {
                spdlog::error("Error sending a frame for encoding");
                throw runtime_error("Error sending a frame for encoding");
            }
      
            while (ret >= 0) {
                ret = avcodec_receive_packet(video_codec_ctx_, video_pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                else if (ret < 0) { 
                    spdlog::error("Error during encoding");
                    throw runtime_error("Error during encoding");
                }
                video_pkt->stream_index = video_stream_->index;
                ret = av_interleaved_write_frame(av_format_ctx_, video_pkt);
                if (ret < 0) {
                    spdlog::error("Couldn't mux packet");
                    throw runtime_error("Couldn't mux packet");
                }
                av_packet_unref(video_pkt);
            }

            current_panoramic_packet.reset();

            double delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();
            if((delta - prev_delta) > 60000.0) {
                prev_delta = delta;
                double current_duration = (double)video_frame_->pts * av_q2d(video_stream_->time_base);

                double pct_done = current_duration * 100.0 / (double)total_duration_;
                double fps = ((double)frame_idx/delta) * 1000.0; 

                double est_total_time = (double)delta * 100.0 / pct_done / 1000.0;
                string est_total_time_str(sToHMS(est_total_time));

                double eta = est_total_time*(100.0-pct_done) / 100.0;
                string eta_str(sToHMS(eta));
                spdlog::info("Output FPS: {:.2f} Done: {:.2f}% Est.Time: {} ETA: {}", fps, pct_done, est_total_time_str, eta_str);
            }
        } // while frame_idx
    }

    // Flush buffers
    int32_t ret;
    ret = avcodec_send_frame(video_codec_ctx_, nullptr);
    while (ret >= 0) {
        ret = avcodec_receive_packet(video_codec_ctx_, video_pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        else if (ret < 0) {
            spdlog::error("Error during encoding");
            throw runtime_error("Error during encoding");
        }
        video_pkt->stream_index = video_stream_->index;
        ret = av_interleaved_write_frame(av_format_ctx_, video_pkt);
        if (ret < 0) {
            spdlog::error("Couldn't mux packet");
            throw runtime_error("Couldn't mux packet");
        }
        av_packet_unref(video_pkt);
    }

    av_write_trailer(av_format_ctx_);

    av_packet_free(&video_pkt);

    avcodec_free_context(&video_codec_ctx_);
    video_codec_ = NULL;
    av_frame_free(&video_frame_);
    //av_free(video_stream_);
    //av_free(left_audio_stream_);
    //av_free(right_audio_stream_);

    avio_closep(&av_format_ctx_->pb);
    avformat_free_context(av_format_ctx_);

    done_ = true;
}

void OutputEncoder::write_audio(AVPacket* packet, AVStream* audio_stream) {
    packet->stream_index = audio_stream->index;
    int r = AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX;
    packet->pts = av_rescale_q_rnd(packet->pts, audio_time_base_, audio_stream->time_base, (AVRounding)r);
    packet->dts = av_rescale_q_rnd(packet->dts, audio_time_base_, audio_stream->time_base, (AVRounding)r);
    packet->duration = av_rescale_q(packet->duration, audio_time_base_, audio_stream->time_base);
    // https://ffmpeg.org/doxygen/trunk/structAVPacket.html#ab5793d8195cf4789dfb3913b7a693903
    packet->pos = -1;

    //https://ffmpeg.org/doxygen/trunk/group__lavf__encoding.html#ga37352ed2c63493c38219d935e71db6c1
    int32_t ret = av_interleaved_write_frame(av_format_ctx_, packet);
    if (ret < 0) {
        spdlog::error("Error muxing packet");
        throw runtime_error("Error muxing packet");
    }
}
