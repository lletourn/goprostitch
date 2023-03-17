#include "outputencoder.hpp"

#include <stdexcept>
#include <iostream>
#include <csignal>
#include <spdlog/spdlog.h>

extern "C" {
  #include <libavutil/error.h>
  #include <libavutil/imgutils.h>
  #include <libavutil/opt.h>
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
  #include <libswscale/swscale.h>
}

using namespace std;

OutputEncoder::OutputEncoder(const string& filename, ThreadSafeQueue<AudioPacket>& left_audio_queue, ThreadSafeQueue<AudioPacket>& right_audio_queue, Rational video_time_base, Rational audio_time_base, uint32_t queue_size)
: filename_(filename), running_(false), done_(false), video_time_base_(video_time_base), audio_time_base_(audio_time_base), left_audio_packet_queue_(left_audio_queue), right_audio_packet_queue_(right_audio_queue), panoramic_packet_queue_(queue_size) {
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

void OutputEncoder::run() {
    #ifdef _GNU_SOURCE
    pthread_setname_np(pthread_self(), "OutputEncoder");
    #endif

    chrono::milliseconds audio_wait(chrono::milliseconds(1));
    chrono::milliseconds video_wait(chrono::milliseconds(100));

    AVFormatContext* fmt_ctx;
    const AVCodec* video_codec;
    AVCodecContext* video_codec_ctx = nullptr;
    AVCodecContext* audio_codec_ctx = nullptr;
    AVStream* video_stream;
    AVStream* left_audio_stream;
    AVStream* right_audio_stream;
    AVFrame* video_frame;
    AVFrame* audio_frame;
    AVPacket* video_pkt = av_packet_alloc();
    AVPacket* audio_pkt = av_packet_alloc();;

    avformat_alloc_output_context2(&fmt_ctx, NULL, NULL, filename_.c_str());
    if (!fmt_ctx) {
        spdlog::error("Could not allocate format context");
        throw runtime_error("Could not allocate format context");
    }

    video_stream = avformat_new_stream(fmt_ctx, NULL);
    if (!video_stream) {
        spdlog::error("Could not allocate video stream");
        throw runtime_error("Could not allocate video stream");
    }
    left_audio_stream = avformat_new_stream(fmt_ctx, NULL);
    if (!left_audio_stream) {
        spdlog::error("Could not allocate left audio stream");
        throw runtime_error("Could not allocate left audio stream");
    }
    right_audio_stream = avformat_new_stream(fmt_ctx, NULL);
    if (!right_audio_stream) {
        spdlog::error("Could not allocate right audio stream");
        throw runtime_error("Could not allocate right audio stream");
    }

    video_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!video_codec) {
        spdlog::error("Codec libx264 not found");
        throw runtime_error("Codec libx264 not found");
    }
  
    video_codec_ctx = avcodec_alloc_context3(video_codec);
    if (!video_codec_ctx) {
        spdlog::error("Could not allocate video codec context");
        throw runtime_error("Could not allocate video codec context");
    }

    spdlog::info("Out Pix FMT: {}", (uint32_t)video_codec->pix_fmts[0]);
    video_codec_ctx->pix_fmt = video_codec->pix_fmts[0];
    AVRational tb;
    tb.num = video_time_base_.num;
    tb.den = video_time_base_.den;
    video_codec_ctx->time_base = tb;
    av_opt_set(video_codec_ctx->priv_data, "preset", "fast", 0);
    av_opt_set(video_codec_ctx->priv_data, "crf", "23", 0);

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        video_codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }


    video_pkt = av_packet_alloc();
    if (!video_pkt) {
        spdlog::error("Couldn't allocate video packet");
        throw runtime_error("Couldn't allocate video packet");
    }


    uint32_t frame_idx=0;
    double prev_fps = -1;
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    while(running_) {
        while(true) {
            unique_ptr<AudioPacket> audio_packet(left_audio_packet_queue_.pop(audio_wait));
            if(audio_packet) {
                left_audio_packets_.push(move(audio_packet));
            } else {
                break;
            }
        }
        while(true) {
            unique_ptr<AudioPacket> audio_packet(right_audio_packet_queue_.pop(audio_wait));
            if(audio_packet) {
                right_audio_packets_.push(move(audio_packet));
            } else {
                break;
            }
        }
        
        unique_ptr<PanoramicPacket> panoramic_packet(panoramic_packet_queue_.pop(video_wait));
        if(panoramic_packet) {
            // Finish setup
            if (frame_idx == 0) {
                
                video_codec_ctx->height = panoramic_packet->height;
                video_codec_ctx->width = panoramic_packet->width;

                int32_t ret = avcodec_open2(video_codec_ctx, video_codec, NULL);
                if (ret < 0) {
                    spdlog::error("Cannot open video encoder");
                    throw runtime_error("Cannot open video encoder");;
                }
                ret = avcodec_parameters_from_context(video_stream->codecpar, video_codec_ctx);
                if (ret < 0) {
                    spdlog::error("Failed to copy video encoder parameters to output stream");
                    throw runtime_error("Failed to copy video encoder parameters to output stream");;
                }
  
                video_stream->time_base = video_codec_ctx->time_base;
                if(spdlog::should_log(spdlog::level::debug))
                    av_dump_format(fmt_ctx, 0, filename_.c_str(), 1);

                video_frame = av_frame_alloc();
                if (!video_frame) {
                    spdlog::error("Could not allocate video frame");
                    throw runtime_error("Could not allocate video frame");
                }
                video_frame->format = video_codec_ctx->pix_fmt;
                video_frame->width  = video_codec_ctx->width;
                video_frame->height = video_codec_ctx->height;
                av_image_fill_linesizes(video_frame->linesize, video_codec_ctx->pix_fmt, video_frame->width);
  
                ret = av_frame_get_buffer(video_frame, 0);
                if (ret < 0) {
                    fprintf(stderr, "Could not allocate the video frame data\n");
                    exit(1);
                }
            }

            while(!left_audio_packets_.empty() && left_audio_packets_.front()->pts_time <= panoramic_packet->pts_time) {
                // Encode audio
                unique_ptr<AudioPacket> audio_packet(move(left_audio_packets_.front()));
                left_audio_packets_.pop();
                audio_packet.reset();
            }
            while(!right_audio_packets_.empty() && right_audio_packets_.front()->pts_time <= panoramic_packet->pts_time) {
                // Encode audio
                unique_ptr<AudioPacket> audio_packet(move(right_audio_packets_.front()));
                right_audio_packets_.pop();
                audio_packet.reset();
            }

            int32_t ret = av_frame_make_writable(video_frame);
            if (ret < 0) {
                spdlog::error("Error making video frame writable");
                throw runtime_error("Error making video frame writable");
            }
            ret = av_image_fill_arrays(video_frame->data, video_frame->linesize, panoramic_packet->data.get(), video_codec_ctx->pix_fmt, video_frame->width, video_frame->height, 1);
            if (ret < 0) {
                spdlog::error("Error filling image array");
                throw runtime_error("Error filling image array");
            }
            ret = avcodec_send_frame(video_codec_ctx, video_frame);
            if (ret < 0) {
                spdlog::error("Error sending a frame for encoding");
                throw runtime_error("Error sending a frame for encoding");
            }
      
            while (ret >= 0) {
                ret = avcodec_receive_packet(video_codec_ctx, video_pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                else if (ret < 0) { 
                    spdlog::error("Error during encoding");
                    throw runtime_error("Error during encoding");
                }
  
                //`fwrite(video_pkt->data, 1, video_pkt->size, outfile);
                av_packet_unref(video_pkt);
            }

            panoramic_packet.reset();
            ++frame_idx;

            auto delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();
            double fps = ((double)frame_idx/delta) * 1000.0; 
            if(fps != prev_fps) {
                spdlog::debug("FPS: {}", fps);
            }
        }
    }

    done_ = true;
}
