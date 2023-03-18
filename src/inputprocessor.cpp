#include "inputprocessor.hpp"

#include <stdexcept>
#include <iostream>
#include <spdlog/spdlog.h>

extern "C" {
  #include <libavutil/error.h>
  #include <libavutil/imgutils.h>
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
  #include <libswscale/swscale.h>
}

using namespace std;

InputProcessor::InputProcessor(const string& filename, uint32_t video_queue_size, uint32_t audio_queue_size)
: filename_(filename), timecode_(0), running_(false), done_(false), video_packet_queue_(video_queue_size), audio_packet_queue_(audio_queue_size), video_time_base_(Rational(0,0)), audio_time_base_(Rational(0,0)) {
}

InputProcessor::~InputProcessor() {
}

void InputProcessor::start() {
    if (thread_.joinable()) {
        return;
    }
    running_ = true;
    done_ = false;
    thread_ = move(thread(&InputProcessor::run, this));
}

void InputProcessor::stop() {
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

bool InputProcessor::is_done() {
    return done_.load();
} 

ThreadSafeQueue<VideoPacket>& InputProcessor::getOutVideoQueue() {
    return video_packet_queue_;
}

ThreadSafeQueue<AudioPacket>& InputProcessor::getOutAudioQueue() {
    return audio_packet_queue_;
}


void InputProcessor::run() {
    #ifdef _GNU_SOURCE
    pthread_setname_np(pthread_self(), "InputProcessor");
    #endif
   

    const uint32_t ERROR_SIZE = 256;
    char error[ERROR_SIZE];
    AVFormatContext* av_format_ctx_;

    av_format_ctx_ = avformat_alloc_context();
    if (!av_format_ctx_) {
        spdlog::error("Could not allocate context.");
        throw runtime_error("Error occured");
    }

    int ret = avformat_open_input(&av_format_ctx_, filename_.c_str(), NULL, NULL);
    if (ret < 0) {
        // couldn't open file
        cerr << "Could not open file " << filename_ << endl;
        throw runtime_error("Error occured");
    }

    ret = avformat_find_stream_info(av_format_ctx_, NULL);
    if (ret < 0) {
        cerr << "Could not find stream information " << filename_ << endl;
        throw runtime_error("Error occured");
    }
    if(spdlog::should_log(spdlog::level::debug))
        av_dump_format(av_format_ctx_, 0, filename_.c_str(), 0);

    AVCodecContext* video_codec_ctx_orig = NULL;
    AVCodecContext* video_codec_ctx = NULL;
    AVCodecContext* audio_codec_ctx_orig = NULL;
    AVCodecContext* audio_codec_ctx = NULL;

    int video_stream = -1;
    int audio_stream = -1;
    for (uint32_t i = 0; i < av_format_ctx_->nb_streams; i++) {
        if (av_format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream = i;

            if(av_format_ctx_->streams[i]->codecpar->sample_aspect_ratio.num != 1 || av_format_ctx_->streams[i]->codecpar->sample_aspect_ratio.den != 1) {
                spdlog::error("SAR is not 1:1 for input video");
                throw runtime_error("SAR is not 1:1 for input video");
            }

            video_time_base_ = Rational(av_format_ctx_->streams[i]->time_base.num, av_format_ctx_->streams[i]->time_base.den);
        }
        else if (av_format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream = i;
            audio_time_base_ = Rational(av_format_ctx_->streams[i]->time_base.num, av_format_ctx_->streams[i]->time_base.den);
        }
    }

    const AVCodec* video_codec = avcodec_find_decoder(av_format_ctx_->streams[video_stream]->codecpar->codec_id);
    if (video_codec == NULL) {
        cerr << "Unsupported video codec!" << endl;
        throw runtime_error("Error occured");
    }
    const AVCodec* audio_codec = avcodec_find_decoder(av_format_ctx_->streams[audio_stream]->codecpar->codec_id);
    if (audio_codec == NULL) {
        cerr << "Unsupported audio codec!" << endl;
        throw runtime_error("Error occured");
    }

    video_codec_ctx_orig = avcodec_alloc_context3(video_codec);
    ret = avcodec_parameters_to_context(video_codec_ctx_orig, av_format_ctx_->streams[video_stream]->codecpar);
    audio_codec_ctx_orig = avcodec_alloc_context3(audio_codec);
    ret = avcodec_parameters_to_context(audio_codec_ctx_orig, av_format_ctx_->streams[audio_stream]->codecpar);

    video_codec_ctx = avcodec_alloc_context3(video_codec);
    ret = avcodec_parameters_to_context(video_codec_ctx, av_format_ctx_->streams[video_stream]->codecpar);
    if (ret != 0) {
        cerr << "Could not copy video codec context." << endl;
        throw runtime_error("Error occured");
    }
    video_codec_ctx->thread_type = FF_THREAD_FRAME;
    video_codec_ctx->thread_count = 0;
    spdlog::info("SAR: {} / {}", video_codec_ctx->sample_aspect_ratio.num, video_codec_ctx->sample_aspect_ratio.den);

    audio_codec_ctx = avcodec_alloc_context3(audio_codec);
    ret = avcodec_parameters_to_context(audio_codec_ctx, av_format_ctx_->streams[audio_stream]->codecpar);
    if (ret != 0) {
        cerr << "Could not copy audio codec context." << endl;
        throw runtime_error("Error occured");
    }

    ret = avcodec_open2(video_codec_ctx, video_codec, NULL);
    if (ret < 0) {
        printf("Could not open video codec.\n");
        throw runtime_error("Error occured");
    }
    ret = avcodec_open2(audio_codec_ctx, audio_codec, NULL);
    if (ret < 0) {
        printf("Could not open audio codec.\n");
        throw runtime_error("Error occured");
    }

    AVFrame *video_frame = nullptr;
    video_frame = av_frame_alloc();
    if (video_frame == nullptr) {
        cerr << "Could not allocate video frame." << endl;
        throw runtime_error("Error occured");
    }

    AVFrame *audio_frame = nullptr;
    audio_frame = av_frame_alloc();
    if (audio_frame == nullptr) {
        cerr << "Could not allocate audio frame." << endl;
        throw runtime_error("Error occured");
    }

    AVPacket* packet = av_packet_alloc();
    if (packet == NULL) {
        cerr << "Could not alloc packet," << endl;
        throw runtime_error("Error occured");
    }

    uint32_t video_idx = 0;
    uint32_t audio_idx = 0;
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    while (av_read_frame(av_format_ctx_, packet) >= 0) {
        if (packet->stream_index == video_stream) {
            chrono::steady_clock::time_point video_start = chrono::steady_clock::now();
            ret = avcodec_send_packet(video_codec_ctx, packet);
            if (ret < 0) {
                cerr << "Error sending video packet for decoding." << endl;
                throw runtime_error("Error occured");
            }

            while (ret >= 0) {
                ret = avcodec_receive_frame(video_codec_ctx, video_frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    cerr << "Error while decoding." << endl;
                    throw runtime_error("Error occured");
                }

                uint32_t data_size = av_image_get_buffer_size(video_codec_ctx->pix_fmt, video_codec_ctx->width, video_codec_ctx->height, 1);
                unique_ptr<uint8_t> data(new uint8_t[data_size]);
                av_image_copy_to_buffer(data.get(), data_size, video_frame->data, video_frame->linesize, video_codec_ctx->pix_fmt, video_codec_ctx->width, video_codec_ctx->height, 1);

                unique_ptr<VideoPacket> input_packet(new VideoPacket);
                input_packet->width = video_codec_ctx->width;
                input_packet->height = video_codec_ctx->height;
                input_packet->pts = video_frame->pts;
                input_packet->pts_time =  video_frame->pts * av_q2d(av_format_ctx_->streams[video_stream]->time_base);
                input_packet->idx = video_idx;
                input_packet->data_size = data_size;
                input_packet->data = move(data);
                ++video_idx;

                auto delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - video_start).count();
                double video_fps = (1.0/delta) * 1000.0;
 
                video_packet_queue_.push(move(input_packet));
                // delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();
                // double all_fps = ((double)video_idx/delta) * 1000.0;
                // spdlog::debug("Input FPS: {} All FPS: {}", video_fps, all_fps);
            }
        } else if (packet->stream_index == audio_stream) {
            ret = avcodec_send_packet(audio_codec_ctx, packet);
            if (ret != AVERROR(EAGAIN) && ret < 0) {
                av_strerror(ret, error, ERROR_SIZE);
                cerr << "Error sending audio packet for decoding: " << error << endl;
                throw runtime_error("Error occured");
            }
            while (ret >= 0) {
                ret = avcodec_receive_frame(audio_codec_ctx, audio_frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    av_strerror(ret, error, ERROR_SIZE);
                    cerr << "Error while decoding: " << error << endl;
                    throw runtime_error("Error occured");
                }
                uint32_t data_size = av_samples_get_buffer_size(
                    NULL,
                    audio_codec_ctx->channels,
                    audio_frame->nb_samples,
                    audio_codec_ctx->sample_fmt,
                    1
                );

                unique_ptr<uint8_t> audio_buf(new uint8_t[data_size]);
                memcpy(audio_buf.get(), (uint8_t*)audio_frame->data, data_size);
                
                // const uint32_t channel_name_buf_size = 128;
                // char* channel_name_buf = new char[channel_name_buf_size];
                // uint64_t channel_name_actual_size = av_channel_layout_describe (audio_frame->ch_layout, channel_name_buf, channel_name_buf_size)
                // string channel_name(channel_name_buf, channel_name_actual_size);
                // delete[] channel_name_buf;

                unique_ptr<AudioPacket> input_packet(new AudioPacket);
                input_packet->pts = audio_frame->pts;
                input_packet->pts_time =  audio_frame->pts * av_q2d(av_format_ctx_->streams[audio_stream]->time_base);
                input_packet->idx = audio_idx;
                input_packet->format = audio_frame->format;
                input_packet->layout = audio_frame->channel_layout;
                input_packet->sample_rate = audio_frame->sample_rate;
                input_packet->data_size = data_size;
                input_packet->data = move(audio_buf);
                ++audio_idx;

                audio_packet_queue_.push(move(input_packet));
            }
        }
        av_packet_unref(packet);
    }

    // Free the YUV frame
    av_frame_free(&video_frame);
    av_free(video_frame);
    av_frame_free(&audio_frame);
    av_free(audio_frame);

    // Close the codecs
    avcodec_close(video_codec_ctx);
    avcodec_close(video_codec_ctx_orig);
    avcodec_close(audio_codec_ctx);
    avcodec_close(audio_codec_ctx_orig);

    // Close the video file
    avformat_close_input(&av_format_ctx_);

    done_ = true;
}
