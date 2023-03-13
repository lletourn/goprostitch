#include "inputprocessor.hpp"

#include <stdexcept>
#include <iostream>
#include <csignal>
#include <spdlog/spdlog.h>

extern "C" {
  #include <libavutil/error.h>
  #include <libavutil/imgutils.h>
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
  #include <libswscale/swscale.h>
}
/*
// compatibility with newer API
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55,28,1)
#define av_frame_alloc avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#endif
*/

using namespace std;

InputProcessor::InputProcessor(const string& filename, uint32_t queue_size)
: filename_(filename), timecode_(0), running_(false), done_(false), packet_queue_(queue_size) {
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

ThreadSafeQueue<InputPacket>& InputProcessor::getOutQueue() {
    return packet_queue_;
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
        }
        else if (av_format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream = i;
        }
    }

    AVCodec * video_codec = NULL;
    video_codec = avcodec_find_decoder(av_format_ctx_->streams[video_stream]->codecpar->codec_id);
    if (video_codec == NULL) {
        cerr << "Unsupported video codec!" << endl;
        throw runtime_error("Error occured");
    }
    AVCodec * audio_codec = NULL;
    audio_codec = avcodec_find_decoder(av_format_ctx_->streams[audio_stream]->codecpar->codec_id);
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

    AVFrame * frame_BGR = nullptr;
    frame_BGR = av_frame_alloc();
    if (frame_BGR == nullptr) {
        cerr << "Could not allocate frame" << endl;
        throw runtime_error("Error occured");
    }

    AVFrame *audio_frame = nullptr;
    audio_frame = av_frame_alloc();
    if (audio_frame == nullptr) {
        cerr << "Could not allocate audio frame." << endl;
        throw runtime_error("Error occured");
    }

    uint8_t* buffer = nullptr;
    int nb_bytes;

    nb_bytes = av_image_get_buffer_size(AV_PIX_FMT_BGR24, video_codec_ctx->width, video_codec_ctx->height, 32);
    buffer = new uint8_t[nb_bytes];

    av_image_fill_arrays(
        frame_BGR->data,
        frame_BGR->linesize,
        buffer,
        AV_PIX_FMT_BGR24,
        video_codec_ctx->width,
        video_codec_ctx->height,
        32
    );

    struct SwsContext* sws_ctx = nullptr;
    AVPacket* packet = av_packet_alloc();
    if (packet == NULL) {
        cerr << "Could not alloc packet," << endl;
        throw runtime_error("Error occured");
    }
    sws_ctx = sws_getContext(   // [13]
        video_codec_ctx->width,
        video_codec_ctx->height,
        video_codec_ctx->pix_fmt,
        video_codec_ctx->width,
        video_codec_ctx->height,
        AV_PIX_FMT_BGR24,
        SWS_BILINEAR,
        NULL,
        NULL,
        NULL
    );

    uint32_t video_idx = 0;
    uint32_t audio_idx = 0;
    while (av_read_frame(av_format_ctx_, packet) >= 0) {
        if (packet->stream_index == video_stream) {
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

                // Convert the image from its native format to BGR
                sws_scale(
                    sws_ctx,
                    (uint8_t const * const *)video_frame->data,
                    video_frame->linesize,
                    0,
                    video_codec_ctx->height,
                    frame_BGR->data,
                    frame_BGR->linesize
                );

                uint32_t data_size = av_image_get_buffer_size(AV_PIX_FMT_BGR24, video_codec_ctx->width, video_codec_ctx->height, 1);
                unique_ptr<uint8_t> data(new uint8_t[data_size]);
                av_image_copy_to_buffer(data.get(), data_size, frame_BGR->data, frame_BGR->linesize, AV_PIX_FMT_BGR24, video_codec_ctx->width, video_codec_ctx->height, 1);

                unique_ptr<InputPacket> input_packet(new InputPacket);
                input_packet->type = InputPacket::InputPacketType::VIDEO;
                input_packet->data_size = data_size;
                input_packet->data = move(data);
                input_packet->width = video_codec_ctx->width;
                input_packet->height = video_codec_ctx->height;
                input_packet->pts = video_frame->pts;
                input_packet->pts_time =  video_frame->pts * av_q2d(av_format_ctx_->streams[video_stream]->time_base);
                input_packet->idx = video_idx;
                ++video_idx;

                packet_queue_.push(move(input_packet));
            }
        } else if (packet->stream_index == audio_stream) {
            ret = avcodec_send_packet(audio_codec_ctx, packet);
            if (ret != AVERROR(EAGAIN) && ret < 0) {
                av_strerror(ret, error, ERROR_SIZE);
                cerr << "Error sending audio packet for decoding: " << error << endl;
                throw runtime_error("Error occured");
            }
            while (ret >= 0) {
                ret = avcodec_receive_frame(video_codec_ctx, audio_frame);
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
                memcpy(audio_buf.get(), audio_frame->data[0], data_size);
                
                // const uint32_t channel_name_buf_size = 128;
                // char* channel_name_buf = new char[channel_name_buf_size];
                // uint64_t channel_name_actual_size = av_channel_layout_describe (audio_frame->ch_layout, channel_name_buf, channel_name_buf_size)
                // string channel_name(channel_name_buf, channel_name_actual_size);
                // delete[] channel_name_buf;

                unique_ptr<InputPacket> input_packet(new InputPacket);
                input_packet->type = InputPacket::InputPacketType::AUDIO;
                input_packet->data_size = data_size;
                input_packet->data = move(audio_buf);
                input_packet->pts = audio_frame->pts;
                input_packet->pts_time =  audio_frame->pts * av_q2d(av_format_ctx_->streams[audio_stream]->time_base);
                input_packet->idx = audio_idx;
                input_packet->format = audio_frame->format;
                input_packet->layout = audio_frame->channel_layout;
                input_packet->sample_rate = audio_frame->sample_rate;
                ++audio_idx;

                packet_queue_.push(move(input_packet));
            }

        }
        av_packet_unref(packet);
    }

    // Free the RGB image
    av_free(buffer);
    av_frame_free(&frame_BGR);
    av_free(frame_BGR);

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
