#include "outputencoder.hpp"

#include <stdexcept>
#include <iostream>
#include <csignal>
#include <spdlog/spdlog.h>

extern "C" {
  #include <libavutil/error.h>
  #include <libavutil/imgutils.h>
  #include <libavutil/opt.h>
}

using namespace std;

OutputEncoder::OutputEncoder(const string& filename, ThreadSafeQueue<AudioPacket>& left_audio_queue, ThreadSafeQueue<AudioPacket>& right_audio_queue, uint32_t width, uint32_t height, Rational video_time_base, Rational audio_time_base, uint32_t queue_size)
: filename_(filename), running_(false), done_(false), video_width_(width), video_height_(height), video_time_base_(video_time_base), audio_time_base_(audio_time_base), left_audio_packet_queue_(left_audio_queue), right_audio_packet_queue_(right_audio_queue), panoramic_packet_queue_(queue_size) {
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


void OutputEncoder::init_video(AVFormatContext* fmt_ctx, AVStream*& video_stream, const AVCodec*& video_codec, AVCodecContext*& video_codec_ctx, AVPacket*& video_pkt, AVFrame*& video_frame) {
    video_stream = avformat_new_stream(fmt_ctx, NULL);
    if (!video_stream) {
        spdlog::error("Could not allocate video stream");
        throw runtime_error("Could not allocate video stream");
    }

    video_codec = avcodec_find_encoder(AV_CODEC_ID_HEVC);
    if (!video_codec) {
        spdlog::error("Codec libx264 not found");
        throw runtime_error("Codec libx264 not found");
    }
  
    video_codec_ctx = avcodec_alloc_context3(video_codec);
    if (!video_codec_ctx) {
        spdlog::error("Could not allocate video codec context");
        throw runtime_error("Could not allocate video codec context");
    }

    video_codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    AVRational tb;
    tb.num = video_time_base_.num;
    tb.den = video_time_base_.den;
    video_codec_ctx->time_base = tb;
    av_opt_set(video_codec_ctx->priv_data, "preset", "fast", 0);
    av_opt_set(video_codec_ctx->priv_data, "crf", "23", 0);
    video_codec_ctx->thread_type = FF_THREAD_FRAME;
    video_codec_ctx->thread_count = 0;

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        video_codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    video_pkt = av_packet_alloc();
    if (!video_pkt) {
        spdlog::error("Couldn't allocate video packet");
        throw runtime_error("Couldn't allocate video packet");
    }

    video_codec_ctx->height = video_height_;
    video_codec_ctx->width = video_width_;

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
        spdlog::error("Could not allocate the video frame data");
        throw runtime_error("Could not allocate the video frame data");
    }

    ret = avio_open(&fmt_ctx->pb, filename_.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        spdlog::error("Could not open output file: {}", filename_);
        throw runtime_error("Could not open output file.");
    }
}

void OutputEncoder::init_audio(AVFormatContext* fmt_ctx, AVStream*& audio_stream, const AVCodec*& audio_codec, AVCodecContext*& audio_codec_ctx, AVPacket*& audio_pkt, AVFrame*& audio_frame, const char* title) {
    char error_msg[AV_ERROR_MAX_STRING_SIZE];

    audio_stream = avformat_new_stream(fmt_ctx, NULL);
    if (!audio_stream) {
        spdlog::error("Could not allocate audio stream");
        throw runtime_error("Could not allocate audio stream");
    }
    audio_codec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!audio_codec) {
        spdlog::error("Codec AAC not found");
        throw runtime_error("Codec AAC not found");
    }
  
    audio_codec_ctx = avcodec_alloc_context3(audio_codec);
    if (!audio_codec_ctx) {
        spdlog::error("Could not allocate audio codec context");
        throw runtime_error("Could not allocate audio codec context");
    }
  
    audio_codec_ctx->channels = 2;
    audio_codec_ctx->channel_layout = av_get_default_channel_layout(2);
    audio_codec_ctx->sample_rate = 48000;
    audio_codec_ctx->sample_fmt = AV_SAMPLE_FMT_FLTP;
    audio_codec_ctx->bit_rate = 128000;
    //audio_codec_ctx->strict_std_compliance = FF_COMPLIANCE_EXPERIMENTAL;

    audio_stream->time_base.den = audio_codec_ctx->sample_rate;
    audio_stream->time_base.num = 1;
    av_dict_set(&audio_stream->metadata, "title", title, 0);
  
    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        audio_codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
  
    audio_pkt = av_packet_alloc();
    if (!audio_pkt) {
        spdlog::error("Couldn't allocate audio packet");
        throw runtime_error("Couldn't allocate audio packet");
    }

    int32_t ret = avcodec_open2(audio_codec_ctx, audio_codec, NULL);
    if (ret < 0) {
        av_strerror(ret, error_msg, AV_ERROR_MAX_STRING_SIZE);
        spdlog::error("Could not open output audio codec (error '{}')", error_msg);
        throw runtime_error("Could not open output audio codec");
    }

    ret = avcodec_parameters_from_context(audio_stream->codecpar, audio_codec_ctx);
    if (ret < 0) {
        spdlog::error("Could not initialize stream parameters");
        throw runtime_error("Could not initialize stream parameters");
    }

    audio_pkt = av_packet_alloc();
    if (!audio_pkt) {
        spdlog::error("Couldn't allocate audio packet");
        throw runtime_error("Couldn't allocate audio packet");
    }

    if (!(audio_frame = av_frame_alloc())) {
        spdlog::error("Could not allocate output frame");
        throw runtime_error("Could not allocate output frame");
    }
  
    audio_frame->nb_samples     = 1024;
    audio_frame->channel_layout = audio_codec_ctx->channel_layout;
    audio_frame->format         = audio_codec_ctx->sample_fmt;
    audio_frame->sample_rate    = audio_codec_ctx->sample_rate;
  
    /* Allocate the samples of the created frame. This call will make
     * sure that the audio frame can hold as many samples as specified. */
    ret = av_frame_get_buffer(audio_frame, 0);
    if (ret < 0) {
        av_strerror(ret, error_msg, AV_ERROR_MAX_STRING_SIZE);
        spdlog::error("Could not allocate output frame samples (error '{}')", error_msg);
        av_frame_free(&audio_frame);
        throw runtime_error("Could not allocate output frame samples");
    }

}


void OutputEncoder::run() {
    #ifdef _GNU_SOURCE
    pthread_setname_np(pthread_self(), "OutputEncoder");
    #endif

    chrono::milliseconds audio_wait(chrono::milliseconds(1));
    chrono::milliseconds video_wait(chrono::milliseconds(100));

    AVFormatContext* fmt_ctx;
    const AVCodec* video_codec;
    const AVCodec* left_audio_codec;
    const AVCodec* right_audio_codec;
    AVCodecContext* video_codec_ctx = nullptr;
    AVCodecContext* left_audio_codec_ctx = nullptr;
    AVCodecContext* right_audio_codec_ctx = nullptr;
    AVStream* video_stream;
    AVStream* left_audio_stream;
    AVStream* right_audio_stream;
    AVFrame* video_frame;
    AVFrame* left_audio_frame;
    AVFrame* right_audio_frame;
    AVPacket* video_pkt;
    AVPacket* left_audio_pkt;
    AVPacket* right_audio_pkt;

    avformat_alloc_output_context2(&fmt_ctx, NULL, NULL, filename_.c_str());
    if (!fmt_ctx) {
        spdlog::error("Could not allocate format context");
        throw runtime_error("Could not allocate format context");
    }

    init_video(fmt_ctx, video_stream, video_codec, video_codec_ctx, video_pkt, video_frame);
    init_audio(fmt_ctx, left_audio_stream, left_audio_codec, left_audio_codec_ctx, left_audio_pkt, left_audio_frame, "Left");
    init_audio(fmt_ctx, right_audio_stream, right_audio_codec, right_audio_codec_ctx, right_audio_pkt, right_audio_frame, "Right");

    int32_t ret = avformat_write_header(fmt_ctx, NULL);
    if (ret < 0) {
        spdlog::error("Error occurred when opening output file");
        throw runtime_error("Error occurred when opening output file");
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
            while(!left_audio_packets_.empty() && left_audio_packets_.front()->pts_time <= panoramic_packet->pts_time) {
                // Encode audio
                unique_ptr<AudioPacket> audio_packet(move(left_audio_packets_.front()));
                left_audio_packets_.pop();

                raise(SIGINT);
                av_samples_copy(left_audio_frame->data, (uint8_t* const*)audio_packet->data.get(), 0, 0, left_audio_frame->nb_samples, left_audio_codec_ctx->channels, left_audio_codec_ctx->sample_fmt);
                int data_present=0;
                encode_audio_frame(left_audio_frame, fmt_ctx, left_audio_codec_ctx, &data_present);
                audio_packet.reset();
            }
            while(!right_audio_packets_.empty() && right_audio_packets_.front()->pts_time <= panoramic_packet->pts_time) {
                // Encode audio
                unique_ptr<AudioPacket> audio_packet(move(right_audio_packets_.front()));
                right_audio_packets_.pop();

                av_samples_copy(right_audio_frame->data, (uint8_t* const*)audio_packet->data.get(), 0, 0, right_audio_frame->nb_samples, right_audio_codec_ctx->channels, right_audio_codec_ctx->sample_fmt);
                int data_present=0;
                encode_audio_frame(right_audio_frame, fmt_ctx, right_audio_codec_ctx, &data_present);
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
                video_pkt->stream_index = video_stream->index;
                ret = av_interleaved_write_frame(fmt_ctx, video_pkt);
                if (ret < 0) {
                    spdlog::error("Couldn't mux packet");
                    throw runtime_error("Couldn't mux packet");
                }
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

    // Flush buffers
    ret = avcodec_send_frame(video_codec_ctx, nullptr);
    while (ret >= 0) {
        ret = avcodec_receive_packet(video_codec_ctx, video_pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            break;
        else if (ret < 0) {
            spdlog::error("Error during encoding");
            throw runtime_error("Error during encoding");
        }
        video_pkt->stream_index = video_stream->index;
        ret = av_interleaved_write_frame(fmt_ctx, video_pkt);
        if (ret < 0) {
            spdlog::error("Couldn't mux packet");
            throw runtime_error("Couldn't mux packet");
        }
        av_packet_unref(video_pkt);
    }

    av_write_trailer(fmt_ctx);

    av_packet_free(&video_pkt);
    av_packet_free(&left_audio_pkt);
    av_packet_free(&right_audio_pkt);

    avcodec_free_context(&video_codec_ctx);
    avcodec_free_context(&left_audio_codec_ctx);
    avcodec_free_context(&right_audio_codec_ctx);

    av_frame_free(&video_frame);
    av_frame_free(&left_audio_frame);
    av_frame_free(&right_audio_frame);

    av_free(video_stream);
    av_free(left_audio_stream);
    av_free(right_audio_stream);

    avio_closep(&fmt_ctx->pb);
    avformat_free_context(fmt_ctx);

    done_ = true;
}


int OutputEncoder::encode_audio_frame(AVFrame *frame, AVFormatContext *output_format_context, AVCodecContext *output_codec_context, int *data_present) {
    AVPacket *output_packet;
    char error_msg[AV_ERROR_MAX_STRING_SIZE];
    int error;
  
    output_packet = av_packet_alloc();
    if (!output_packet) {
        spdlog::error("Could not allocate packet");
        throw runtime_error("Could not allocate packet");
    }

    error = avcodec_send_frame(output_codec_context, frame);
    if (error == AVERROR_EOF) {
        error = 0;
    } else if (error < 0) {
        av_strerror(error, error_msg, AV_ERROR_MAX_STRING_SIZE);
        spdlog::error("Could not send packet for encoding (error '{}')", error_msg);
    } else {
        error = avcodec_receive_packet(output_codec_context, output_packet);
        if (error == AVERROR(EAGAIN)) {
            error = 0;
        } else if (error == AVERROR_EOF) {
            error = 0;
        } else if (error < 0) {
            av_strerror(error, error_msg, AV_ERROR_MAX_STRING_SIZE);
            spdlog::error("Could not encode frame (error '{}')", error_msg);
        } else {
            *data_present = 1;
        }
  
        /* Write one audio frame from the temporary packet to the output file. */
        if (*data_present && (error = av_write_frame(output_format_context, output_packet)) < 0) {
            av_strerror(error, error_msg, AV_ERROR_MAX_STRING_SIZE);
            spdlog::error("Could not write frame (error '{}')", error_msg);
        }
    }
    av_packet_free(&output_packet);
    return error;
}
