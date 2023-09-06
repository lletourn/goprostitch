#include "inputprocessor.hpp"

#include <stdexcept>
#include <spdlog/spdlog.h>

extern "C" {
  #include <libavutil/error.h>
  #include <libavutil/imgutils.h>
  #include <libswscale/swscale.h>
}

using namespace std;

InputProcessor::InputProcessor(const string& filename, uint32_t offset, uint32_t queue_size)
: filename_(filename), offset_(offset), timecode_(0), running_(false), done_(false), video_packet_queue_(queue_size), audio_packet_queue_(queue_size), video_time_base_(Rational(0,0)), audio_time_base_(Rational(0,0)) {
    av_format_ctx_ = NULL;
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

ThreadSafeQueue<AVPacket, PacketDeleter>& InputProcessor::getOutAudioQueue() {
    return audio_packet_queue_;
}

void InputProcessor::initialize() {
    char error_msg[AV_ERROR_MAX_STRING_SIZE];
    if(av_format_ctx_ != NULL) {
        throw runtime_error("Input processor already initialized");
    }

    av_format_ctx_ = avformat_alloc_context();
    if (!av_format_ctx_) {
        spdlog::error("Could not allocate context.");
        throw runtime_error("Error occured");
    }

    int ret = avformat_open_input(&av_format_ctx_, filename_.c_str(), NULL, NULL);
    if (ret < 0) {
        // couldn't open file
        spdlog::error("Could not open file: {} ", filename_);
        throw runtime_error("Could not open file");
    }

    ret = avformat_find_stream_info(av_format_ctx_, NULL);
    if (ret < 0) {
        spdlog::error("Could not find stream information: {}", filename_);
        throw runtime_error("Could not find stream information");
    }
    if(spdlog::should_log(spdlog::level::debug))
        av_dump_format(av_format_ctx_, 0, filename_.c_str(), 0);

    for (uint32_t i = 0; i < av_format_ctx_->nb_streams; i++) {
        if (av_format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_ = i;

            duration_ = (double)av_format_ctx_->streams[i]->duration * av_q2d(av_format_ctx_->streams[i]->time_base);
            if(av_format_ctx_->streams[i]->codecpar->sample_aspect_ratio.num != 1 || av_format_ctx_->streams[i]->codecpar->sample_aspect_ratio.den != 1) {
                spdlog::error("SAR is not 1:1 for input video");
                throw runtime_error("SAR is not 1:1 for input video");
            }

            video_time_base_ = Rational(av_format_ctx_->streams[i]->time_base.num, av_format_ctx_->streams[i]->time_base.den);
        }
        else if (av_format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_ = i;
            audio_time_base_ = Rational(av_format_ctx_->streams[i]->time_base.num, av_format_ctx_->streams[i]->time_base.den);
        }
    }

    video_codec_ = avcodec_find_decoder(av_format_ctx_->streams[video_stream_]->codecpar->codec_id);
    if (video_codec_ == NULL) {
        spdlog::error("Unsupported video codec!");
        throw runtime_error("Unsupported video codec");
    }

    video_codec_ctx_orig_ = avcodec_alloc_context3(video_codec_);
    ret = avcodec_parameters_to_context(video_codec_ctx_orig_, av_format_ctx_->streams[video_stream_]->codecpar);

    video_codec_ctx_ = avcodec_alloc_context3(video_codec_);
    ret = avcodec_parameters_to_context(video_codec_ctx_, av_format_ctx_->streams[video_stream_]->codecpar);
    if (ret != 0) {
        spdlog::error("Could not copy video codec context.");
        throw runtime_error("Error occured");
    }
    video_codec_ctx_->thread_type = FF_THREAD_FRAME;
    video_codec_ctx_->thread_count = 1;
    spdlog::info("SAR: {} / {}", video_codec_ctx_->sample_aspect_ratio.num, video_codec_ctx_->sample_aspect_ratio.den);

    ret = avcodec_open2(video_codec_ctx_, video_codec_, NULL);
    if (ret < 0) {
        spdlog::error("Could not open video codec.\n");
        throw runtime_error("Error occured");
    }
}

void InputProcessor::run() {
    #ifdef _GNU_SOURCE
    pthread_setname_np(pthread_self(), "InputProcessor");
    #endif
   
    char error_msg[AV_ERROR_MAX_STRING_SIZE];

    AVFrame *video_frame = nullptr;
    video_frame = av_frame_alloc();
    if (video_frame == nullptr) {
        spdlog::error("Could not allocate video frame.");
        throw runtime_error("Error occured");
    }

    AVPacket* packet = av_packet_alloc();
    if (packet == NULL) {
        spdlog::error("Could not alloc packet,");
        throw runtime_error("Error occured");
    }
    uint32_t video_idx = 0;
    int32_t ret;
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    while (av_read_frame(av_format_ctx_, packet) >= 0 && running_.load() == true) {
        if (packet->stream_index == video_stream_) {
            chrono::steady_clock::time_point video_start = chrono::steady_clock::now();
            ret = avcodec_send_packet(video_codec_ctx_, packet);
            if (ret < 0) {
                spdlog::error("Error sending video packet for decoding.");
                throw runtime_error("Error occured");
            }

            while (ret >= 0) {
                ret = avcodec_receive_frame(video_codec_ctx_, video_frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    cerr << "Error while decoding." << endl;
                    throw runtime_error("Error occured");
                }

                if(video_idx >= offset_) {
                    uint32_t data_size = av_image_get_buffer_size(video_codec_ctx_->pix_fmt, video_codec_ctx_->width, video_codec_ctx_->height, 1);
                    unique_ptr<uint8_t[]> data(new uint8_t[data_size]);
                    av_image_copy_to_buffer(data.get(), data_size, video_frame->data, video_frame->linesize, video_codec_ctx_->pix_fmt, video_codec_ctx_->width, video_codec_ctx_->height, 1);

                    unique_ptr<VideoPacket> input_packet(new VideoPacket);
                    input_packet->width = video_codec_ctx_->width;
                    input_packet->height = video_codec_ctx_->height;
                    input_packet->pts = video_frame->pts;
                    input_packet->pts_time =  video_frame->pts * av_q2d(av_format_ctx_->streams[video_stream_]->time_base);
                    input_packet->idx = video_idx-offset_;
                    input_packet->data_size = data_size;
                    input_packet->data = move(data);

                    auto delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - video_start).count();
                    double video_fps = (1.0/delta) * 1000.0;
 
                    video_packet_queue_.push(move(input_packet));
                    // delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();
                    // double all_fps = ((double)video_idx/delta) * 1000.0;
                    // spdlog::debug("Input FPS: {} All FPS: {}", video_fps, all_fps);
                }
                ++video_idx;
            }
        } else if (packet->stream_index == audio_stream_) {
            AVPacket* audio_packet = av_packet_alloc();
            if (audio_packet == NULL) {
                spdlog::error("Could not alloc packet,");
                throw runtime_error("Error occured");
            }
            av_packet_move_ref(audio_packet, packet);
            avpacket_unique_ptr p(audio_packet);
            audio_packet_queue_.push(move(p));
        }
        av_packet_unref(packet);
    }

    av_packet_free(&packet);

    // Free the YUV frame
    av_frame_free(&video_frame);
    av_free(video_frame);

    // Close the codecs
    avcodec_free_context(&video_codec_ctx_);
    avcodec_free_context(&video_codec_ctx_orig_);

    // Close the video file
    avformat_close_input(&av_format_ctx_);

    running_ = false;
    done_ = true;
}
