#include "inputprocessor.hpp"

#include <queue>
#include <stdexcept>
#include <spdlog/spdlog.h>

extern "C" {
  #include <libavutil/error.h>
  #include <libavutil/imgutils.h>
  #include <libavutil/opt.h>
}

using namespace std;

string gen_random(const int len) {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    std::string tmp_s;
    tmp_s.reserve(len);

    for (int i = 0; i < len; ++i) {
        tmp_s += alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    return tmp_s;
}

static AVPixelFormat hw_pix_fmt;
static AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    const enum AVPixelFormat *p;

    for (p = pix_fmts; *p != -1; p++) {
        if (*p == hw_pix_fmt)
            return *p;
    }

    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
}


InputProcessor::InputProcessor(const string& filename, bool use_gpu, uint32_t offset, uint32_t queue_size)
: filename_(filename),
  use_gpu_(use_gpu),
  offset_(offset),
  timecode_(0),
  running_(false),
  done_(false),
  video_packet_queue_(queue_size),
  audio_packet_queue_(queue_size),
  video_time_base_(Rational(0,0)),
  video_frame_rate_(Rational(0,0)),
  audio_time_base_(Rational(0,0)) {
    pixel_format_ = PixelFormat::PIX_FMT_NONE;
    av_format_ctx_ = nullptr;
    hw_device_ctx_ = nullptr;
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

    AVHWDeviceType type;
    if(use_gpu_) {
        type = av_hwdevice_find_type_by_name("cuda");
        if (type == AV_HWDEVICE_TYPE_NONE) {
            spdlog::error("Device type {} is not supported.", "cuda");
            spdlog::error("Available device types:");
            while((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE)
                spdlog::error(" %s", av_hwdevice_get_type_name(type));
            throw runtime_error("");
        }
    }

    // To get the udat gopro specific fields.
    AVDictionary* opt = nullptr;
    av_dict_set(&opt, "export_all", "1", 0);
    int ret = avformat_open_input(&av_format_ctx_, filename_.c_str(), NULL, &opt);
    if (ret < 0) {
        // couldn't open file
        spdlog::error("Could not open file: {} ", filename_);
        throw runtime_error("Could not open file");
    }
    AVDictionaryEntry *e;
    if (e = av_dict_get(opt, "", NULL, AV_DICT_IGNORE_SUFFIX)) {
        spdlog::error("Option {} not recognized by the demuxer.", e->key);
        throw runtime_error("Option not recognized by the demuxer.");
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
            video_stream_idx_ = i;

            duration_ = (double)av_format_ctx_->streams[i]->duration * av_q2d(av_format_ctx_->streams[i]->time_base);
            if(av_format_ctx_->streams[i]->codecpar->sample_aspect_ratio.num != 1 || av_format_ctx_->streams[i]->codecpar->sample_aspect_ratio.den != 1) {
                spdlog::error("SAR is not 1:1 for input video");
                throw runtime_error("SAR is not 1:1 for input video");
            }

            video_time_base_ = Rational(av_format_ctx_->streams[i]->time_base.num, av_format_ctx_->streams[i]->time_base.den);
        } else if (av_format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx_ = i;
            audio_time_base_ = Rational(av_format_ctx_->streams[i]->time_base.num, av_format_ctx_->streams[i]->time_base.den);
        }
    }

    video_codec_ = avcodec_find_decoder(av_format_ctx_->streams[video_stream_idx_]->codecpar->codec_id);
    if (video_codec_ == NULL) {
        spdlog::error("Unsupported video codec!");
        throw runtime_error("Unsupported video codec");
    }

    if(use_gpu_) {
        for (uint32_t i = 0;; i++) {
            const AVCodecHWConfig *config = avcodec_get_hw_config(video_codec_, i);
            if (!config) {
                spdlog::error("Decoder {} does not support device type {}.", video_codec_->name, av_hwdevice_get_type_name(type));
                throw runtime_error("Error occured");
            }
            if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
                config->device_type == type) {
                hw_pix_fmt = config->pix_fmt;
                break;
            }
        }
    }
    spdlog::info("Using decoder codec: {}", video_codec_->name);

    video_codec_ctx_ = avcodec_alloc_context3(video_codec_);
    if(!video_codec_ctx_) {
        spdlog::error("Could not allocate video codec context.");
        throw runtime_error("Error occured");
    }

    ret = avcodec_parameters_to_context(video_codec_ctx_, av_format_ctx_->streams[video_stream_idx_]->codecpar);
    if (ret != 0) {
        spdlog::error("Could not copy video codec context.");
        throw runtime_error("Error occured");
    }

    if (use_gpu_) {
        video_codec_ctx_->get_format  = get_hw_format;

        if ((av_hwdevice_ctx_create(&hw_device_ctx_, type, NULL, NULL, 0)) < 0) {
            spdlog::error("Failed to create specified HW device.");
            throw runtime_error("Failed to create specified HW device.");
        }
        video_codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    }
    colorspace_ = video_codec_ctx_->colorspace;
    color_range_ = video_codec_ctx_->color_range;
    video_frame_rate_ = Rational(av_format_ctx_->streams[video_stream_idx_]->r_frame_rate.num, av_format_ctx_->streams[video_stream_idx_]->r_frame_rate.den);
    video_codec_ctx_->thread_type = FF_THREAD_FRAME;
    video_codec_ctx_->thread_count = 1;

    ret = avcodec_open2(video_codec_ctx_, video_codec_, NULL);
    if (ret < 0) {
        spdlog::error("Could not open video codec.\n");
        throw runtime_error("Error occured");
    }

    // Print the udat, container, metadata. GPMF contains the gopro data but we are missing the size...
    // const AVDictionaryEntry *tag = NULL;
    // while ((tag = av_dict_iterate(av_format_ctx_->metadata, tag))) {
    //     cout << "Key: " << tag->key << " Value: " << tag->value << endl;
    // }
}

void InputProcessor::run() {
    spdlog::info("[inputproc] Starting..");
    #ifdef _GNU_SOURCE
    pthread_setname_np(pthread_self(), "InputProcessor");
    #endif

    char error_msg[AV_ERROR_MAX_STRING_SIZE];

    AVFrame *video_frame = nullptr;
    AVPacket* packet = av_packet_alloc();
    if (packet == NULL) {
        spdlog::error("Could not alloc packet,");
        throw runtime_error("Error occured");
    }

    unique_ptr<queue<AVPacketUniquePTR>> audio_buffer = unique_ptr<queue<AVPacketUniquePTR>>(new queue<AVPacketUniquePTR>());
    double start_timestamp = std::numeric_limits<double>::max();

    uint32_t video_idx = 0;
    int32_t ret;
    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    uint32_t output_frame_size = 0;
    AVFrame *tmp_frame = nullptr;
    AVFrame *sw_frame = nullptr;
    while (av_read_frame(av_format_ctx_, packet) >= 0 && running_.load() == true) {
        if (packet->stream_index == video_stream_idx_) {
            chrono::steady_clock::time_point video_start = chrono::steady_clock::now();
            ret = avcodec_send_packet(video_codec_ctx_, packet);
            if (ret < 0) {
                spdlog::error("Error sending video packet for decoding.");
                throw runtime_error("Error occured");
            }

            while (ret >= 0) {
                video_frame = av_frame_alloc();
                if (video_frame == nullptr) {
                    spdlog::error("Could not allocate video frame.");
                    throw runtime_error("Error occured");
                }

                ret = avcodec_receive_frame(video_codec_ctx_, video_frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    av_frame_free(&video_frame);
                    break;
                } else if (ret < 0) {
                    av_frame_free(&video_frame);
                    spdlog::error("Error while decoding.");
                    throw runtime_error("Error while decoding.");
                }

                if (video_frame->format == hw_pix_fmt) {
                    if (!(sw_frame = av_frame_alloc())) {
                        spdlog::error("Can not alloc sw_frame");
                        throw runtime_error("Can not alloc sw_frame");
                    }

                    if (av_hwframe_transfer_data(sw_frame, video_frame, 0) < 0) {
                        spdlog::error("Error transferring the data to system memory.");
                        throw runtime_error("Error transferring the data to system memory");
                    }

                    // Important or else pts, time_base and other params aren't set from the frame that comes back from the HW decoder.
                    if (av_frame_copy_props(sw_frame, video_frame) < 0) {
                        spdlog::error("Error transferring the frame properties.");
                        throw runtime_error("Error transferring the frame properties");
                    }

                    tmp_frame = sw_frame;
                } else {
                    tmp_frame = video_frame;
                }

                if (pixel_format_ == PIX_FMT_NONE) {
                    switch ((AVPixelFormat)tmp_frame->format) {
                        case AV_PIX_FMT_NV12:
                            pixel_format_ = PIX_FMT_YUV420_NV12;
                            break;
                        case AV_PIX_FMT_YUV420P:
                            pixel_format_ = PIX_FMT_YUV420_P;
                            break;
                        default:
                            spdlog::error("Unsupported pixel format input: {}", tmp_frame->format);
                            throw runtime_error("Unsupported pixel format input");
                    }

                    output_frame_size = av_image_get_buffer_size((AVPixelFormat)tmp_frame->format, tmp_frame->width, tmp_frame->height, 1);
                }

                if (video_idx >= offset_) {
                    unique_ptr<uint8_t[]> data(new uint8_t[output_frame_size]);
                    ret = av_image_copy_to_buffer(data.get(), output_frame_size, tmp_frame->data, tmp_frame->linesize, (AVPixelFormat)tmp_frame->format, tmp_frame->width, tmp_frame->height, 1);
                    if(ret < 0) {
                        spdlog::error("Error copying image to buffer.");
                        throw runtime_error("Error copying image to buffer.");
                    }

                    unique_ptr<VideoPacket> input_packet(new VideoPacket);
                    input_packet->width = video_codec_ctx_->width;
                    input_packet->height = video_codec_ctx_->height;
                    input_packet->pts = tmp_frame->pts;
                    spdlog::debug("Time: {}", tmp_frame->pts * av_q2d(av_format_ctx_->streams[video_stream_idx_]->time_base));
                    input_packet->pts_time =  tmp_frame->pts * av_q2d(av_format_ctx_->streams[video_stream_idx_]->time_base);
                    input_packet->idx = video_idx-offset_;
                    input_packet->data_size = output_frame_size;
                    input_packet->data = move(data);

                    if(start_timestamp == std::numeric_limits<double>::max())
                        start_timestamp = input_packet->pts_time;

                    auto delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - video_start).count();
                    double video_fps = (1.0/delta) * 1000.0;

                    spdlog::trace("[inputproc] sending frame {}", video_idx-offset_);
                    video_packet_queue_.push(move(input_packet));
                    spdlog::trace("[inputproc] sent frame {}", video_idx-offset_);
                    // delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();
                    // double all_fps = ((double)video_idx/delta) * 1000.0;
                    // spdlog::debug("Input FPS: {} All FPS: {}", video_fps, all_fps);
                }

                ++video_idx;
                av_frame_free(&video_frame);
                video_frame = nullptr;
                av_frame_free(&sw_frame);
                sw_frame = nullptr;
            }
        } else if (packet->stream_index == audio_stream_idx_) {
            AVPacket* audio_packet = av_packet_alloc();
            if (audio_packet == NULL) {
                spdlog::error("Could not alloc packet,");
                throw runtime_error("Error occured");
            }
            av_packet_move_ref(audio_packet, packet);
            AVPacketUniquePTR p(audio_packet);

            if(start_timestamp == std::numeric_limits<double>::max()) {
                audio_buffer->push(move(p));
            } else {
                if(audio_buffer) {
                    while(!audio_buffer->empty()) {
                        AVPacketUniquePTR old_packet(move(audio_buffer->front()));
                        audio_buffer->pop();
                        double old_packet_timestamp = old_packet->pts * av_q2d(av_format_ctx_->streams[audio_stream_idx_]->time_base);
                        if(old_packet_timestamp >= start_timestamp) {
                            audio_packet_queue_.push(move(old_packet));
                        }
                    }
                    audio_buffer.reset(nullptr);
                }
                double packet_timestamp = p->pts * av_q2d(av_format_ctx_->streams[audio_stream_idx_]->time_base);
                if(packet_timestamp >= start_timestamp)
                    audio_packet_queue_.push(move(p));
            }
        }
        av_packet_unref(packet);
    }

    spdlog::info("[inputproc] Cleaning up");
    av_packet_free(&packet);
    // sws_freeContext(sws_ctx);

    // Close the codecs
    avcodec_free_context(&video_codec_ctx_);

    // Close the video file
    avformat_close_input(&av_format_ctx_);
    av_buffer_unref(&hw_device_ctx_);

    running_ = false;
    done_ = true;
    spdlog::info("[inputproc] Done");
}
