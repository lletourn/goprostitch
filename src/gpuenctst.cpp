#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

extern "C" {
  #include <libavutil/error.h>
  #include <libavutil/imgutils.h>
  #include <libavutil/opt.h>
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
  #include <libswscale/swscale.h>
}

AVPixelFormat pix_fmt(AV_PIX_FMT_YUV444P);
#define AV_TS_MAX_STRING_SIZE 32

using namespace std;
using namespace cv;

static inline string av_ts_make_string(int64_t ts) {
    char buf[AV_TS_MAX_STRING_SIZE];
    if (ts == AV_NOPTS_VALUE) snprintf(buf, AV_TS_MAX_STRING_SIZE, "NOPTS");
    else                      snprintf(buf, AV_TS_MAX_STRING_SIZE, "%" PRId64, ts);
    return string(buf);
}

string av_ts_make_time_string2(int64_t ts, AVRational tb) {
    char buf[AV_TS_MAX_STRING_SIZE];
    if (ts == AV_NOPTS_VALUE) {
        snprintf(buf, AV_TS_MAX_STRING_SIZE, "NOPTS");
    } else {
        double val = av_q2d(tb) * ts;
        double log = (fpclassify(val) == FP_ZERO ? -INFINITY : floor(log10(fabs(val))));
        int precision = (isfinite(log) && log < 0) ? -log + 5 : 6;
        int last = snprintf(buf, AV_TS_MAX_STRING_SIZE, "%.*f", precision, val);
        last = FFMIN(last, AV_TS_MAX_STRING_SIZE - 1) - 1;
        for (; last && buf[last] == '0'; last--);
        for (; last && buf[last] != 'f' && (buf[last] < '0' || buf[0] > '9'); last--);
        buf[last + 1] = '\0';
    }
    return string(buf);
}

static inline string av_ts_make_time_string2(int64_t ts,  const AVRational *tb) {
    return av_ts_make_time_string2(ts, *tb);
}
  
static void log_packet(const AVFormatContext *fmt_ctx, const AVPacket *pkt) {
    AVRational *time_base = &fmt_ctx->streams[pkt->stream_index]->time_base;
 
    printf("pts:%s pts_time:%s dts:%s dts_time:%s duration:%s duration_time:%s stream_index:%d\n",
             av_ts_make_string(pkt->pts).c_str(), av_ts_make_time_string2(pkt->pts, time_base).c_str(),
             av_ts_make_string(pkt->dts).c_str(), av_ts_make_time_string2(pkt->dts, time_base).c_str(),
             av_ts_make_string(pkt->duration).c_str(), av_ts_make_time_string2(pkt->duration, time_base).c_str(),
             pkt->stream_index);
}


void create_frame(Mat& img, uint32_t width, uint32_t height) {
    if(img.cols != width || img.rows != height || img.channels() != 3)
        throw new runtime_error("Bad Mat");

    img.setTo(0);

    // Get the current time
    time_t now = time(0);
    tm* localtm = localtime(&now);
    char time_str[9];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", localtm);
  
    // Define text properties
    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 1;
    Scalar color(255, 255, 255); // White color in BGR
    int thickness = 2;
  
    // Calculate text size
    int baseline = 0;
    Size textSize = getTextSize(time_str, fontFace, fontScale, thickness, &baseline);
  
    // Calculate text position for centering
    int textX = (width - textSize.width) / 2;
    int textY = (height + textSize.height) / 2;
  
    // Put the text on the image
    putText(img, time_str, Point(textX, textY), fontFace, fontScale, color, thickness, LINE_AA);
}

class OutputEncoder {
 public:
    OutputEncoder(uint32_t width, uint32_t height)
        : filename_("a.mp4"),
          video_width_(width),
          video_height_(height),
          video_frame_rate_((AVRational){60000, 1001}),
          colorspace_(AVColorSpace::AVCOL_SPC_BT709),
          color_range_(AVColorRange::AVCOL_RANGE_UNSPECIFIED),
          pool_size_(8),
          frame_idx_(0) {
    };

    ~OutputEncoder() {};

    void initialize() {
        avformat_alloc_output_context2(&av_format_ctx_, NULL, NULL, filename_.c_str());
        if (!av_format_ctx_) {
            spdlog::error("Could not allocate format context");
            throw runtime_error("Could not allocate format context");
        }
    
        init_video();
        
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

    void encode (AVFrame* video_frame) {
        AVPacket* video_pkt;

        video_pkt = av_packet_alloc();
        if (!video_pkt) {
            spdlog::error("Couldn't allocate video packet");
            throw runtime_error("Couldn't allocate video packet");
        }

        chrono::steady_clock::time_point start = chrono::steady_clock::now();
        double prev_delta = 0;

        video_frame->pts = frame_idx_;
        video_frame->time_base = (AVRational){video_frame_rate_.den, video_frame_rate_.num};
        video_frame->duration = 1;
        int32_t ret = avcodec_send_frame(video_codec_ctx_, video_frame);
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

            video_pkt->time_base = video_codec_ctx_->time_base;
            av_packet_rescale_ts(video_pkt, video_codec_ctx_->time_base, video_stream_->time_base);
            video_pkt->time_base = video_stream_->time_base;
            video_pkt->stream_index = video_stream_->index;
            log_packet(av_format_ctx_, video_pkt);
            ret = av_interleaved_write_frame(av_format_ctx_, video_pkt);
            if (ret < 0) {
                spdlog::error("Couldn't mux packet");
                throw runtime_error("Couldn't mux packet");
            }
            av_packet_unref(video_pkt);
        }
        frame_idx_++;
    };

    void close() {
        AVPacket* video_pkt;

        video_pkt = av_packet_alloc();
        if (!video_pkt) {
            spdlog::error("Couldn't allocate video packet");
            throw runtime_error("Couldn't allocate video packet");
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
            av_packet_rescale_ts(video_pkt, video_codec_ctx_->time_base, video_stream_->time_base);
            video_pkt->stream_index = video_stream_->index;
            log_packet(av_format_ctx_, video_pkt);
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

        avio_closep(&av_format_ctx_->pb);
        avformat_free_context(av_format_ctx_);
    };

 private:
    void init_video() {
        AVDictionary *opt=NULL;
    
        video_codec_ = avcodec_find_encoder_by_name("hevc_nvenc");
        if (!video_codec_) {
            spdlog::error("Video codec not found");
            throw runtime_error("Video codec not found");
        }
    
        video_stream_ = avformat_new_stream(av_format_ctx_, NULL);
        if (!video_stream_) {
            spdlog::error("Could not allocate video stream");
            throw runtime_error("Could not allocate video stream");
        }
    
        video_codec_ctx_ = avcodec_alloc_context3(video_codec_);
        if (!video_codec_ctx_) {
            spdlog::error("Could not allocate video codec context");
            throw runtime_error("Could not allocate video codec context");
        }
    
        double fr_dbl = av_q2d(video_frame_rate_);
        cout << "AAAAA: " << fr_dbl << endl;
        uint32_t gop = (uint32_t) (round(fr_dbl / 10.0) * 100.0); // 59.94 would be 599.4, we want 600.)
    
        video_codec_ctx_->codec_id = video_codec_->id;
        video_codec_ctx_->height = video_height_;
        video_codec_ctx_->width = video_width_;
        video_codec_ctx_->colorspace = colorspace_;
        video_codec_ctx_->color_range = color_range_;
        video_codec_ctx_->pix_fmt = pix_fmt;
        video_codec_ctx_->time_base = (AVRational){video_frame_rate_.den, video_frame_rate_.num};;
        // video_codec_ctx_->rc_max_rate = 11*1024*1024;
        // video_codec_ctx_->gop_size = gop;
        // video_codec_ctx_->max_b_frames = 3;
        video_codec_ctx_->framerate = video_frame_rate_;
        // // video_codec_ctx_->profile = NV_ENC_PROFILE_HEVC_MAIN; NOT ACCESSIBLE PUBLICALY. Use codec priv_data
        // // video_codec_ctx_->level = NV_ENC_LEVEL_HEVC_51; NOT ACCESSIBLE PUBLICALY. Use codec priv_data
        // video_codec_ctx_->qmin = 24;
        // video_codec_ctx_->qmax = 51;
        // video_codec_ctx_->rc_buffer_size = 20*1024*1024;
        // av_opt_set(video_codec_ctx_->priv_data, "preset", "p5", 0);
        // av_opt_set(video_codec_ctx_->priv_data, "profile", "main", 0);
        // av_opt_set(video_codec_ctx_->priv_data, "level", "5.1", 0);
        // av_opt_set(video_codec_ctx_->priv_data, "rc", "vbr", 0);
        // av_opt_set(video_codec_ctx_->priv_data, "b_ref_mode", "each", 0);
        // av_opt_set(video_codec_ctx_->priv_data, "cq", "24", 0);
        // av_opt_set(video_codec_ctx_->priv_data, "spatial-aq", "1", 0);
        // // av_opt_set(video_codec_ctx_->priv_data, "rc-lookahead", "20", 0);
    
        if (av_format_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
            video_codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        av_dict_set(&opt, "flags", "+frame_duration", AV_DICT_MULTIKEY);
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

        video_stream_->time_base = (AVRational){ 1, video_frame_rate_.num };
        video_stream_->avg_frame_rate = video_codec_ctx_->framerate;
        if(spdlog::should_log(spdlog::level::debug))
            av_dump_format(av_format_ctx_, 0, filename_.c_str(), 1);
    
    };

 private:
    const std::string filename_;
    uint32_t video_width_;
    uint32_t video_height_;
    AVRational video_frame_rate_;
    AVColorSpace colorspace_;
    AVColorRange color_range_;
    uint32_t pool_size_;
    uint32_t frame_idx_;

    AVFormatContext* av_format_ctx_;
    AVStream* video_stream_;
    const AVCodec* video_codec_;
    AVCodecContext* video_codec_ctx_;
};


int main(int argc, const char ** argv) {
    cv::setNumThreads(0);
    av_log_set_level(AV_LOG_DEBUG);

    spdlog::set_pattern("%Y%m%dT%H:%M:%S.%e [%^%l%$] -%n- -%t- : %v");
    spdlog::set_level(spdlog::level::debug);

    int width = 4730;
    int height = 1712;

    // Create a blank image
    Mat img(height, width, CV_8UC3);
    Mat img_yuv;

    AVFrame* src_frame = av_frame_alloc();
    if (!src_frame) {
        spdlog::error("Could not allocate video frame");
        throw runtime_error("Could not allocate video frame");
    }
    src_frame->format = pix_fmt;
    src_frame->width  = width;
    src_frame->height = height;
    av_image_fill_linesizes(src_frame->linesize, pix_fmt, width);
    int32_t ret = av_frame_get_buffer(src_frame, 0);
    if (ret < 0) {
        spdlog::error("Could not allocate the video frame data");
        throw runtime_error("Could not allocate the video frame data");
    }
    SwsContext* sws_ctx = sws_getContext(width, height, AV_PIX_FMT_BGR24, width, height, pix_fmt, SWS_BICUBIC, NULL, NULL, NULL);

    OutputEncoder encoder(width, height);
    encoder.initialize();
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    double delta = 0.0;
    int linesize[4];
    av_image_fill_linesizes(linesize, AV_PIX_FMT_BGR24, width);

    uint8_t* d[4];
    for(int i=0; i < 60*10; ++i) {
        chrono::steady_clock::time_point frame_start = chrono::steady_clock::now();
        create_frame(img, width, height);

        // cvtColor(img, img_yuv, COLOR_BGR2YUV_I420);
        // memcpy(data.get(), img_yuv.data, data_size);
        av_image_fill_pointers(d, AV_PIX_FMT_BGR24, height, img.data, linesize);
        ret = av_frame_make_writable(src_frame);
        if (ret < 0) {
            spdlog::error("Error making video frame writable");
            throw runtime_error("Error making video frame writable");
        }

        sws_scale(sws_ctx, d, linesize, 0, height, src_frame->data, src_frame->linesize);

        encoder.encode(src_frame);
        //encoder.encode((const uint8_t*)img_yuv.data);
        double frame_delta = 16.66666 - (double)chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - frame_start).count();
        if(frame_delta > 0.0)
            this_thread::sleep_for(chrono::milliseconds((int)frame_delta));

        delta = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - start).count();
        //cout << "FD: " << frame_delta << " Time: " << (delta/1000.0) << endl;
    }
    av_frame_free(&src_frame);
    encoder.close();
}
