import av
import fractions
import logging


logger = logging.getLogger(__name__)


def handle_queues(video_queue, left_audio_queue, right_audio_queue, output_container, output_video_stream, output_audio_stream_left, output_audio_stream_right):
    while not video_queue.empty():
        panorama = video_queue.get()
        av_videoframe = av.VideoFrame.from_ndarray(panorama.img, format='bgr24')
        av_videoframe.pts = panorama.pts
        new_packet = output_video_stream.encode(av_videoframe)
        output_container.mux(new_packet)

    while not left_audio_queue.empty():
        audio_packet = left_audio_queue.get()
        av_audioframe = av.AudioFrame.from_ndarray(audio_packet.data, format=audio_packet.format, layout=audio_packet.layout)
        av_audioframe.sample_rate = audio_packet.sample_rate
        av_audioframe.pts = audio_packet.pts
        new_packet = output_audio_stream_left.encode(av_audioframe)
        output_container.mux(new_packet)

    while not right_audio_queue.empty():
        audio_packet = right_audio_queue.get()
        av_audioframe = av.AudioFrame.from_ndarray(audio_packet.data, format=audio_packet.format, layout=audio_packet.layout)
        av_audioframe.sample_rate = audio_packet.sample_rate
        av_audioframe.pts = audio_packet.pts
        new_packet = output_audio_stream_right.encode(av_audioframe)
        output_container.mux(new_packet)


def process_output(output_filename, output_width, output_height, stop_encoding, video_queue, left_audio_queue, right_audio_queue):
    try:
        with av.open(output_filename, mode='w') as output_container:
            options = {"crf": str(23)}
            output_video_stream = output_container.add_stream('libx264')
            output_video_stream.options = options
            output_video_stream.width = output_width
            output_video_stream.height = output_height
            output_video_stream.pix_fmt = 'yuvj420p'
            output_video_stream.time_base = fractions.Fraction(1, 60000)
            output_audio_stream_left = output_container.add_stream('aac', rate=48000)
            output_audio_stream_left.metadata['title'] = 'Left'
            output_audio_stream_right = output_container.add_stream('aac', rate=48000)
            output_audio_stream_right.metadata['title'] = 'Right'

            while not stop_encoding.value:
                handle_queues(video_queue, left_audio_queue, right_audio_queue, output_container, output_video_stream, output_audio_stream_left, output_audio_stream_right)
            # One last time, empty the queues
            handle_queues(video_queue, left_audio_queue, right_audio_queue, output_container, output_video_stream, output_audio_stream_left, output_audio_stream_right)

            output_container.mux(output_video_stream.encode())
            output_container.mux(output_audio_stream_left.encode())
            output_container.mux(output_audio_stream_right.encode())
    except Exception:
        logger.exception(f"Output processing {output_filename} caught exception")
