import argparse
import av
import collections
import cv2
import fractions
import logging
import numpy as np
# from skimage import exposure
import time


logger = logging.getLogger(__name__)


class MatchHistogram:
    """
    Modified from https://github.com/scikit-image/scikit-image/blob/main/skimage/exposure/histogram_matching.py
    """
    def __init__(self, reference_image):
        self.channel_reference_quantiles = list()
        self.channel_reference_values = list()
        for channel in range(reference_image.shape[-1]):
            channel_quantiles, channel_values = self._build_reference_histogram(reference_image[..., channel])
            self.channel_reference_quantiles.append(channel_quantiles)
            self.channel_reference_values.append(channel_values)

    def _build_reference_histogram(self, reference_channel):
        counts = np.bincount(reference_channel.reshape(-1))

        # omit values where the count was 0
        channel_values = np.nonzero(counts)[0]
        counts = counts[channel_values]
        channel_quantiles = np.cumsum(counts) / reference_channel.size

        return channel_quantiles, channel_values

    def match(self, image):
        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            channel_img = image[..., channel]
            lookup = channel_img.reshape(-1)
            counts = np.bincount(lookup)
            quantiles = np.cumsum(counts) / channel_img.size

            interp_a_values = np.interp(quantiles, self.channel_reference_quantiles[channel], self.channel_reference_values[channel])
            matched_channel = interp_a_values[lookup].reshape(channel_img.shape)
            matched[..., channel] = matched_channel
        return matched


class InputProcessor():

    def __init__(self, video_filename):
        self.video_filename = video_filename
        self.video_container = None
        self.packets = None
        self.frame_iter = None
        self.frame_idx = 0

        self.timecode = None
        self.read_video_frames = collections.deque()
        self.read_audio_frames = collections.deque()

        self._open()

    def _open(self):
        self.video_container = av.open(self.video_filename)
        self.packets = self.video_container.demux()
        self.frame_iter = iter(self.packets)

    def ready(self):
        return self.timecode is not None

    # def get_video(self, frame_idx):
    #     while True:
    #         if len(self.read_video_frames) == 0:
    #             return None

    #         frame_idx_since_midnight = self.read_video_frames[0].idx + self.index_timecode[0]
    #         if frame_idx_since_midnight < frame_idx:
    #             self.read_video_frames.popleft()
    #         elif frame_idx_since_midnight == frame_idx:
    #             return self.read_video_frames.popleft()
    #         else:
    #             return None

    def pop_video(self):
        if len(self.read_video_frames) == 0:
            return None
        return self.read_video_frames.popleft()

    def pop_audio(self):
        if len(self.read_audio_frames) == 0:
            return None
        return self.read_audio_frames.popleft()

    def next(self):
        packet = None
        try:
            packet = next(self.frame_iter)
        except StopIteration:
            return False
        self._process_packet(packet)
        return True

    def _process_packet(self, packet):
        raw_frames = packet.decode()
        if packet.stream.type == 'data':
            if packet.stream.name is None:
                # raw_bytes = bytes(packet)
                # https://github.com/gopro/gpmf-parser
                # Raw data to be stored in Big Endian
                # frame_from_midnight = int.from_bytes(raw_bytes, "big")  # Skipping because of bad sync
                frame_from_midnight = 0
                self.timecode = frame_from_midnight
        elif packet.stream.type == 'video':
            for raw_frame in raw_frames:
                self.read_video_frames.append(raw_frame)
                self.frame_idx += 1
        elif packet.stream.type == 'audio':
            for raw_frame in raw_frames:
                self.read_audio_frames.append(raw_frame)
        else:
            logger.warning(f"Unhandled stream type: {packet.stream.type}")

    def close(self):
        if self.video_container:
            self.video_container.close()
            self.video_container = None


def main():
    parser = argparse.ArgumentParser(description='Run detection on hockey broadcast videos.')
    parser.add_argument("--left", type=str, help="Left videos")
    parser.add_argument("--right", type=str, help="Right videos")
    parser.add_argument("-o", "--output", type=str, required=True, help="output")
    parser.add_argument("-l", "--log", help="log level (default: info)", choices=["debug", "info", "warning", "error", "critical"], default="info")
    args = parser.parse_args()

    logdatefmt = '%Y%m%dT%H:%M:%S'
    logformat = '%(asctime)s.%(msecs)03d [%(levelname)s] -%(name)s- -%(threadName)s- : %(message)s'
    logging.basicConfig(datefmt=logdatefmt, format=logformat, level=args.log.upper())

    left_video_filename = args.left
    right_video_filename = args.right
    output_filename = args.output
    output_width = 7060
    output_height = 2400

    left_processor = InputProcessor(left_video_filename)
    right_processor = InputProcessor(right_video_filename)

    process_left = True
    process_right = True
    frame_idx_since_midnight = None

    left_frame = None
    right_frame = None
    match_histograms = None

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

        start_time = time.time()
        while process_left and process_right:
            if process_left:
                process_left = left_processor.next()
            if process_right:
                process_right = right_processor.next()

            if frame_idx_since_midnight is None and left_processor.ready() and right_processor.ready():
                if left_processor.timecode >= right_processor.timecode:
                    frame_idx_since_midnight = left_processor.timecode
                if left_processor.timecode < right_processor.timecode:
                    frame_idx_since_midnight = right_processor.timecode

            if frame_idx_since_midnight is not None:
                if left_frame is None:
                    left_frame = left_processor.pop_video()
                if right_frame is None:
                    right_frame = right_processor.pop_video()

                if left_frame is not None and right_frame is not None:
                    left_image = left_frame.to_ndarray(format='bgr24')
                    right_image = right_frame.to_ndarray(format='bgr24')
                    frame_idx_since_midnight += 1

                    throughput = frame_idx_since_midnight / (time.time()-start_time)
                    logger.info(f"Read frames: {frame_idx_since_midnight} {throughput:0.2f}")

                    if match_histograms is None:
                        match_histograms = MatchHistogram(left_image)

                    pano = fix_and_stitch(match_histograms, left_image, right_image)
                    pano = pano[385:385+output_height, 670:670+output_width]
                    av_videoframe = av.VideoFrame.from_ndarray(pano, format='bgr24')
                    av_videoframe.pts = left_frame.pts
                    new_packet = output_video_stream.encode(av_videoframe)
                    output_container.mux(new_packet)
                    if frame_idx_since_midnight > 60*5:
                        break

                    # scale = 0.1
                    # cv2.imshow("Left", cv2.resize(left_frame.data, (1280, 720)))
                    # cv2.imshow("Right", cv2.resize(right_frame.data, (1280, 720)))
                    # cv2.imshow("Pano", cv2.resize(pano, (int(pano.shape[1]*scale), int(pano.shape[0]*scale))))
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                    left_frame = None
                    right_frame = None

            while True:
                audio_frame = left_processor.pop_audio()
                if audio_frame is None:
                    break
                new_packet = output_audio_stream_left.encode(audio_frame)
                output_container.mux(new_packet)

            while True:
                audio_frame = right_processor.pop_audio()
                if audio_frame is None:
                    break
                new_packet = output_audio_stream_right.encode(audio_frame)
                output_container.mux(new_packet)
        output_container.mux(output_video_stream.encode())
        output_container.mux(output_audio_stream_left.encode())
        output_container.mux(output_audio_stream_right.encode())

    left_processor.close()
    right_processor.close()


def fix_and_stitch(match_histograms, left_frame, right_frame):
    # right_frame = exposure.match_histograms(right_frame, left_frame, multichannel=True)
    right_frame = match_histograms.match(right_frame)

    pano = None
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch([left_frame, right_frame])
    if status != cv2.Stitcher_OK:
        raise RuntimeError("Can't stitch images, error code = %d" % status)

    return pano


if __name__ == '__main__':
    main()
