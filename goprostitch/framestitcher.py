import cv2
from dataclasses import dataclass
import logging
import numpy
import queue
# from skimage import exposure
import time


logger = logging.getLogger(__name__)


@dataclass
class FrameParts:
    left: numpy.ndarray
    right: numpy.ndarray
    idx: int
    pts: int


@dataclass
class Panorama:
    img: numpy.ndarray
    idx: int
    pts: int


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
        counts = numpy.bincount(reference_channel.reshape(-1))

        # omit values where the count was 0
        channel_values = numpy.nonzero(counts)[0]
        counts = counts[channel_values]
        channel_quantiles = numpy.cumsum(counts) / reference_channel.size

        return channel_quantiles, channel_values

    def match(self, image):
        matched = numpy.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            channel_img = image[..., channel]
            lookup = channel_img.reshape(-1)
            counts = numpy.bincount(lookup)
            quantiles = numpy.cumsum(counts) / channel_img.size

            interp_a_values = numpy.interp(quantiles, self.channel_reference_quantiles[channel], self.channel_reference_values[channel])
            matched_channel = interp_a_values[lookup].reshape(channel_img.shape)
            matched[..., channel] = matched_channel
        return matched


def stitch_frames(frame_queue, pano_queue, stop, reference_image, camera_left, camera_right, crop_x, crop_y, output_width, output_height):
    try:
        match_histograms = MatchHistogram(reference_image)
        while not stop.value:
            try:
                frame = frame_queue.get(timeout=2)
                # right_frame = exposure.match_histograms(right_frame, left_frame, multichannel=True)
                frame.right = match_histograms.match(frame.right)

                stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
                stitcher.setTransformCams([frame.left, frame.right], camera_left, camera_right)
                # status, pano = stitcher.composePanorama([frame.left, frame.right])
                status, pano = stitcher.composePanorama()
                if status != cv2.Stitcher_OK:
                    raise RuntimeError("Can't compose images, error code = %d" % status)
                pano = pano[crop_y:crop_y+output_height, crop_x:crop_x+output_width]

                p = Panorama(img=pano, idx=frame.idx, pts=frame.pts)
                pano_queue.put(p)
            except queue.Empty:
                time.sleep(0.5)
    except Exception:
        logger.exception("Frame stitcher caught exception")
