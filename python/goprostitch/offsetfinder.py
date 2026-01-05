#!/bin/env python3
import argparse
import concurrent.futures
import cv2
from dataclasses import dataclass
from enum import Enum
import json
import logging
import multiprocessing
import numpy
import sys

from goprostitch.framereader import FrameReader

logger = logging.getLogger(__name__)


def process_video(video_filename: str, crop_l: int, crop_r: int, crop_t: int, crop_b: int, queue: multiprocessing.Queue) -> None:
    frame_id = 0
    with FrameReader(video_filename, frame_skips_before_seek=59) as video:
        while True:
            frame = video.get_frame(frame_id)
            crop_rgb = frame.frame[crop_t:crop_b, crop_l:crop_r]
            crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2GRAY)

            queue.put((frame_id, crop_gray.copy()), block=True, timeout=None)
            frame_id += 1


def main() -> None:
    parser = argparse.ArgumentParser(description='Run detection on hockey broadcast videos.')
    parser.add_argument('--left', required=True, type=str, help='Left video')
    parser.add_argument('--right', required=True, type=str, help='Right video')
    parser.add_argument("-l", "--log", help="log level (default: info)", choices=["debug", "info", "warning", "error", "critical"], default="info")
    args = parser.parse_args()

    logdatefmt = '%Y%m%dT%H:%M:%S'
    logformat = '%(asctime)s.%(msecs)03d [%(levelname)s] -%(name)s- -%(threadName)s- : %(message)s'
    logging.basicConfig(datefmt=logdatefmt, format=logformat, level=args.log.upper())

    left_video_filename = args.left
    right_video_filename = args.right

    left_q: multiprocessing.Queue = multiprocessing.Queue(maxsize=200)
    right_q: multiprocessing.Queue = multiprocessing.Queue(maxsize=200)

    left_process = multiprocessing.Process(target=process_video, args=(left_video_filename, 3050, 3058, 355, 369, left_q), daemon=True)
    right_process = multiprocessing.Process(target=process_video, args=(right_video_filename, 938, 949, 400, 413, right_q), daemon=True)
    left_process.start()
    right_process.start()

    prev_left_crop = None
    prev_right_crop = None

    # cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Left", 200, 200)
    # cv2.moveWindow("Left", 20, 20)

    # cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Right", 200, 200)
    # cv2.moveWindow("Right", 500, 20)

    # cv2.namedWindow("Left Diff", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Left Diff", 200, 200)
    # cv2.moveWindow("Left Diff", 20, 500)

    # cv2.namedWindow("Right Diff", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Right Diff", 200, 200)
    # cv2.moveWindow("Right Diff", 500, 500)
    
    while True:
        left_frame_id, left_crop = left_q.get()
        right_frame_id, right_crop = right_q.get()

        must_wait = False
        if prev_left_crop is not None and prev_right_crop is not None:
            left_norm = cv2.norm(left_crop, prev_left_crop, cv2.NORM_L2)
            left_diff = cv2.absdiff(left_crop, prev_left_crop)
            right_norm = cv2.norm(right_crop, prev_right_crop, cv2.NORM_L2)
            right_diff = cv2.absdiff(right_crop, prev_right_crop)
            # if left_norm > 150 or right_norm > 150:
            logger.debug("Queue sizes: L=%s  R=%s", left_q.qsize(), right_q.qsize())
            logger.info(f"Fl: {left_frame_id} Fr: {right_frame_id} L: {left_norm} R: {right_norm} La: {left_diff.sum()} Ra: {right_diff.sum()}")

            if left_norm > 200:
                left_out = numpy.concatenate((prev_left_crop, left_crop), axis=1)
                cv2.imwrite(f"left-{left_frame_id}.jpg", left_out)
                cv2.imwrite(f"leftdiff-{left_frame_id}-{int(left_norm)}.jpg", left_diff)
                # cv2.imshow("Left", left_crop)
                # cv2.imshow("Left Diff", left_diff)
                # must_wait = True
            if right_norm > 200:
                right_out = numpy.concatenate((prev_right_crop, right_crop), axis=1)
                cv2.imwrite(f"right-{right_frame_id}.jpg", right_out)
                cv2.imwrite(f"rightdiff-{right_frame_id}-{int(right_norm)}.jpg", right_diff)
                # cv2.imshow("Right", right_crop)
                # cv2.imshow("Right Diff", right_diff)
                # must_wait = True
        else:
            # cv2.imshow("Left", left_crop)
            # cv2.imshow("Right", right_crop)
            # must_wait = True
            pass

        if must_wait:
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                break

        prev_left_crop = left_crop
        prev_right_crop = right_crop
    left_process.join()
    right_process.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
