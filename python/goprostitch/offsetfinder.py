#!/bin/env python3
import argparse
import concurrent.futures
import cv2
from dataclasses import dataclass
from enum import Enum
import json
import logging
import numpy
import sys

from goprostitch.framereader import FrameReader

logger = logging.getLogger(__name__)


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

    frame_id = 0
    prev_left_crop = None
    prev_right_crop = None

    # left, right, top, bottom
    left_rect = [3050, 3058, 355, 369]
    right_rect = [940, 949, 400, 413]
    with FrameReader(left_video_filename, frame_skips_before_seek=59) as left_video, FrameReader(right_video_filename, frame_skips_before_seek=59) as right_video:
        logger.info("Videos loaded...")
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
            left_frame = left_video.get_frame(frame_id)
            right_frame = right_video.get_frame(frame_id)

            # copy to make continuous
            left_crop = left_frame.frame[left_rect[2]:left_rect[3], left_rect[0]:left_rect[1]].copy()
            right_crop = right_frame.frame[right_rect[2]:right_rect[3], right_rect[0]:right_rect[1]].copy()
            left_crop = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
            right_crop = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY)

            if prev_left_crop is not None and prev_right_crop is not None:
                left_norm = cv2.norm(left_crop, prev_left_crop, cv2.NORM_L2)
                left_diff = cv2.absdiff(left_crop, prev_left_crop)
                right_norm = cv2.norm(right_crop, prev_right_crop, cv2.NORM_L2)
                right_diff = cv2.absdiff(right_crop, prev_right_crop)
                
                logger.info(f"F: {frame_id} L: {left_norm} R: {right_norm} La: {left_diff.sum()} Ra: {right_diff.sum()}")
                must_wait = False
                if left_norm > 200:
                    cv2.imwrite(f"left-{frame_id}.jpg", left_crop)
                    cv2.imwrite(f"leftdiff-{frame_id}.jpg", left_diff)
                    # cv2.imshow("Left", left_crop)
                    # cv2.imshow("Left Diff", left_diff)
                    # must_wait = True
                elif right_norm > 200:
                    cv2.imwrite(f"right-{frame_id}.jpg", right_crop)
                    cv2.imwrite(f"rightdiff-{frame_id}.jpg", right_diff)
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
            frame_id += 1

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
