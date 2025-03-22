#!/bin/env python3
import argparse
import cv2
import logging

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

    cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Left", 1280, 720)
    cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Right", 1280, 720)
    with FrameReader(left_video_filename) as left_video, FrameReader(right_video_filename) as right_video:
        while True:
            left_frame, left_frame_data = left_video.next_frame()
            right_frame, right_frame_data = right_video.next_frame()
            logger.info("Read frame: %s", left_frame_data.frame_id)
            cv2.imshow("Left", left_frame)
            cv2.imshow("Right", right_frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
