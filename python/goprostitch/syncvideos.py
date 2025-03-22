#!/bin/env python3
import argparse
import cv2
import logging
import sys

from goprostitch.framereader import FrameReader

logger = logging.getLogger(__name__)


def get_movement() -> int:
    key = cv2.waitKeyEx()

    if key == 65361:  # Left
        return -1
    elif key == 65363:  # Right
        return 1
    elif key == 65362:  # Up
        return 60
    elif key == 65364:  # Down
        return -60
    elif key == 65360:  # FnLeft
        pass
    elif key == 65367:  # FnRight
        pass
    elif key == 65365:  # PgUp
        return int(59.94*60)
    elif key == 65366:  # PgDown
        return -1 * int(59.94*60)
    elif key == ord('q') or key == ord('Q'):
        sys.exit(0)
    else:
        print(f"Key: {key}")
    return key


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

    current_frame = 1
    with FrameReader(left_video_filename) as left_video, FrameReader(right_video_filename) as right_video:
        while True:
            left_frame = left_video.get_frame(current_frame)
            right_frame = right_video.get_frame(current_frame)
            logger.info("Read frame: %s", left_frame.frame_id)
            cv2.imshow("Left", left_frame.frame)
            cv2.imshow("Right", right_frame.frame)
            delta = get_movement()

            current_frame += delta
            if current_frame < 0:
                current_frame = 0
            elif current_frame > left_video.get_frame_count() or current_frame > right_video.get_frame_count():
                current_frame = min(left_video.get_frame_count(), right_video.get_frame_count()) - 1


if __name__ == '__main__':
    main()
