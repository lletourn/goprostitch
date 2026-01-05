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


class KeyCommands(Enum):
    NOTHING = 0
    MOVE = 1
    QUIT = 2
    STATUS = 3
    WRITE = 4
    SAVE_START = 5
    LOCK = 6


@dataclass(slots=True)
class KeyCommand:
    command: KeyCommands
    frame_movement: int = sys.maxsize


def get_movement() -> KeyCommand:
    print("Command?")
    key = cv2.waitKeyEx()

    if key == 65361:  # Left
        return KeyCommand(command=KeyCommands.MOVE, frame_movement=-1)
    elif key == 65363:  # Right
        return KeyCommand(command=KeyCommands.MOVE, frame_movement=1)
    elif key == 65362:  # Up
        return KeyCommand(command=KeyCommands.MOVE, frame_movement=60 * 15)
    elif key == 65364:  # Down
        return KeyCommand(command=KeyCommands.MOVE, frame_movement=-60 * 15)
    elif key == 65360:  # FnLeft
        return KeyCommand(command=KeyCommands.MOVE, frame_movement=-60)
    elif key == 65367:  # FnRight
        return KeyCommand(command=KeyCommands.MOVE, frame_movement=60)
    elif key == 65365:  # PgUp
        return KeyCommand(command=KeyCommands.MOVE, frame_movement=60 * 30)
    elif key == 65366:  # PgDown
        return KeyCommand(command=KeyCommands.MOVE, frame_movement=-60 * 30)
    elif key == ord('l') or key == ord('L'):
        return KeyCommand(command=KeyCommands.LOCK)
    elif key == ord('s') or key == ord('S'):
        return KeyCommand(command=KeyCommands.STATUS)
    elif key == ord('t') or key == ord('T'):
        return KeyCommand(command=KeyCommands.SAVE_START)
    elif key == ord('w') or key == ord('W'):
        return KeyCommand(command=KeyCommands.WRITE)
    elif key == ord('q') or key == ord('Q'):
        return KeyCommand(command=KeyCommands.QUIT)
    else:
        print(f"Key: {key}")
    return KeyCommand(command=KeyCommands.NOTHING)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run detection on hockey broadcast videos.')
    parser.add_argument('--left', required=True, type=str, help='Left video')
    parser.add_argument('--right', required=True, type=str, help='Right video')
    parser.add_argument('--camintrinsics', required=True, type=str, help='Camera intrinsics to undistort')
    parser.add_argument("-l", "--log", help="log level (default: info)", choices=["debug", "info", "warning", "error", "critical"], default="info")
    args = parser.parse_args()

    logdatefmt = '%Y%m%dT%H:%M:%S'
    logformat = '%(asctime)s.%(msecs)03d [%(levelname)s] -%(name)s- -%(threadName)s- : %(message)s'
    logging.basicConfig(datefmt=logdatefmt, format=logformat, level=args.log.upper())

    with open(args.camintrinsics, "r") as f:
        camera_intrinsics = json.load(f)
    K = numpy.array(camera_intrinsics["K"], dtype=numpy.float64)
    distcoeffs = numpy.array(camera_intrinsics["D"], dtype=numpy.float64)

    left_video_filename = args.left
    right_video_filename = args.right

    left_frame_idx = 1
    right_frame_idx = 0
    lock_right = False
    start_offset = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        with FrameReader(left_video_filename, frame_skips_before_seek=59) as left_video, FrameReader(right_video_filename, frame_skips_before_seek=59) as right_video:
            logger.info("Videos loaded...")
            cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Left", 1280, 720)
            cv2.namedWindow("Right", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Right", 1280, 720)

            while True:
                future_tasks = (
                                executor.submit(left_video.get_frame, left_frame_idx),
                                executor.submit(right_video.get_frame, right_frame_idx)
                                )
                concurrent.futures.wait(future_tasks, timeout=None, return_when=concurrent.futures.ALL_COMPLETED)
                left_frame = future_tasks[0].result()
                right_frame = future_tasks[1].result()
                # left_frame = left_video.get_frame(left_frame_idx)
                # right_frame = right_video.get_frame(right_frame_idx)
                logger.info("Read frame: %s", left_frame.frame_id)
                cv2.imshow("Left", left_frame.frame)
                cv2.imshow("Right", right_frame.frame)
                key_movement = get_movement()

                if key_movement.command == KeyCommands.QUIT:
                    break
                elif key_movement.command == KeyCommands.LOCK:
                    lock_right = not lock_right
                elif key_movement.command == KeyCommands.SAVE_START:
                    start_offset = min(left_frame_idx, right_frame_idx)
                elif key_movement.command == KeyCommands.STATUS:
                    left_offset = 0
                    right_offset = 0
                    if right_frame_idx > left_frame_idx:
                        right_offset = right_frame_idx - left_frame_idx
                    elif right_frame_idx < left_frame_idx:
                        left_offset = left_frame_idx - right_frame_idx
                    print("Status:")
                    print(f"\tLeft Frame idx    : {left_frame_idx}")
                    print(f"\tRight Frame idx   : {right_frame_idx}")
                    print(f"\tRight Locked      : {lock_right}")
                    print(f"\tDelta             : {right_frame_idx - left_frame_idx}")
                    print(f"\tStart Offset      : {start_offset}")
                    print(f"\tLeft stitch start : {start_offset + left_offset}")
                    print(f"\tRight stitch start: {start_offset + right_offset}")
                elif key_movement.command == KeyCommands.WRITE:
                    cv2.imwrite("left.png", left_frame.frame)
                    cv2.imwrite("right.png", right_frame.frame)

                    h, w = left_frame.frame.shape[:2]
                    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, distcoeffs, (w, h), 0, (w, h))
                    mapx, mapy = cv2.initUndistortRectifyMap(K, distcoeffs, None, newcameramtx, (w, h), 5)  # type: ignore
                    left_undistorted = cv2.remap(left_frame.frame, mapx, mapy, cv2.INTER_LINEAR)
                    right_undistorted = cv2.remap(right_frame.frame, mapx, mapy, cv2.INTER_LINEAR)
                    cv2.imwrite("left_undistorted.png", left_undistorted)
                    cv2.imwrite("right_undistorted.png", right_undistorted)

                    with open("offs.txt", "w") as f:
                        print(f"Start offset: {start_offset}",  file=f)
                        print(f"Left: {left_frame_idx-1}-{left_frame_idx}",  file=f)
                        print(f"Right: {right_frame_idx-1}-{right_frame_idx}",  file=f)
                        print(f"{right_frame_idx - left_frame_idx}",  file=f)
                elif key_movement.command == KeyCommands.LOCK:
                    lock_right = not lock_right
                elif key_movement.command == KeyCommands.MOVE:
                    if lock_right:
                        left_frame_idx += key_movement.frame_movement
                    else:
                        left_frame_idx += key_movement.frame_movement
                        right_frame_idx += key_movement.frame_movement

                    if left_frame_idx < 0:
                        left_frame_idx = 0
                    elif left_frame_idx > left_video.get_frame_count():
                        left_frame_idx = left_video.get_frame_count() - 1

                    if right_frame_idx < 0:
                        right_frame_idx = 0
                    elif right_frame_idx > right_video.get_frame_count():
                        right_frame_idx = right_video.get_frame_count() - 1


if __name__ == '__main__':
    main()
