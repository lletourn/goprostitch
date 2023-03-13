import argparse
from ctypes import c_bool
import cv2
import logging
import multiprocessing
import numpy as np
# import threading
import time

from goprostitch.framestitcher import FrameParts
from goprostitch.framestitcher import stitch_frames
from goprostitch.inputprocessor import InputPacketType
from goprostitch.inputprocessor import process_input
from goprostitch.outputwriter import process_output


logger = logging.getLogger(__name__)


def handle_final_data(out_video_queue, out_left_audio_queue, out_right_audio_queue, panorama_queue, panoramas, left_audio_packets, right_audio_packets, video_idx, left_audio_idx, right_audio_idx):
    while not panorama_queue.empty():
        panorama = panorama_queue.get()
        panoramas[panorama.idx] = panorama

    while video_idx in panoramas or left_audio_idx in left_audio_packets or right_audio_idx in right_audio_packets:
        if video_idx in panoramas:
            out_video_queue.put(panoramas[video_idx])
            del panoramas[video_idx]
            video_idx += 1

        if left_audio_idx in left_audio_packets:
            out_left_audio_queue.put(left_audio_packets[left_audio_idx])
            del left_audio_packets[left_audio_idx]
            left_audio_idx += 1

        if right_audio_idx in right_audio_packets:
            out_right_audio_queue.put(right_audio_packets[right_audio_idx])
            del right_audio_packets[right_audio_idx]
            right_audio_idx += 1

    return video_idx, left_audio_idx, right_audio_idx


def main():
    parser = argparse.ArgumentParser(description='Run detection on hockey broadcast videos.')
    parser.add_argument("--jobs", type=int, help="Number of workers.")
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
    nb_workers = args.jobs
    output_width = 7060
    output_height = 2400
    crop_x = 670
    crop_y = 385

    logger.info("Setting up input processors")
    left_queue = multiprocessing.Queue(maxsize=10)
    left_processor = multiprocessing.Process(target=process_input, args=(left_video_filename, left_queue))
    # left_processor = threading.Thread(target=process_input, args=(left_video_filename, left_queue))
    right_queue = multiprocessing.Queue(maxsize=10)
    right_processor = multiprocessing.Process(target=process_input, args=(right_video_filename, right_queue))
    # right_processor = threading.Thread(target=process_input, args=(right_video_filename, right_queue))
    left_processor.start()
    right_processor.start()

    out_video_queue = multiprocessing.Queue(maxsize=10)
    out_left_audio_queue = multiprocessing.Queue(maxsize=10)
    out_right_audio_queue = multiprocessing.Queue(maxsize=10)
    stop_encoding = multiprocessing.Value(c_bool, False)
    output_processor = multiprocessing.Process(target=process_output, args=(output_filename, output_width, output_height, stop_encoding, out_video_queue, out_left_audio_queue, out_right_audio_queue))
    output_processor.start()

    logger.info("Setting up camera params")
    camera_left = cv2.detail.CameraParams()
    camera_left.focal = 1380.9097896938035
    camera_left.aspect = 1.0
    camera_left.ppx = 516.5
    camera_left.ppy = 290.5
    camera_left.R = np.array([[9.7980595e-01, -5.8920074e-02, -1.9107230e-01], [-9.6140185e-10,  9.5559806e-01, -2.9467332e-01], [1.9995050e-01, 2.8872266e-01, 9.3630064e-01]], dtype=np.float32)
    camera_left.t = np.array([[0.], [0.], [0.]], dtype=np.float64)

    camera_right = cv2.detail.CameraParams()
    camera_right.focal = 1384.4108312478572
    camera_right.aspect = 1.0
    camera_right.ppx = 516.5
    camera_right.ppy = 290.5
    camera_right.R = np.array([[9.7996306e-01, 5.6246426e-02, 1.9107229e-01], [2.2138309e-08, 9.5929933e-01, -2.8239137e-01], [-1.9917902e-01, 2.7673310e-01, 9.4007784e-01]], dtype=np.float32)
    camera_right.t = np.array([[0.], [0.], [0.]], dtype=np.float64)

    left_video_packets = dict()
    left_audio_packets = dict()
    right_video_packets = dict()
    right_audio_packets = dict()

    logger.info("Finding reference image for white balance correction")
    reference_image = None
    while reference_image is None:
        packet = left_queue.get()
        if packet.type == InputPacketType.AUDIO:
            left_audio_packets[packet.idx] = packet
        elif packet.type == InputPacketType.VIDEO:
            left_video_packets[packet.idx] = packet
            if packet.idx == 0:
                reference_image = packet.data

    logger.info(f"Starting {nb_workers} panorama stitchers")
    worker_processes = list()
    stop_workers = multiprocessing.Value(c_bool, False)

    panorama_queue = multiprocessing.Queue(maxsize=30)
    stitch_queue = multiprocessing.Queue(maxsize=15)
    for i in range(nb_workers):
        args = (stitch_queue, panorama_queue, stop_workers, reference_image, camera_left, camera_right, crop_x, crop_y, output_width, output_height)
        p = multiprocessing.Process(target=stitch_frames, args=args)
        # p = threading.Thread(target=stitch_frames, args=args)
        p.start()
        worker_processes.append(p)
    panoramas = dict()

    logger.info(f"Setting up output video {output_filename}")

    start_time = time.time()
    last_video_idx_throughput = 0
    video_idx = 0
    left_audio_idx = 0
    right_audio_idx = 0
    last_left_video_idx = -1
    last_right_video_idx = -1

    last_video_idx = None
    logger.info("Starting process loop")

    time_deltas = list()
    while left_processor.is_alive() or right_processor.is_alive():
        total_time = time.time()
        while not left_queue.empty():
            packet = left_queue.get()
            if packet.type == InputPacketType.AUDIO:
                left_audio_packets[packet.idx] = packet
            elif packet.type == InputPacketType.VIDEO:
                left_video_packets[packet.idx] = packet
                if packet.idx > last_left_video_idx:
                    last_left_video_idx = packet.idx
        while not right_queue.empty():
            packet = right_queue.get()
            if packet.type == InputPacketType.AUDIO:
                right_audio_packets[packet.idx] = packet
            elif packet.type == InputPacketType.VIDEO:
                right_video_packets[packet.idx] = packet
                if packet.idx > last_right_video_idx:
                    last_right_video_idx = packet.idx

        processing_time_start = time.time()
        if not left_processor.is_alive() and not right_processor.is_alive():
            last_video_idx = last_right_video_idx
            if last_left_video_idx < last_right_video_idx:
                last_video_idx = last_left_video_idx
        elif not left_processor.is_alive():
            last_video_idx = last_left_video_idx
        elif not right_processor.is_alive():
            last_video_idx = last_right_video_idx

        idx_to_delete = list()
        for k, v in left_video_packets.items():
            if k in right_video_packets:
                idx_to_delete.append(k)
                frame_data = FrameParts(left=v.data, right=right_video_packets[k].data, idx=k, pts=v.pts)
                stitch_queue.put(frame_data)

        # Clean up queues past last frame
        if last_video_idx is not None:
            for k in left_video_packets:
                if k > last_video_idx:
                    idx_to_delete.append(k)
            for k in right_video_packets:
                if k > last_video_idx:
                    idx_to_delete.append(k)
        for k in idx_to_delete:
            if k in left_video_packets:
                del left_video_packets[k]
            if k in right_video_packets:
                del right_video_packets[k]

        bef_handler = time.time()
        video_idx, left_audio_idx, right_audio_idx = handle_final_data(out_video_queue, out_left_audio_queue, out_right_audio_queue, panorama_queue, panoramas, left_audio_packets, right_audio_packets, video_idx, left_audio_idx, right_audio_idx)
        aft_handler = time.time()

        time_deltas.append([bef_handler-processing_time_start, aft_handler-processing_time_start, aft_handler-total_time])
        if video_idx >= last_video_idx_throughput+60:  # 60 == ~1 frame
            last_video_idx_throughput = video_idx
            throughput = video_idx / (time.time()-start_time)
            logger.info(f"Processed frame: {video_idx} {throughput:0.2f} ILeft: {left_queue.qsize()} IRight: {right_queue.qsize()} IStitch: {stitch_queue.qsize()} OPano: {panorama_queue.qsize()} OVideo: {out_video_queue.qsize()} OLeft: {out_left_audio_queue.qsize()} ORight: {out_right_audio_queue.qsize()}")

            bef = 0.0
            aft = 0.0
            tot = 0.0
            for a in time_deltas:
                bef += a[0]
                aft += a[1]
                tot += a[2]
            nb = len(time_deltas)
            logger.debug(f"Before: {bef / nb:0.3f} After: {aft / nb:0.3f} Total: {tot / nb:0.3f}")
            time_deltas.clear()

        # scale = 0.1
        # cv2.imshow("Pano", cv2.resize(pano, (int(pano.shape[1]*scale), int(pano.shape[0]*scale))))
        # cv2.imshow("Left", cv2.resize(left_image, (1280, 720)))
        # cv2.imshow("Right", cv2.resize(right_image, (1280, 720)))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    left_processor.join()
    right_processor.join()
    while video_idx != last_video_idx:
        video_idx, left_audio_idx, right_audio_idx = handle_final_data(out_video_queue, out_left_audio_queue, out_right_audio_queue, panorama_queue, panoramas, left_audio_packets, right_audio_packets, video_idx, left_audio_idx, right_audio_idx)
    throughput = video_idx / (time.time()-start_time)
    logger.info(f"Processed final frame: {video_idx} {throughput:0.2f}")

    stop_workers.value = True
    for worker in worker_processes:
        worker.join()

    stop_encoding.value = True
    output_processor.join()


def print_camera_params(camera):
    print(camera.focal)
    print(camera.aspect)
    print(camera.ppx)
    print(camera.ppy)
    print(camera.R)
    print(camera.R.dtype)
    print(camera.t)
    print(camera.t.dtype)


if __name__ == '__main__':
    main()
