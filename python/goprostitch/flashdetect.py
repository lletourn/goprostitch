import argparse
import cv2
import logging
import numpy as np

from goprostitch.framereader import FrameReader


FRAME_SKIP = 5


def find_flash_region(
    frame_reader: FrameReader,
    box_size: int = 50,
) -> tuple[int, int, int, int] | None:
    frame_idx = 0
    img_details = frame_reader.get_frame(frame_idx)
    img = img_details.frame

    h, w = img.shape[:2]
    half = box_size // 2
    mouse_pos = [w // 2, h // 2]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos[0] = x
            mouse_pos[1] = y

    win = "Select flash center (press 'a' to accept, 'q' to cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        display = img.copy()
        cx, cy = mouse_pos
        bx = max(0, min(cx - half, w - box_size))
        by = max(0, min(cy - half, h - box_size))
        cv2.rectangle(display, (bx, by), (bx + box_size, by + box_size), (0, 255, 0), 2)
        cv2.imshow(win, display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("a"):
            cv2.destroyWindow(win)
            return (bx, by, box_size, box_size)
        elif key == ord("n"):
            frame_idx += FRAME_SKIP
            img_details = frame_reader.get_frame(frame_idx)
            img = img_details.frame
            display = img.copy()
            cx, cy = mouse_pos
            bx = max(0, min(cx - half, w - box_size))
            by = max(0, min(cy - half, h - box_size))
            cv2.rectangle(display, (bx, by), (bx + box_size, by + box_size), (0, 255, 0), 2)
            cv2.imshow(win, display)
        elif key == ord("q"):
            cv2.destroyWindow(win)
            return None


def detect_flash_changes(
    frame_reader: FrameReader,
    region: tuple[int, int, int, int],
    brightness_threshold: float = 200.0,
    change_threshold: float = 50.0,
) -> list[float]:
    x, y, w, h = region

    means = []
    logging.info("Finding crop means")
    for frame_idx in range(0, frame_reader.get_frame_count(), FRAME_SKIP):
        img = frame_reader.get_frame(frame_idx)
        crop = img.frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        means.append({"mean": float(gray.mean()), "time": img.frame_data.time})

    flash_on = []
    last_flash = -1
    logging.info("Computing frame crop differences")
    for i in range(1, len(means)):
        diff = means[i]['mean'] - means[i - 1]['mean']
        delta = means[i]['time'] - means[i - 1]['time']
        if diff > change_threshold and means[i]['mean'] >= brightness_threshold:
            if last_flash == -1:
                last_flash = means[i]['time']
            else:
                flash_on.append(means[i]['time'] - last_flash)
                last_flash = means[i]['time']
        elif -diff > change_threshold and means[i - 1]['mean'] >= brightness_threshold:
            pass

    return flash_on

def parse_region(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("Region must be x,y,w,h (4 comma-separated ints)")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError("Region values must be integers")


def main():
    parser = argparse.ArgumentParser(description='Detect flash transitions')
    parser.add_argument('--video', required=True, type=str, help='Video')
    parser.add_argument("--brightness", type=float, default=200.0, help="Mean brightness threshold for 'flash on' (default: 200).")
    parser.add_argument("--change", type=float, default=50.0, help="Min brightness delta between frames (default: 50).")
    parser.add_argument("-l", "--log", help="log level (default: info)", choices=["debug", "info", "warning", "error", "critical"], default="info")
    args = parser.parse_args()

    logdatefmt = '%Y%m%dT%H:%M:%S'
    logformat = '%(asctime)s.%(msecs)03d [%(levelname)s] -%(name)s- -%(threadName)s- : %(message)s'
    logging.basicConfig(datefmt=logdatefmt, format=logformat, level=args.log.upper())

    with FrameReader(args.video, frame_skips_before_seek=59) as frame_reader:
        logging.info("Finding center...")
        center_crop = find_flash_region(frame_reader)
        logging.info("Center: %s,%s,%s,%s", center_crop[0], center_crop[1], center_crop[2], center_crop[3])
        result = detect_flash_changes(frame_reader, center_crop, args.brightness, args.change)
    logging.info("Done")
    for res in result:
        print(res)


if __name__ == "__main__":
    main()
