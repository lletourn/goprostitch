import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np
import logging

from goprostitch.framereader import FrameReader

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChangeEvent:
    frame_idx: int
    pts_time: float
    pts: int
    ocr_text: str | None = None


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y1:self.y2, self.x1:self.x2].copy()


# ---------------------------------------------------------------------------
# Bounding box selection UI
# ---------------------------------------------------------------------------

class PointSelector:
    """Collects two points via mouse click + 'a' key to define a bounding box."""

    def __init__(self, window_name: str):
        self.window_name = window_name
        self.points: list[tuple[int, int]] = []
        self.current_mouse: tuple[int, int] | None = None

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_mouse = (x, y)

    def select(self, frame_reader: FrameReader) -> BBox:
        frame = frame_reader.get_frame(1).frame
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1080)
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        logging.info("Click on the FIRST corner of the digit region, then press 'a'.")
        logging.info("Click on the SECOND corner, then press 'a'.")

        while len(self.points) < 2:
            display = frame.copy()

            # draw confirmed points
            for pt in self.points:
                cv2.circle(display, pt, 5, (0, 255, 0), -1)

            # draw live crosshair
            if self.current_mouse is not None:
                mx, my = self.current_mouse
                h, w = display.shape[:2]
                cv2.line(display, (mx, 0), (mx, h), (0, 255, 255), 1)
                cv2.line(display, (0, my), (w, my), (0, 255, 255), 1)

            # if one point confirmed and mouse is live, draw provisional rectangle
            if len(self.points) == 1 and self.current_mouse is not None:
                cv2.rectangle(display, self.points[0], self.current_mouse, (0, 255, 0), 2)

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('a') and self.current_mouse is not None:
                self.points.append(self.current_mouse)
                logging.info(f"  Point {len(self.points)} confirmed: {self.current_mouse}")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                sys.exit("Aborted by user.")

        cv2.destroyAllWindows()

        (ax, ay), (bx, by) = self.points
        return BBox(
            x1=min(ax, bx),
            y1=min(ay, by),
            x2=max(ax, bx),
            y2=max(ay, by),
        )


def select_bbox(frame_reader: FrameReader) -> BBox:
    """Show the first frame and let the user pick a bounding box."""
    selector = PointSelector("Select digit region")
    return selector.select(frame_reader)


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def binarize(crop: np.ndarray) -> np.ndarray:
    """Convert to grayscale and apply Otsu threshold."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, binary = cv2.threshold(gray, 182, 255, cv2.THRESH_BINARY)
    # cv2.imshow("Gray", gray)
    # cv2.imshow("OTSU", binary)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return binary


def compute_change_metric(prev_crop: np.ndarray, curr_crop: np.ndarray) -> float:
    """
    Return a scalar measuring how different two binarized crops are.
    Uses mean absolute difference on Otsu-thresholded images.
    """
    b1 = binarize(prev_crop)
    b2 = binarize(curr_crop)
    return float(np.mean(np.abs(b1.astype(np.float32) - b2.astype(np.float32))))


# ---------------------------------------------------------------------------
# OCR (optional)
# ---------------------------------------------------------------------------

_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    return _ocr_engine


def ocr_crop(crop: np.ndarray) -> str:
    """Run PaddleOCR on a crop and return the concatenated recognized text."""
    engine = get_ocr_engine()
    result = engine.ocr(crop, cls=False)
    if not result or not result[0]:
        return ""
    texts = [line[1][0] for line in result[0]]
    return " ".join(texts)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def detect_changes(
    frame_reader: FrameReader,
    bbox: BBox,
    threshold: float = 30.0,
    use_ocr: bool = False,
    img_output: str = None
) -> list[ChangeEvent]:
    """
    Iterate over frames, detect digit changes in the bbox crop,
    and return a list of ChangeEvents.
    """
    events: list[ChangeEvent] = []
    prev_crop: np.ndarray | None = None
    frame_count = 0

    # window_name = "Change"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, 1280, 720)
    last_found_frame = -100
    for frame_idx in range(0, frame_reader.get_frame_count(), 1):
        frame = frame_reader.get_frame(frame_idx)
        frame_bgr = frame.frame
        curr_crop = bbox.crop(frame_bgr)

        if prev_crop is not None:
            metric = compute_change_metric(prev_crop, curr_crop)

            # print(metric)
            # cv2.imshow(window_name, frame_bgr)
            # key = cv2.waitKey() & 0xFF
            # if key == ord('q'):
            #     cv2.destroyAllWindows()
            #     sys.exit("Aborted by user.")
            if metric > threshold:
                if (frame_idx - last_found_frame) <= 5:
                    continue
                last_found_frame = frame_idx
                
                if img_output is not None:
                    cv2.imwrite(f"{img_output}/img-{frame_idx:06d}.jpg", frame_bgr)

                ocr_text = None
                if use_ocr:
                    ocr_text = ocr_crop(curr_crop)

                event = ChangeEvent(
                    frame_idx=frame_idx,
                    pts=frame.frame_data.pts,
                    pts_time=frame.frame_data.time,
                    ocr_text=ocr_text,
                )
                events.append(event)

                # cv2.imshow(window_name, frame_bgr)
                # key = cv2.waitKey() & 0xFF
                # if key == ord('q'):
                #     cv2.destroyAllWindows()
                #     sys.exit("Aborted by user.")

                if len(events) % 50 == 0:
                    logging.info(f"  ... {len(events)} changes detected so far (frame {frame_idx})")

        prev_crop = curr_crop

    logging.info(f"Scanned {frame_count} frames, found {len(events)} change events.")
    return events


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GoPro drift detector")
    parser.add_argument('--video', required=True, type=str, help='Video')
    parser.add_argument("--ocr", action="store_true", help="Run PaddleOCR on each change")
    parser.add_argument("--threshold", type=float, default=35.0, help="Mean abs diff threshold to count as a change (default: 30.0)")
    parser.add_argument("--output", type=str, default=None, help="Output directory)")
    parser.add_argument("-l", "--log", help="log level (default: info)", choices=["debug", "info", "warning", "error", "critical"], default="info")
    args = parser.parse_args()

    logdatefmt = '%Y%m%dT%H:%M:%S'
    logformat = '%(asctime)s.%(msecs)03d [%(levelname)s] -%(name)s- -%(threadName)s- : %(message)s'
    logging.basicConfig(datefmt=logdatefmt, format=logformat, level=args.log.upper())

    with FrameReader(args.video, frame_skips_before_seek=59) as frame_reader:
        logging.info("Get box...")
        bbox = select_bbox(frame_reader)
        logging.info(f"Bounding box: ({bbox.x1}, {bbox.y1}) -> ({bbox.x2}, {bbox.y2})")

        events = detect_changes(frame_reader, bbox, threshold=args.threshold, use_ocr=args.ocr, img_output=args.output)

    # save results
    output_data = {
        "video": args.video,
        "bbox": asdict(bbox),
        "threshold": args.threshold,
        "ocr_enabled": args.ocr,
        "events": [asdict(e) for e in events],
    }
    with open(f"{args.output}/out.json", "w") as f:
        json.dump(output_data, f, indent=2)

    # logging.info first/last few events as a quick summary
    if events:
        logging.info(f"\nFirst change: frame {events[0].frame_idx}, pts_time={events[0].pts_time:.4f}s")
        logging.info(f"Last change:  frame {events[-1].frame_idx}, pts_time={events[-1].pts_time:.4f}s")
        if len(events) > 1:
            deltas = [events[i+1].pts_time - events[i].pts_time for i in range(len(events)-1)]
            logging.info(f"Mean interval between changes: {sum(deltas)/len(deltas):.4f}s")
            logging.info(f"Min interval:  {min(deltas):.4f}s")
            logging.info(f"Max interval:  {max(deltas):.4f}s")


if __name__ == "__main__":
    main()
