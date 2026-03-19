"""Detect the angle of the far rink boards by clicking two points on each image.

Click two points on the far rink board edge in each image. The script computes
the rotation angle needed to make that line horizontal, shows a preview of the
rotated result, and prints the angles.

Usage:
    python detect_board_angle.py <left_image> <right_image>
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def get_two_points(window_name: str, image: np.ndarray) -> list[tuple[int, int]]:
    """Display an image and let the user click two points on the far board.

    Returns:
        List of two (x, y) points in original image coordinates.
    """
    points: list[tuple[int, int]] = []
    display_img = image.copy()

    # Scale down for display if image is large
    max_display_w = 1920
    h, w = image.shape[:2]
    scale = min(1.0, max_display_w / w)
    disp_w = int(w * scale)
    disp_h = int(h * scale)

    def on_mouse(event: int, x: int, y: int, flags: int, param: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            # Convert display coords back to original image coords
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            points.append((orig_x, orig_y))
            # Draw on display image
            cv2.circle(display_img, (orig_x, orig_y), int(8 / scale), (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(display_img, points[0], points[1], (0, 255, 0), int(3 / scale))
            redraw()

    def redraw() -> None:
        shown = cv2.resize(display_img, (disp_w, disp_h))
        cv2.imshow(window_name, shown)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    print(f"  Click two points on the far rink board in '{window_name}', then press any key.")

    while True:
        key = cv2.waitKey(50)
        if key != -1 and len(points) >= 2:
            break
        if key == 27:  # ESC to abort
            cv2.destroyWindow(window_name)
            print("  Aborted.")
            sys.exit(0)

    cv2.destroyWindow(window_name)
    return points


def compute_angle(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    dx = np.abs(p2[0] - p1[0])
    dy = np.abs(p2[1] - p1[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image around its center by the given angle."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def show_preview(window_name: str, image: np.ndarray) -> None:
    """Show a scaled preview and wait for a keypress."""
    max_display_w = 1920
    h, w = image.shape[:2]
    scale = min(1.0, max_display_w / w)
    shown = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.imshow(window_name, shown)
    print(f"  Preview shown in '{window_name}'. Press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect rink board angle by clicking two points")
    parser.add_argument("left", help="Path to left undistorted image")
    parser.add_argument("right", help="Path to right undistorted image")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for rotated images (default: same as left image)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.left).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, float] = {}

    for label, path in [("left", args.left), ("right", args.right)]:
        print(f"\n=== {label.upper()} image: {path} ===")
        image = cv2.imread(path)
        if image is None:
            print(f"  ERROR: Could not read {path}")
            sys.exit(1)

        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")

        points = get_two_points(f"{label} - click far board", image)
        p1, p2 = points[0], points[1]
        print(f"  Point 1: {p1}")
        print(f"  Point 2: {p2}")
        is_neg_angle = True
        if p2[0] < p1[0]:
            is_neg_angle = False

        angle = compute_angle(p1, p2)
        if is_neg_angle:
            angle = -1 * angle
        print(f"  Board angle: {angle:.3f} deg")

        rotated = rotate_image(image, angle)
        show_preview(f"{label} - rotated ({angle:.1f} deg)", rotated)

        out_path = output_dir / f"{label}_rotated.png"
        cv2.imwrite(str(out_path), rotated)
        print(f"  Saved {out_path}")

        results[label] = angle

    print(f"\n=== Summary ===")
    print(f"  Left rotation:  {results['left']:.3f} deg")
    print(f"  Right rotation: {results['right']:.3f} deg")


if __name__ == "__main__":
    main()
