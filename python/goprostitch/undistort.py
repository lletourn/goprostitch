import cv2  # type: ignore
import json
import numpy
import sys

calibration_filename = sys.argv[1]
input_img_filename = sys.argv[2]
output_img_filename = sys.argv[3]

with open(calibration_filename, "r") as f:
    data = json.load(f)
    K = numpy.array(data["K"], dtype=numpy.float64)
    distcoeffs = numpy.array(data["D"], dtype=numpy.float64)

img = cv2.imread(input_img_filename)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, distcoeffs, (w, h), 0, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(K, distcoeffs, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

print(dst.shape)
cv2.namedWindow("Corrected", cv2.WINDOW_NORMAL)
cv2.namedWindow("Orig", cv2.WINDOW_NORMAL)
cv2.imshow("Corrected", dst)
cv2.imshow("Orig", img)
cv2.waitKey(0)

cv2.imwrite(output_img_filename, dst)
