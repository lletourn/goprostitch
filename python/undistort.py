import cv2  # type: ignore
import numpy
import sys

K = numpy.array([[1768.8181032986872, 0, 1.9237091208494583e+03], [0, 1.7679751216421994e+03, 1.0957257960681286e+03], [0, 0, 1]], dtype=numpy.float64)
distcoeffs = numpy.array([-2.3918513751023274e-01, 6.9352326700516775e-02, -5.8377437741893799e-05, 1.9806067043683098e-04,-1.0478418302033846e-02], dtype=numpy.float64)

img = cv2.imread(sys.argv[1])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, distcoeffs, (w, h), 0, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(K, distcoeffs, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

print(dst.shape)
cv2.imshow("Corrected", dst)
cv2.imshow("Orig", img)
cv2.waitKey(0)

cv2.imwrite(sys.argv[2], dst)
