import cv2
import numpy as np

img = cv2.imread('girl.png', 1)

panel = np.zeros([100, 700], np.uint8)
cv2.namedWindow("panel")


def nothing(x):
    pass


cv2.createTrackbar("L - h", "panel", 0, 179, nothing)
cv2.createTrackbar("U - h", "panel", 179, 179, nothing)

cv2.createTrackbar("L - s", "panel", 0, 255, nothing)
cv2.createTrackbar("U - s", "panel", 255, 255, nothing)

cv2.createTrackbar("L - v", "panel", 0, 255, nothing)
cv2.createTrackbar("U - v", "panel", 255, 255, nothing)
while True:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lh = cv2.getTrackbarPos("L - h", "panel")
    uh = cv2.getTrackbarPos("U - h", "panel")

    ls = cv2.getTrackbarPos("L - s", "panel")
    us = cv2.getTrackbarPos("U - s", "panel")

    lv = cv2.getTrackbarPos("L - v", "panel")
    uv = cv2.getTrackbarPos("U - v", "panel")

    lower_green = np.array([lh, ls, lv])
    upper_green = np.array([uh, us, uv])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(img, img, mask=mask)
    fg = cv2.bitwise_and(img, img, mask=mask_inv)
    cv2.imshow("panel", panel)
    cv2.imshow("bg", bg)
    cv2.imshow("fg", fg)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

#sources: Pysource on youtube