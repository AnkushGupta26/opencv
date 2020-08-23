import cv2
import numpy as np
import math
from pynput.keyboard import Key, Controller
keyboard = Controller()

def slope(c1, c2):
    return (float)(c2[1]-c1[1])/(c2[0]-c1[0])


def Distance(c1, c2):
    dist = math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
    return dist

cap = cv2.VideoCapture(1)

# def pas(x):
#     pass
#
# cv2.namedWindow('track')
# cv2.createTrackbar('LH', 'track', 18, 255, pas)
# cv2.createTrackbar('LS', 'track', 40, 255, pas)
# cv2.createTrackbar('LV', 'track', 90, 255, pas)
# cv2.createTrackbar('HH', 'track', 27, 255, pas)
# cv2.createTrackbar('HS', 'track', 255, 255, pas)
# cv2.createTrackbar('HV', 'track', 255, 255, pas)

while True:

    # lh = cv2.getTrackbarPos('LH', 'track')
    # ls = cv2.getTrackbarPos('LS', 'track')
    # lv = cv2.getTrackbarPos('LV', 'track')
    # hh = cv2.getTrackbarPos('HH', 'track')
    # hs = cv2.getTrackbarPos('HS', 'track')
    # hv = cv2.getTrackbarPos('HV', 'track')

    _, frame = cap.read()
    hsvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([95, 150, 90])
    upper = np.array([120, 255, 255])
    mask = cv2.inRange(hsvframe, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    res = cv2.bitwise_and(frame,frame, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 2:
        (x1, y1, w1, h1) = cv2.boundingRect(contours[0])
        (x2, y2, w2, h2) = cv2.boundingRect(contours[1])
        c1 = (x1 + int(w1/2), y1 + int(h1/2))
        c2 = (x2 + int(w2/2), y2 + int(h2/2))
        cv2.line(res, c1, c2, (0, 255, 0), 3)
        # front and back
        if Distance(c1, c2)>300:
            print('w')
            keyboard.press(Key.up)
            keyboard.release(Key.up)

        else:
            print('s')
            keyboard.press(Key.down)
            keyboard.release(Key.down)
        # front and back end

        # Left and Right
        if slope(c1, c2) > 0.3:
            print('a')
            keyboard.press(Key.left)
            keyboard.release(Key.left)
        elif slope(c1, c2) < -0.3:
            print('d')
            keyboard.press(Key.right)
            keyboard.release(Key.right)

    cv2.imshow('Detect', res)
    cv2.waitKey(1)
