import sys
import os
import time
import argparse
import numpy
import cv2

import jetson.inference
import jetson.utils


def bgr2rgba(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)


def bgr2gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)


def brightness_contrast(frame, brightness=255, contrast=127):
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buffer = cv2.addWeighted(frame, alpha_b, frame, 0, gamma_b)
    else:
        buffer = frame.copy()

    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buffer = cv2.addWeighted(buffer, alpha_c, buffer, 0, gamma_c)

    return buffer


def crop(frame, x, y, width, height):
    return frame[y:y+height, x:x+width]


def cuda(frame):
    buffer = jetson.utils.cudaFromNumpy(frame)
    return buffer


def overlay(frame, x, y, width, height, color=(255, 255, 0), thickness=2, left_label=None, right_label=None):
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)

    if (left_label is not None):
        cv2.putText(frame, left_label, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), thickness)

    if (right_label is not None):
        cv2.putText(frame, right_label, (x + width - len(right_label) * 9, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), thickness)


cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 24)

time.sleep(2.0)  # gives camera time to start up
net = jetson.inference.detectNet("ssd-mobilenet-v2", sys.argv, 0.1)
if cap.isOpened():
    cv2.namedWindow("camera", cv2.WINDOW_AUTOSIZE)

    while True:
        _, frame = cap.read()
        cuda_img = cuda(bgr2rgba(brightness_contrast(frame)))
        detections = net.Detect(cuda_img, 640, 480, "box,labels,conf")
        color = (255, 255, 0)

        for detection in detections:
            if detection.Confidence > 0.6:
                color = (0, 255, 0)
            elif detection.Confidence < 0.4:
                color = (255, 0, 0)

            overlay(frame,
                    round(detection.Left),
                    round(detection.Top),
                    round(detection.Width),
                    round(detection.Height),
                    color, 1,
                    str(detection.ClassID),
                    str(round(detection.Confidence, 2)))

        cv2.imshow("camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break
else:
    print("camera open failed")

cv2.destroyAllWindows()
