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


def crop(frame, x, y, width, height):
    return frame[y:y+height, x:x+width]


def resize(frame, size=(640, 480)):
    return cv2.resize(frame, size, fx=0,fy=0, interpolation = cv2.INTER_CUBIC)


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


resolution = (1280, 768)
width, height = resolution
cap = cv2.VideoCapture("videos/crowd.mp4")
net = jetson.inference.detectNet("ssd-mobilenet-v2", sys.argv, 0.1)
if cap.isOpened():
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

    while True:
        _, frame = cap.read()
        frame = resize(frame, resolution)
        cuda_img = cuda(bgr2rgba(frame))
        detections = net.Detect(cuda_img, width, height, "box,labels,conf")
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

        cv2.imshow("video", frame)
        if cv2.waitKey(1) == ord('q'):
            break
else:
    print("video open failed")

cv2.destroyAllWindows()
