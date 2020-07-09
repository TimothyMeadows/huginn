import numpy
import argparse
import imutils
import time
from pydarknet import Detector, Image
import cv2
import os


def resize(frame, size=(640, 480)):
    return cv2.resize(frame, size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)


def overlay(frame, x, y, w, h, color=(255, 255, 0), thickness=2, left_label=None, right_label=None):
    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)),
                  (int(x + w / 2), int(y + h / 2)), color, thickness)

    if (left_label is not None):
        cv2.putText(frame, left_label, (int(x - (w / 2)), int(y - (h / 2))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    if (right_label is not None):
        cv2.putText(frame, right_label, (int(x - (w / 2)), int(y + (h / 2))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


weightsPath = os.path.sep.join(["yolov3", "yolov3.weights"])
configPath = os.path.sep.join(["yolov3", "yolov3_gpu.cfg"])
dataPath = os.path.sep.join(["yolov3", "coco.data"])

net = Detector(bytes(configPath, encoding="utf-8"), bytes(weightsPath,
                                                          encoding="utf-8"), 0, bytes(dataPath, encoding="utf-8"))
cap = cv2.VideoCapture("videos/airport.mp4")
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter("output.mp4", fourcc, 24, (800, 600), True)
		
(W, H) = (None, None)
if cap.isOpened():
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    while True:
        (grabbed, frame) = cap.read()
        if not grabbed:
            break

        frame = resize(frame, (800, 600))
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        img_darknet = Image(frame)
        results = net.detect(img_darknet)
        color = (255, 255, 0)

        for cat, score, bounds in results:
            x, y, w, h = bounds
            name = str(cat.decode("utf-8"))
            print(name + ": " + str(score))
            if score > 0.6:
                color = (0, 255, 0)
            elif score < 0.4:
                color = (255, 0, 0)

            overlay(frame, x, y, w, h, color, 1, name, str(round(score, 2)))

        # cv2.imshow("video", frame)
        # if cv2.waitKey(1) == ord('q'):
            #break

        writer.write(frame)
else:
    print("unable to open video")
cap.release()
