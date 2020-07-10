import numpy
import argparse
import imutils
from PIL import Image as PILImage
import time
from pydarknet import Detector, Image
import cv2
import os
import imagehash

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

def phash(frame):
    frame_rgb = PILImage.fromarray(frame, 'RGB')
    frame_hash = imagehash.average_hash(frame_rgb)
    frame_rgb.close()
    return frame_hash

def overlay(frame, x, y, w, h, color=(255, 255, 0), thickness=2, left_label=None, right_label=None):
    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)),
                  (int(x + w / 2), int(y + h / 2)), color, thickness)

    if (left_label is not None):
        cv2.putText(frame, left_label, (int(x - (w / 2)), int(y - (h / 2) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    if (right_label is not None):
        cv2.putText(frame, right_label, (int(x - (w / 2)), int(y + (h / 2) + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

def overlays(frame, results):
    color = (255, 255, 0)
    for cat, score, bounds in results:
        x, y, w, h = bounds
        name = str(cat.decode("utf-8"))
        #print(name + ": " + str(score))
        if score > 0.6:
            color = (0, 255, 0)
        elif score < 0.4:
            color = (255, 0, 0)

        overlay(frame, x, y, w, h, color, 1, name, str(round(score, 2)))


weightsPath = os.path.sep.join(["yolov3", "yolov3.weights"])
configPath = os.path.sep.join(["yolov3", "yolov3_gpu.cfg"])
dataPath = os.path.sep.join(["yolov3", "coco.data"])

net = Detector(bytes(configPath, encoding="utf-8"), bytes(weightsPath,
                                                          encoding="utf-8"), 0, bytes(dataPath, encoding="utf-8"))
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 24)
		
(W, H) = (640, 480)
skip = False
results = None
previous_frame_hash = None
ham = 1

if cap.isOpened():
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    while True:
        (error, frame) = cap.read()
        if not error:
            print("camera read error")
            break

        frame = brightness_contrast(frame)
        frame_hash = phash(frame)
        if previous_frame_hash is not None:
            shifts = previous_frame_hash - frame_hash
            if shifts <= ham:
                if results is not None:
                    skip = True

        if skip and results is not None:
            overlays(frame, results)

            skip = False
            cv2.imshow("video", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        img_darknet = Image(frame)
        results = net.detect(img_darknet)
        overlays(frame, results)

        previous_frame_hash = frame_hash
        cv2.imshow("video", frame)
        if cv2.waitKey(1) == ord('q'):
            break
else:
    print("unable to open camera")
cap.release()
