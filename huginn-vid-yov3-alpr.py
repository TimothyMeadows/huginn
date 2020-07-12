import numpy
import argparse
import imutils
import time
#from PIL import Image as PILmage
from pydarknet import Detector, Image
from openalpr import Alpr
import cv2
import os


def resize(frame, size=(640, 480)):
    return cv2.resize(frame, size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)


def overlay(frame, x, y, w, h, color=(255, 255, 0), thickness=2, left_label=None, right_label=None):
    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)),
                  (int(x + w / 2), int(y + h / 2)), color, thickness)

    if (left_label is not None):
        cv2.putText(frame, left_label, (int(x - (w / 2)), int(y - (h / 2) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    if (right_label is not None):
        cv2.putText(frame, right_label, (int(x - (w / 2)), int(y + (h / 2) + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

#def crop(frame, x, y, w, h):
    #img = PILmage.fromarray(numpy.uint8(frame)).convert('RGB')
    #img = img.crop((int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)))
    #return img


def crop(frame, x, y, w, h):
    return frame[y:y+h, x:x+w]


weightsPath = os.path.sep.join(["yolov3", "yolov3.weights"])
configPath = os.path.sep.join(["yolov3", "yolov3_gpu.cfg"])
dataPath = os.path.sep.join(["yolov3", "coco.data"])

net = Detector(bytes(configPath, encoding="utf-8"), bytes(weightsPath,
                                                          encoding="utf-8"), 0, bytes(dataPath, encoding="utf-8"))

alpr = Alpr("gb", "/etc/openalpr/openalpr.conf",
            "/usr/share/openalpr/runtime_data")

if not alpr.is_loaded():
    print("Error loading OpenALPR")
    exit()

alpr.set_top_n(1)
alpr.set_default_region('gb')
alpr.set_detect_region(False)

cap = cv2.VideoCapture("videos/traffic-uk.mp4")
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter("output.mp4", fourcc, 24, (1980, 1024), True)

(W, H) = (None, None)
if cap.isOpened():
    #cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    while True:
        (grabbed, frame) = cap.read()
        if not grabbed:
            break

        frame = resize(frame, (1980, 1024))
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        _, img_alpr = cv2.imencode('.jpg', frame)
        results = alpr.recognize_array(img_alpr.tostring())
        for plate in results['results']:
            for candidate in plate['candidates']:
                if candidate['matches_template']:
                    x = int(min(plate['coordinates'],
                                key=lambda ev: ev['x'])['x'])
                    y = int(min(plate['coordinates'],
                                key=lambda ev: ev['y'])['y'])
                    w = int(max(plate['coordinates'],
                                key=lambda ev: ev['x'])['x'])
                    h = int(max(plate['coordinates'],
                                key=lambda ev: ev['y'])['y'])

                    print("Cords: " + str((x, y, w, h)) + " Plate: " + str(candidate['plate']) + " Confidence: " + str(
                        candidate['confidence']) + " Region: " + str(plate['region']))

                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 255), 2)
                    cv2.putText(frame, str(
                        candidate['plate']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

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

        #cv2.imshow("video", frame)
        # if cv2.waitKey(1) == ord('q'):
            # break

        writer.write(frame)
else:
    print("unable to open video")
cap.release()
