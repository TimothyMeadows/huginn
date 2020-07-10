import cv2
import numpy
from openalpr import Alpr

alpr = Alpr("us", "/etc/openalpr/openalpr.conf",
            "/usr/share/openalpr/runtime_data")

if not alpr.is_loaded():
    print("Error loading OpenALPR")
else:
    print("Using OpenALPR " + alpr.get_version())

alpr.set_top_n(7)
alpr.set_default_region("wa")
alpr.set_detect_region(False)
jpeg_bytes = open("images/ca-plate.png", "rb").read()
results = alpr.recognize_array(jpeg_bytes)

print("Image size: %dx%d" %(results['img_width'], results['img_height']))
print("Processing Time: %f" % results['processing_time_ms'])