from ultralytics import YOLO
import cv2
import time

model = YOLO("/home/markc/Downloads/robosub_2024_v2.engine", "segment")
testIm = cv2.imread("/home/markc/Downloads/mapping.png")

for i in range(10):
    start = time.time()
    results = model(testIm)
    end = time.time()
    print(f"took {(end - start)}s")
# for result in results:
#     boxes = result.boxes
#     masks = result.masks
#     result.show()