import os
from ultralytics import YOLO
import cv2

image_name = "/project/Pallets_segment/test/images/1523528-9928_jpg.rf.c3c0b7d079251f6ea4cfd4305e66e11b.jpg"
ptName = os.path.join("runs", "segment", "train", "weights", "best.pt")
model = YOLO(ptName)
result = model(image_name, save=True, conf=0.5)