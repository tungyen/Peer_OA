import os
from ultralytics import YOLO
import cv2

image_name = "/project/Pallets_segment/test/images/1523528-9928_jpg.rf.c3c0b7d079251f6ea4cfd4305e66e11b.jpg"
ptName = os.path.join("runs", "segment", "train", "weights", "best.pt")
model = YOLO(ptName)
result = model(image_name, save=True, conf=0.5, project="seg_save", name="")

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps  # a list contains map50-95(B) of each category
metrics.seg.map  # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps  # a list contains map50-95(M) of each category