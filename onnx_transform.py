from ultralytics import YOLO
import os

ptName_seg = os.path.join("runs", "segment", "train", "weights", "best.pt")
ptName_detect = os.path.join("runs", "detect", "train", "weights", "best.pt")
# Load a model
model_detect = YOLO(ptName_detect)  # load an official model
model_seg = YOLO(ptName_seg)

model_detect.export(format="onnx")
model_seg.export(format="onnx")