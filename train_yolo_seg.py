from ultralytics import YOLO

model = YOLO("yolo11l-seg.pt")
model.train(data = "/project/Pallets_segment/data.yaml", epochs = 200, batch = 4, device = "cuda")

