from ultralytics import YOLO

model = YOLO("yolo11l.pt")
model.train(data = "/project/Pallets_detection/data.yaml", epochs = 200, batch = 4, device = "cuda")

