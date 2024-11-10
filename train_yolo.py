from ultralytics import YOLO

model = YOLO("yolo11l.pt")
model.train(data = "/project/dataset/data.yaml", epochs = 100, batch = 4, device = "cuda")