import os
from ultralytics import YOLO
import cv2

image_name = "/project/Pallets_detection/test/images/3164402-2943_jpg.rf.964e945a2fea7bdc6a3939fdaaad969b.jpg"
ptName = os.path.join("runs", "detect", "train", "weights", "best.pt")
model = YOLO(ptName)
image = cv2.imread(image_name)
pred = model(image)

confidence_threshold = 0.5
for p in pred:
    for bbox in p.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        class_id = int(class_id)
        color = (255, 0, 0)
        if score < confidence_threshold: continue
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        label = f'id: {class_id}, score: {score:.2f} '
        cv2.putText(image, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    file_name = os.path.basename(image_name)[:-4]
    cv2.imwrite('./res_img/detection_pred.png', image)