This is the repository for the Pallets Detection task from Peer Robotics Vision Intern Online Assessment.

# _Data collection and preprocessing_ #
First of all, I used Grounding DINO and Segment Anything (SAM) as the auto-labeling tool for pallets and ground. For the code just following the file `automated-dataset-annotation-and-evaluation-with-grounding-dino-and-sam.ipynb`.

Later in the RoboFlow, I modify the auto-labeling result manually and then launched the dataset with `train`, `validation`, and `test` dataset with percentage `70%`, `20%`, and `10%` respectively. Also the data augmentation technique including `changing light condition`, `random flip`, and `random crop` are applied to both segmentation and detection dataset.

# _Model training and validation step_ #
For the model, I choose YoloV11 and YoloV11-seg as the object detection model and semantic segmentation model respectively due to their great trade-off between efficiency and accuracy. Based on ultralytics package, I trained both model in epoch 200 and batch size as 4. The pre-trained model I used is the Yolov11-large, which is compatible for edge device under GPU and also has higher accuracy compared to nano model or small model. By running the code `train_yolo.py` and `train_yolo_seg`, the validation metric is shown below for both model:

![image](https://github.com/tungyen/Peer_OA/blob/master/res_img/detetion_pred.png)
![image](https://github.com/tungyen/Peer_OA/blob/master/res_img/segment_pred.png)

# _Transforming to Onnx format for edge device_ #
In this part, I chose to transform both Yolo model to Onnx format for the following usage. The code is specified in `onnx_transform.py`


# _Ros subscribe code_ #