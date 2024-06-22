import torch

from ultralytics import YOLO

model_yaml_file_path = "ultralytics/ultralytics/cfg/models/v8/yolov8-cls-resnet50.yaml"
# ckpt_path = "ultralytics/pretrained/yolov8n.pt"
ckpt_path = "runs/detect/train7/weights/best.pt"
data_yaml_file_path = "datasets/data.yaml"

model = YOLO(model_yaml_file_path)
model = YOLO(ckpt_path)

results = model.train(
    data=data_yaml_file_path, 
    epochs = 100, 
    imgsz=640, 
    stream_buffer=True, 
    cos_lr=True
    )