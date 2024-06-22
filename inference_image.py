from ultralytics import YOLO

# path of checkpoint
ckpt_path = "runs/detect/train8/weights/best.pt"

# save path
save_path = "result.png"

# test dataset
img = "datasets/laser_v6/valid/images/-_mov-0018_jpg.rf.448855c7abe2a6239b2b46e5ac2f5b18.jpg"

# Load a model
model = YOLO(ckpt_path)

model.predict(img, save = True, imgsz = 640)