from ultralytics import YOLO

ckpt_path = "runs/detect/face_n_laser_480/weights/best.pt"

model = YOLO(ckpt_path)

model.export(
    format="engine",
    # half=True
)