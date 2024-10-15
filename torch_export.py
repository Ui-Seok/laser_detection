from ultralytics import YOLO

ckpt_path = "runs/detect/train_ori_ep300_ext_data/weights/best.pt"

model = YOLO(ckpt_path)

model.export(
    format="engine",
    # half=True
)