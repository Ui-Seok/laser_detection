from ultralytics import YOLO

# ckpt_path = "ultralytics/pretrained/yolov8n.pt"   # 225 layers, 3157200 parameters, 0 gradients, 8.9 GFLOPs
# ckpt_path = "runs/detect/train_cls_ep200/weights/best.pt"   # 225 layers, 3011043 parameters, 0 gradients, 8.2 GFLOPs
# ckpt_path = "runs/detect/train_res50_ep200/weights/best.pt"   # 225 layers, 3011043 parameters, 0 gradients, 8.2 GFLOPs
ckpt_path = "runs/detect/train_ori_ep300_ext_data/weights/best.pt"   # 225 layers, 3011043 parameters, 0 gradients, 8.2 GFLOPs

model = YOLO(ckpt_path)
model.info(True)