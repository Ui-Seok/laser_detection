from ultralytics import YOLOv10

# model = YOLOv10()
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10n.pt')

model.train(
    data='datasets/data.yaml', 
    epochs=500,
    optimizer="AdamW",
    batch=16, 
    imgsz=480,
    cos_lr=True,
    name="yolov10_train",
    iou = 0.5,
    lr0 = 0.001,
    lrf = 0.0001
    )