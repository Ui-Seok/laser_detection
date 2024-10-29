import argparse
from ultralytics import YOLO

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-yaml-file', type=str, default="datasets/data.yaml")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=480)
    parser.add_argument('--optimizer', type=str, default='auto')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--freeze', type=int, default=10)
    parser.add_argument('--mosaic', type=float, default=1.0)
    
    args = parser.parse_args()
    return args

def main(args):
    model = YOLO(args.checkpoint)

    freeze_list = list(model.model.parameters())[:args.freeze]
    for param in freeze_list:
        param.requires_grad = False
    
    model.model.nc = 2
    
    results = model.train(
        data=args.data_yaml_file,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        optimizer=args.optimizer,
        name=args.name,
        cos_lr=True,
        freeze=args.freeze,
        mosaic=args.mosaic,
        perspective=0.0003,
        hsv_s=0.4,
        hsv_v=0.4,
        scale=0.1,
        iou=0.5,
        lr0=0.001,
        lrf=0.0001,
        label_smoothing=0.1,       # 라벨 스무딩
        mixup=0.1,                 # mixup 추가
    )
    
if __name__ == "__main__":
    args = parse_opt()
    main(args)