# transfer_train.py
import argparse
from ultralytics import YOLO
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-yaml-file', type=str, default="datasets/data.yaml")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--stage', type=int, default=1)  # 스테이지 구분을 위한 인자
    parser.add_argument('--freeze', type=int, default=10)
    
    args = parser.parse_args()
    return args

def main(args):
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} not found")
        return 1
        
    if not os.path.exists(args.data_yaml_file):
        print(f"Error: Data YAML file {args.data_yaml_file} not found")
        return 1
    
    print(f"Loading model from {args.checkpoint}")
    model = YOLO(args.checkpoint)
    
    print(f"Starting training stage {args.stage}...")
    
    # Stage별 다른 하이퍼파라미터 설정
    if args.stage == 1:
        lr0 = 0.0005
        lrf = 0.00005
    else:  # stage 2
        lr0 = 0.00005
        lrf = 0.000005
    
    try:
        results = model.train(
            data=args.data_yaml_file,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.img_size,
            optimizer=args.optimizer,
            name=args.name,
            cos_lr=True,
            freeze=args.freeze,
            mosaic=0.0,
            iou=0.5,
            lr0=lr0,
            lrf=lrf,
            warmup_epochs=3,
            label_smoothing=0.1,
            degrees=5.0,
            scale=0.1,
            perspective=0.0003,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.4,
            translate=0.1
        )
        print(f"Stage {args.stage} completed successfully")
        return 0
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return 1
    
if __name__ == "__main__":
    args = parse_opt()
    exit_code = main(args)
    exit(exit_code)