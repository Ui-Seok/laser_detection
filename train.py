import argparse

from ultralytics import YOLO


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-yaml-file', type=str, default="ultralytics/ultralytics/cfg/models/v8/yolov8.yaml", help="model yaml file path")
    parser.add_argument('--checkpoint', type=str, required=True, help="model checkpoint path")
    parser.add_argument('--data-yaml-file', type=str, default="datasets/data.yaml", help="dataset yaml file path")
    
    parser.add_argument('--epochs', type=int, default=200, help="training epochs")
    parser.add_argument('--img-size', type=int, default=640, help="training image size")
    
    parser.add_argument('--name', type=str, default=None, help="training output directory name")
    
    args = parser.parse_args()
    return args

def main(args):
    model_yaml_file_path = args.model_yaml_file
    ckpt_path = args.checkpoint
    data_yaml_file_path = args.data_yaml_file

    model = YOLO(model_yaml_file_path)
    # model = YOLO(ckpt_path)
    model = model.load(ckpt_path)

    results = model.train(
        data = data_yaml_file_path, 
        epochs = args.epochs, 
        imgsz = args.img_size,
        name = args.name,
        cos_lr = True
        )
    
    
if __name__ == "__main__":
    args = parse_opt()
    main(args)
