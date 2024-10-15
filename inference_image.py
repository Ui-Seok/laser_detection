import argparse

from ultralytics import YOLO


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="model checkpoint path")
    parser.add_argument('--test-image', type=str, required=True, help="test image path")

    parser.add_argument('--img-size', type=int, default=640, help="training image size")
    
    args = parser.parse_args()
    return args

def main(args):
    ckpt_path = args.checkpoint
    
    test_image = args.test_image
    
    model = YOLO(ckpt_path)
    model.predict(
        test_image, 
        save=True, 
        imgsz=args.img_size
        )
    

if __name__ == "__main__":
    args = parse_opt()
    main(args)
