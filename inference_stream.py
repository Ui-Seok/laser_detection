import cv2
import argparse

from ultralytics import YOLO


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="model checkpoint path")
    parser.add_argument('--video-path', type=str, default=0, help="test image path")
    parser.add_argument('--image-size', type=str, default=320, help="input image(frame) size")
    
    args = parser.parse_args()
    return args

def main(args):
    ckpt_path = args.checkpoint
    video_path = args.video_path
    
    model = YOLO(ckpt_path, task="detect")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Failed to load video")
        
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            results = model(
                frame, 
                imgsz=args.image_size
                )
            
            vis_frame = results[0].plot()
            
            cv2.imshow("Inference", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_opt()
    main(args)
