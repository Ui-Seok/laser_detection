import cv2
import argparse
import numpy as np

from ultralytics import YOLO


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="runs/new_transfer_single_stage/weights/best.pt", help="model checkpoint path")
    parser.add_argument('--video-path', type=str, default=0, help="test image path")
    parser.add_argument('--image-size', type=str, default=640, help="input image(frame) size")
    parser.add_argument('--conf', type=float, default=0.1, help="input image(frame) size")
    parser.add_argument('--iou', type=float, default=0.4, help="input image(frame) size")
    parser.add_argument('--augment', type=bool, default=False, help="input image(frame) size")
    parser.add_argument('--agnostic_nms', type=bool, default=False, help="input image(frame) size")
    
    args = parser.parse_args()
    return args

# def apply_preprocessing(frame):
#     gamma = 0.8
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) ** 255
#                       for i in np.arange(0, 256)]).astype("uint8")
#     frame_gamma = cv2.LUT(frame, table)

#     frame_lab = cv2.cvtColor(frame_gamma, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(frame_lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     l = clahe.apply(l)
#     frame_lab = cv2.merge((l, a, b))
#     frame_enhanced = cv2.cvtColor(frame_lab, cv2.COLOR_LAB2BGR)

#     frame_filtered = cv2.bilateralFilter(frame_enhanced, 9, 75, 75)

#     return frame_filtered

######################################################################

# def adjust_brightness_contrast(frame, brightness=0, contrast=1.0):
#     return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

# def nothing(x):
#     pass

# cv2.namedWindow('Controls')
# cv2.createTrackbar('Brightness', 'Controls', 0, 100, nothing)
# cv2.createTrackbar('Contrast', 'Controls', 100, 200, nothing)

######################################################################

def enhance_laser_visibility(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:,:,2] = clahe.apply(hsv[:,:,2])

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def is_point_in_bbox(laser_box, face_box):
    # Get center point of laser box
    laser_center_x = (laser_box[0] + laser_box[2]) / 2
    laser_center_y = (laser_box[1] + laser_box[3]) / 2
    
    # Check if laser center point is within face box
    if (face_box[0] <= laser_center_x <= face_box[2] and 
        face_box[1] <= laser_center_y <= face_box[3]):
        return True
    return False

def main(args):
    ckpt_path = args.checkpoint
    video_path = args.video_path
    
    model = YOLO(ckpt_path, task="detect")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Failed to load video")
        
        return

    scale_factor = 2

    while cap.isOpened():
        ret, frame = cap.read()
        
        
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            # processed_frame = apply_preprocessing(frame)
            # brightness = cv2.getTrackbarPos('Brightness', 'Controls') - 50
            # contrast = cv2.getTrackbarPos('Contrast', 'Controls') / 100.0

            # processed_frame = adjust_brightness_contrast(frame, brightness, contrast)

            # processed_frame = enhance_laser_visibility(frame)
            
            results = model(
                frame, 
                imgsz=args.image_size,
                conf=args.conf,
                iou=args.iou,
                augment=args.augment,
                agnostic_nms=False
                )
            
            # Get face and laser coordinates
            face_indices = (results[0].boxes.cls == 1)  # Class 1 is face
            laser_indices = (results[0].boxes.cls == 0)  # Class 0 is laser-point
            
            face_boxes = results[0].boxes.xyxy[face_indices]
            laser_boxes = results[0].boxes.xyxy[laser_indices]
            
            alert = "SAFE!"
            a_color = (0, 255, 0)
            # Check if laser points are in any face
            if len(laser_boxes) > 0 and len(face_boxes) > 0:
                for laser_box in laser_boxes:
                    for face_box in face_boxes:
                        if is_point_in_bbox(laser_box, face_box):
                            # Draw red rectangle around the face for alert
                            cv2.rectangle(
                                frame,
                                (int(face_box[0]), int(face_box[1])),
                                (int(face_box[2]), int(face_box[3])),
                                (0, 0, 255),  # Red color in BGR
                                3  # Thickness
                            )
                            alert = "WARNING!"
                            a_color = (0, 0, 255)
            # Add warning text
            cv2.putText(
                frame,
                alert,
                (60, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                a_color,
                3
            )
            
            vis_frame = results[0].plot()
            
            vis_frame = cv2.resize(vis_frame, (frame.shape[1] * scale_factor, frame.shape[0] * scale_factor))

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
