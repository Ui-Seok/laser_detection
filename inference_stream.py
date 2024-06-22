import cv2

from ultralytics import YOLO

# model path
ckpt_path = "runs/detect/train8/weights/best.pt"

# Load a model
model = YOLO(ckpt_path)

# Open the video file
video_path = ""
cap = cv2.VideoCapture(video_path)

# Streaming
while cap.isOpened():
    let, frame = cap.read()
    
    if let:
        results = model(frame)
        
        vis_frame = results[0].plot()
        
        cv2.imshow("Inference", vis_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    else:
        break
    
cap.release()
cv2.destroyAllWindows()