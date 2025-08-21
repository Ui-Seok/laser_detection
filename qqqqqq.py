# import cv2
# import time

# # print(cv2.getBuildInformation())

# url = "http://192.168.4.1/stream"

# # cap = cv2.VideoCapture(url)

# for i in range(5):
#     cap = cv2.VideoCapture(url)

#     if not cap.isOpened():
#         print(f"Retry {i}-th ...")
#         time.sleep(3)
#     else:
#         break

# while True:
#     # Read the frames
#     ret, frame = cap.read()

#     if not ret:
#         print("Can't read the frames")

#     # Show the frames
#     cv2.imshow('ESP32-CAM Stream', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resouce
# cap.release()
# cv2.destroyAllWindows()

###########################################################################
###########################################################################

import requests
import cv2
import numpy as np

from io import BytesIO
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

import time

def get_mjpeg_frames(url):
    try:
        r = requests.get(url, stream=True, timeout=10)
        bytes_data = bytes()

        print("Stream connected. Starting to receive frames ...")
        frame_received = 0
        start_time = time.time()

        for chunk in r.iter_content(chunk_size=4096):
            if not chunk:
                continue

            bytes_data += chunk

            while True:
                a = bytes_data.find(b'\xff\xd8')
                if a == -1:
                    break

                b = bytes_data.find(b'\xff\xd9', a)
                if b == -1:
                    break

                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]

                try:
                    img = Image.open(BytesIO(jpg))
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                    frame_received += 1
                    elapsed = time.time() - start_time
                    if elapsed >= 5:
                        fps = frame_received / elapsed
                        print(f"Now Frmae process: {fps:.1f} fps (Total {frame_received} frames)")
                        frame_received = 0
                        start_time = time.time()

                    yield frame
                except UnidentifiedImageError:
                    print(f"Skip damaged frames. Size: {len(jpg)} bytes.")
                except Exception as e:
                    print(f"Error: {str(e)}")

    except requests.exceptions.RequestException as e:
        print(f"Stream connection error: {str(e)}")
    except Exception as e:
        print(f"Uneffected error: {str(e)}")

if __name__ == "__main__":
    url = 'http://10.42.0.54/stream'
    print(f"Trying to connect stream: {url}")

    ckpt_path = "/home/ubuntu/laser_detection/models/yolov11_s_laser/laser_detector_s_laser2/weights/best.pt"

    model = YOLO(ckpt_path, task="detect")
    model_f = YOLO('yolov8n-face.pt', task="detect")

    frame_counter = 0
    process_evenry_n_frame = 4
    latest_boxes = []

    try:
        for frame in get_mjpeg_frames(url):
            frame_counter += 1

            if frame_counter % process_evenry_n_frame == 0:
                results = model(frame, iou=0.6, agnostic_nms=False, verbose=False)
                result_f = model_f(frame, iou=0.5, agnostic_nms=False, verbose=False)

                face_indices = (result_f[0].boxes.cls == 0)  # Class 1 is face
                laser_indices = (results[0].boxes.cls == 0)  # Class 0 is laser-point

                face_boxes = result_f[0].boxes.xyxy[face_indices]
                laser_boxes = results[0].boxes.xyxy[laser_indices]

                latest_boxes = (face_boxes, laser_boxes)

            if latest_boxes:
                face_boxes, laser_boxes = latest_boxes

                for face_box in face_boxes:
                    horizontal = face_box[2] - face_box[0]
                    vertical = face_box[3] - face_box[1]
                    cv2.rectangle(
                        frame,
                        (int(face_box[0] - horizontal/5), int(face_box[1] - vertical/5)),
                        (int(face_box[2] + horizontal/5), int(face_box[3] + vertical/100)),
                        (0, 0, 255),  # Red color in BGR
                        3  # Thickness
                    )

                for laser_box in laser_boxes:
                    cv2.rectangle(
                        frame,
                        (int(laser_box[0]), int(laser_box[1])),
                        (int(laser_box[2]), int(laser_box[3])),
                        (255, 0, 0),  # Red color in BGR
                        3  # Thickness
                    )
            # else:
            #     alert = "SAFE!"
            #     # GPIO.output(led_pin, GPIO.HIGH)
            #     a_color = (0, 255, 0)

            cv2.imshow('ESP32-CAM Stream', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting ...")
                break
    except  KeyboardInterrupt:
        print("Exiting ...")
    finally:
        cv2.destroyAllWindows()
        print("End of process")
