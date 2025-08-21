#!/usr/bin/env python3

import cv2
import argparse
import numpy as np
import asyncio
from bleak import BleakClient, BleakScanner

from ultralytics import YOLO

import requests

from io import BytesIO
from PIL import Image, UnidentifiedImageError

import time


class BLEController:
    def __init__(self, address, characteristic_uuid):
        self.address = address
        self.characteristic_uuid = characteristic_uuid
        self.client = None

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    async def _async_connect(self):
        print(f"Connecting to {self.address}...")
        self.client = BleakClient(self.address)
        await self.client.connect()
        print(f"Connected: {self.client.is_connected}")

    def connect(self):
        self.loop.run_until_complete(self._async_connect())

    async def _async_disconnect(self):
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            print("Disconnected")

    def disconnect(self):
        self.loop.run_until_complete(self._async_disconnect)

    async def _async_send_command(self, data):
        if self.client and self.client.is_connected:
            await self.client.write_gatt_char(self.characteristic_uuid, data)
    
    def send_command(self, data):
        self.loop.run_until_complete(self._async_send_command(data))


# -------------------- BLUETOOTH SETTING --------------------

XAIO_ADDRESS = "EA:BD:20:3A:DD:9A"

RELAY_SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
RELAY_CHARACTERISTIC_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"

# -------------------- BLUETOOTH SETTING --------------------


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="/home/ubuntu/laser_detection/models/yolov11_s_laser/laser_detector_s_laser2/weights/best.pt", help="model checkpoint path")
    parser.add_argument('--video-path', type=str, default="http://10.42.0.54/stream", help="test image path")
    parser.add_argument('--image-size', type=str, default=640, help="input image(frame) size")
    parser.add_argument('--conf', type=float, default=0.4, help="input image(frame) size")
    parser.add_argument('--iou', type=float, default=0.4, help="input image(frame) size")
    parser.add_argument('--augment', type=bool, default=False, help="input image(frame) size")
    parser.add_argument('--agnostic_nms', type=bool, default=False, help="input image(frame) size")
    
    args = parser.parse_args()
    return args

def get_mjpeg_frames(url):
    max_retries = 10
    for attempt in range(max_retries):
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
            if attempt < max_retries - 1:
                print(f"Retrying in 10 seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(10)
            else:
                print("Max retries reached. Exiting.")
                break
        except Exception as e:
            print(f"Uneffected error: {str(e)}")
            break

def enhance_laser_visibility(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:,:,2] = clahe.apply(hsv[:,:,2])

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def is_point_in_bbox(laser_box, face_box, horizontal, vertical):
    # Get center point of laser box
    laser_center_x = (laser_box[0] + laser_box[2]) / 2
    laser_center_y = (laser_box[1] + laser_box[3]) / 2
    
    # Check if laser center point is within face box
    if (face_box[0]- horizontal/5 <= laser_center_x <= face_box[2] + horizontal/5 and 
        face_box[1] - vertical/5 <= laser_center_y <= face_box[3] + vertical/100):
        return True
    # if (face_box[0] <= laser_center_x <= face_box[2] and 
    #     face_box[1] <= laser_center_y <= face_box[3]):
    #     return True
    return False

def main(args):
    ckpt_path = args.checkpoint
    video_path = args.video_path
    
    model = YOLO(ckpt_path, task="detect")
    model_f = YOLO('yolov8n-face.pt', task="detect")
    
    scale_factor = 1

    current_alert_state = "SAFE"
    alert = "SAFE!"
    a_color = (0, 255, 0)

    frame_counter = 0
    process_evenry_n_frame = 5
    latest_boxes = []

    ble_controller = BLEController(XAIO_ADDRESS, RELAY_CHARACTERISTIC_UUID)

    ble_controller.connect()

    try:
        for frame in get_mjpeg_frames(video_path):
            frame_counter += 1
            # frame = cv2.rotate(frame, cv2.ROTATE_180)   # CAMERA ROTATED
            
            # processed_frame = apply_preprocessing(frame)
            # brightness = cv2.getTrackbarPos('Brightness', 'Controls') - 50
            # contrast = cv2.getTrackbarPos('Contrast', 'Controls') / 100.0

            # processed_frame = adjust_brightness_contrast(frame, brightness, contrast)

            # processed_frame = enhance_laser_visibility(frame)

            new_alert_state = "SAFE"
            
            if frame_counter % process_evenry_n_frame == 0:
                results = model(
                    frame, 
                    imgsz=args.image_size,
                    conf=args.conf,
                    iou=args.iou,
                    augment=args.augment,
                    agnostic_nms=False,
                    verbose=False
                    )

                results_f = model_f(
                    frame, 
                    imgsz=args.image_size,
                    conf=args.conf,
                    iou=0.5,
                    augment=args.augment,
                    agnostic_nms=False,
                    verbose=False
                    )
                
                # Get face and laser coordinates
                face_indices = (results_f[0].boxes.cls == 0)  # Class 1 is face
                laser_indices = (results[0].boxes.cls == 0)  # Class 0 is laser-point
                
                face_boxes = results_f[0].boxes.xyxy[face_indices]
                laser_boxes = results[0].boxes.xyxy[laser_indices]

                latest_boxes = (face_boxes, laser_boxes)
            
            # alert = "SAFE!"
            # GPIO.output(led_pin, GPIO.HIGH)
            # a_color = (0, 255, 0)

            # Check if laser points are in any face
            if latest_boxes:
                if len(face_boxes) > 0:
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

                if len(laser_boxes) > 0:
                    for laser_box in laser_boxes:
                        cv2.rectangle(
                            frame,
                            (int(laser_box[0]), int(laser_box[1])),
                            (int(laser_box[2]), int(laser_box[3])),
                            (255, 0, 0),  # Red color in BGR
                            3  # Thickness
                        )
                else:
                    new_alert_state = "SAFE"
                    a_color = (0, 255, 0)
                
                # cv2.imshow("Test", frame)

                if len(laser_boxes) > 0 and len(face_boxes) > 0:
                    found = False

                    for laser_box in laser_boxes:
                        for face_box in face_boxes:
                            horizontal = face_box[2] - face_box[0]
                            vertical = face_box[3] - face_box[1]
                            if is_point_in_bbox(laser_box, face_box, horizontal, vertical):
                                # Draw red rectangle around the face for alert
                                new_alert_state = "WARNING"
                                a_color = (0, 0, 255)
                                found = True
                                break
                        if 'found' in locals() and found:
                            break
                        else:
                            new_alert_state = "SAFE"
                            a_color = (0, 255, 0)
                            break

            if new_alert_state != current_alert_state:
                current_alert_state = new_alert_state
                if current_alert_state == "WARNING":
                    alert = "WARNING!"
                    a_color = (0, 0, 255)
                    ble_controller.send_command(b'\x01')
                    print("STATE CHANGE: WARNING! -> Sending BLE ON")
                else:
                    alert = "SAFE!"
                    a_color = (0, 255, 0)
                    ble_controller.send_command(b'\x00')
                    print("STATE CHANGE: SAFE! -> Sending BLE OFF")

            # Add warning text
            # print(len(face_boxes), len(laser_boxes), alert)
            
            # vis_frame = results[0].plot()
            
            # vis_frame = cv2.resize(frame, (frame.shape[1] * scale_factor, frame.shape[0] * scale_factor))

            # frame_rotated = cv2.rotate(frame, cv2.ROTATE_180)            
            cv2.putText(frame, alert, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, a_color, 3)

            frame = cv2.resize(frame, (frame.shape[1] * scale_factor, frame.shape[0] * scale_factor))
            cv2.imshow("Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting")
                break

    except KeyboardInterrupt:
        print("Exiting ...")
    finally:
        ble_controller.disconnect()
        cv2.destroyAllWindows()
        print("End of process")

if __name__ == "__main__":
    args = parse_opt()

    main(args)
