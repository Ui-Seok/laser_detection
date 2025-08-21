import cv2
import time

# print(cv2.getBuildInformation())

url = "http://10.42.0.54/stream"

# cap = cv2.VideoCapture(url)

for i in range(5):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"Retry {i}-th ...")
        time.sleep(3)
    else:
        break

while True:
    # Read the frames
    ret, frame = cap.read()

    if not ret:
        print("Can't read the frames")

    # Show the frames
    cv2.imshow('ESP32-CAM Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resouce
cap.release()
cv2.destroyAllWindows()

###########################################################################
###########################################################################

# import requests
# import cv2
# import numpy as np

# from io import BytesIO
# from PIL import Image, UnidentifiedImageError

# import time

# def get_mjpeg_frames(url):
#     try:
#         r = requests.get(url, stream=True, timeout=10)
#         bytes_data = bytes()

#         print("Stream connected. Starting to receive frames ...")
#         frame_received = 0
#         start_time = time.time()

#         for chunk in r.iter_content(chunk_size=4096):
#             if not chunk:
#                 continue

#             bytes_data += chunk

#             while True:
#                 a = bytes_data.find(b'\xff\xd8')
#                 if a == -1:
#                     break

#                 b = bytes_data.find(b'\xff\xd9', a)
#                 if b == -1:
#                     break

#                 jpg = bytes_data[a:b+2]
#                 bytes_data = bytes_data[b+2:]

#                 try:
#                     img = Image.open(BytesIO(jpg))
#                     frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#                     frame_received += 1
#                     elapsed = time.time() - start_time
#                     if elapsed >= 5:
#                         fps = frame_received / elapsed
#                         print(f"Now Frmae process: {fps:.1f} fps (Total {frame_received} frames)")
#                         frame_received = 0
#                         start_time = time.time()

#                     yield frame
#                 except UnidentifiedImageError:
#                     print(f"Skip damaged frames. Size: {len(jpg)} bytes.")
#                 except Exception as e:
#                     print(f"Error: {str(e)}")

#     except requests.exceptions.RequestException as e:
#         print(f"Stream connection error: {str(e)}")
#     except Exception as e:
#         print(f"Uneffected error: {str(e)}")

# if __name__ == "__main__":
#     url = 'http://192.168.4.1/stream'
#     print(f"Trying to connect stream: {url}")

#     try:
#         for frame in get_mjpeg_frames(url):
#             cv2.imshow('ESP32-CAM Stream', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("Exiting ...")
#                 break
#     except  KeyboardInterrupt:
#         print("Exiting ...")
#     finally:
#         cv2.destroyAllWindows()
#         print("End of process")
