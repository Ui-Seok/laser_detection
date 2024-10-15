import cv2

cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()