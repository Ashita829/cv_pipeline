
import cv2

cap = cv2.VideoCapture("data/test.mp4")
print("Opened:", cap.isOpened())
print("Resolution:", int(cap.get(3)), "x", int(cap.get(4)))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
ret, frame = cap.read()
print("Frame shape:", frame.shape if ret else "could not read")
cap.release()
