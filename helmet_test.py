from ultralytics import YOLO
import cv2

# load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# load test image
img = cv2.imread("bike.jpg")

# run prediction
results = model(img, conf=0.4)

# show output
results[0].show()
