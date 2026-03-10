from ultralytics import YOLO

# load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# start webcam
model.predict(source=0, show=True, conf=0.4)

