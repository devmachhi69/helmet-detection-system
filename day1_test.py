from ultralytics import YOLO
import cv2

# Load pre-trained YOLO model
model = YOLO("yolov8n.pt")

# Run detection on image
results = model("bike.jpg")

# Show detection result
for r in results:
    img = r.plot()
    cv2.imshow("YOLO Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
