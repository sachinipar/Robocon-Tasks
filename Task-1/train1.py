from ultralytics import YOLO

# Load the pretrained YOLOv8n classification model
model = YOLO("yolov8n-cls.pt")

# Train
model.train(
        data="C:/Users/lokha/OneDrive/Pictures/robocondataset",
        epochs=50,
        imgsz=224,
        batch=16
)

