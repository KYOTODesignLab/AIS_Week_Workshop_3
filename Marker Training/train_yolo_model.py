import os
from ultralytics import YOLO

# data path and output path
data = os.path.join(os.path.dirname(__file__), "data.yaml")
output = os.path.join(os.path.dirname(__file__), "models", "sumizuke_yolov8n_3.pt")

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt") 

# Train on data.yaml
model.train(data=data, epochs=150, imgsz=640)

# Save the model to a file
model.save(output)