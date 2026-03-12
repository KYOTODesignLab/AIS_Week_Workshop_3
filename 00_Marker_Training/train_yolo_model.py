import os
from ultralytics import YOLO

if __name__ == '__main__':
    data   = os.path.join(os.path.dirname(__file__), "data.yaml")
    default_model = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
    output = os.path.join(os.path.dirname(__file__), "models", "AIS26ws3_yolov8n_3.pt")

    # ── Choose starting weights ───────────────────────────────────
    # To start fresh from the base YOLOv8n backbone:
    #   START_FROM = "yolov8n.pt"
    # To continue training your previously saved model:
    #   START_FROM = output   (or any other .pt path)
    # NOTE: If nc (number of classes) changed, always start from the base model.
    START_FROM = default_model

    print(f"Starting from: {START_FROM}")
    model = YOLO(START_FROM)

    # Train on data.yaml ("cuda" or "cpu")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model.train(data=data, epochs=150, augment=True, imgsz=1224, batch=8, device=device)

    # Save back to the same file
    model.save(output)
    print(f"Model saved to: {output}")