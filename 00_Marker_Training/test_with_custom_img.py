from ultralytics import YOLO

import os, random

# Set random seed
random.seed(50)

# Relative path to YOLOv5 model
path = os.path.join(os.path.dirname(__file__), "models", "AIS26ws3_yolov8n_3.pt")
model = YOLO(path)

# Test on custom images randomly selected from folder
folder = os.path.join(os.path.dirname(__file__), "LabelImg", "marker_images", "raw")



# Get 5 random images
images = [f for f in os.listdir(folder) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".JPG")]
images = random.sample(images, 3)

# Process each image
for img in images:
    img = os.path.join(folder, img)

    # Get results
    results = model(img)

    # Process results list
    for result in results:
        # Get result class
        print(f"Class: {result.names}")
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
