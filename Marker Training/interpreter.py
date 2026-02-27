import paho.mqtt.client as mqtt
import numpy as np
import cv2
from compas.geometry import Line, closest_point_on_line, distance_point_point

# MQTT Broker Configuration
BROKER = "test.mosquitto.org"  # Public MQTT broker
TOPIC = "craftsx/img"  # Same topic as in Nicla Vision script

from ultralytics import YOLO

path = "../models/sumizuke_yolov8n_3.pt"
model = YOLO(path)

all_classes = {
    "0": "target",
    "1": "angle",
    "2": "negative",
    "3": "end",
    "4": "diameter",
    "5": "depth",
}


def preprocess_image(img_bytes):
    """Convert received image bytes to a NumPy array and resize for FOMO."""
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (320, 320))  # Resize to 256x256
    return img_resized


def infer_yolo(img):
    """Perform inference on the image using the YOLO model."""
    results = model(img)
    boxess = []
    classes = []
    for result in results:
        boxes = result.boxes
        boxess.append(boxes.xywh)
        # Get result class for each box
        class_indices = result.boxes.cls.cpu().numpy().astype(int)  # Class indices
        # get confidence scores
        scores = result.boxes.conf.cpu().numpy()
        classes.append(class_indices)

    return boxess, classes, scores


def draw_text(img, text, x, y):
    """Draw text on the image."""
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )


def draw_rectangle(img, x1, y1, x2, y2, color):
    """Draw rectangle on the image."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def draw_base_line(img, target, end, color):
    """Draw line on the image."""
    # cv2.line(img, target, end, color, 2)
    line = Line([target[0], target[1], 0], [end[0], end[1], 0])
    return line

def find_angle_value(line, angle, negative):
    """Find angle between two lines."""
    length = line.length
    if negative is not None:
        angle = negative
        sign = -1
    else:
        sign    = 1 
    point = closest_point_on_line(point = [angle[0], angle[1], 0], line = line)
    d = distance_point_point(line.start, point)
    slope = d/length * sign

    # Round to one decimal place
    slope = round(slope, 1)
    return slope

def find_depth_value(line, depth):
    """Find depth between two lines."""
    length = line.length
    point = closest_point_on_line(point = [depth[0], depth[1], 0], line = line)
    d = distance_point_point(line.start, point)
    slope = d/length
    # Round to one decimal place
    slope = round(slope, 1)
    return slope

def find_diameter_value(diameter):
    """Find diameter value."""
    map = {0: 20, 1:24, 2:28, 3:32}
    return map[diameter]

def interpret_results(img_array):
    boxes, classes, _ = infer_yolo(img_array)
    print("Boxes:", boxes)
    print("Classes:", classes)
    # Draw boxes on the image
    for box, clas in zip(boxes, classes):
        # Box is a tensor of [x, y, w, h]
        target, angle, negative, end, diameter, depth = (
            None,
            None,
            None,
            None,
            0,
            None,
        )
        for b, c in zip(box, clas):
            x, y, w, h = b
            x1, y1, x2, y2 = (
                int(x - w / 2),
                int(y - h / 2),
                int(x + w / 2),
                int(y + h / 2),
            )

            if c == 0:
                # make red
                draw_rectangle(img_array, x1, y1, x2, y2, (0, 0, 255))
                # Display the class name
                draw_text(img_array, all_classes[str(c)], x1, y1 - 10)
                target =[x,y]
            elif c == 1:
                # make green
                draw_rectangle(img_array, x1, y1, x2, y2, (0, 255, 0))
                # Display the class name
                draw_text(img_array, all_classes[str(c)], x1, y1 - 10)
                angle = [x,y]
            elif c == 2:
                # make blue
                draw_rectangle(img_array, x1, y1, x2, y2, (255, 0, 0))
                # Display the class name
                draw_text(img_array, all_classes[str(c)], x1, y1 - 10)
                negative = [x,y]
            elif c == 3:
                # make yellow
                draw_rectangle(img_array, x1, y1, x2, y2, (0, 255, 255))
                # Display the class name
                draw_text(img_array, all_classes[str(c)], x1, y1 - 10)
                end = [x,y]
            elif c == 4:
                # make pink
                draw_rectangle(img_array, x1, y1, x2, y2, (255, 0, 255))
                # Display the class name
                draw_text(img_array, all_classes[str(c)], x1, y1 - 10)
                diameter += 1
            elif c == 5:
                # make orange
                draw_rectangle(img_array, x1, y1, x2, y2, (0, 165, 255))
                # Display the class name
                draw_text(img_array, all_classes[str(c)], x1, y1 - 10)
                depth = [x,y]
            else:
                print("Unknown")
        
        if target and end:
            try:
                line = draw_base_line(img_array, target, end, (255, 255, 255))
                if angle or negative:
                    slope_ratio = find_angle_value(line, angle, negative)
                    draw_text(img_array, str(slope_ratio), 10,10)
                else:
                    slope_ratio = None
                if depth:
                    depth_ratio = find_depth_value(line, depth)
                    draw_text(img_array, str(depth_ratio), 10,30)
                else:
                    depth_ratio = None
                
                if diameter:
                    diam_value = find_diameter_value(diameter)
                    draw_text(img_array, str(diam_value), 10, 50)
                else:
                    diam_value = None
            except Exception as e:
                print("Error:", e)
        else:
            print("Target and end points not found")
            line, slope_ratio, depth_ratio, diam_value = None, None, None, None
        print("Line:", line)
        print("Slope Ratio:", slope_ratio)
        print("Depth Ratio:", depth_ratio)
        print("Diameter Value:", diam_value)
    return line, slope_ratio, depth_ratio, diam_value


def on_message(client, userdata, message):
    """Callback function for MQTT message reception."""
    try:
        img_bytes = message.payload
        img_array = preprocess_image(img_bytes)
        
        # Interpret the results
        interpret_results(img_array)  

        # Display the image
        cv2.imshow("Nicla Vision", img_array)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            client.disconnect()
            cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    # Set up MQTT client
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(BROKER)
    client.subscribe(TOPIC)

    print("Listening for images from Nicla Vision...")
    client.loop_forever()

