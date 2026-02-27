import paho.mqtt.client as mqtt
import numpy as np
import time
import cv2 as cv
import os, sys
import threading

from compas.geometry import angle_vectors_signed, Frame

from interpreter import infer_yolo, interpret_results
import json

# Import the CCTDecodeRelease module
module_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..","../../crafts_extended"))
module_path = os.path.join(module_path, 'CCTDecode', 'CCTDecode')
if module_path not in sys.path:
    sys.path.append(module_path)

import CCTDecodeRelease as cct

marker_list = {"113": "011", "105": "012", "089": "013", "101": "014", "085": "015", "077": "016", "125": "017", "099": "018",
               "083": "019", "075": "01a", "123": "01b", "071": "01c", "119": "01d", "111": "01e", "095": "01f", "135": "021",
               "209": "022", "177": "023", "201": "024", "169": "025", "153": "026", "249": "027", "197": "028", "165": "029",
               "149": "02a", "245": "02b", "141": "02c", "237": "02d", "221": "02e", "189": "02f", "163": "031", "147": "032",
               "243": "033", "139": "034", "235": "035", "219": "036", "187": "037", "231": "039", "215": "03a", "183": "03b",
               "207": "03c", "175": "03d", "159": "03e", "255": "03f", "281": "044", "277": "045", "275": "046", "287": "047",
               "291": "048", "329": "049", "297": "04a", "489": "04b", "473": "04d", "441": "04e", "377": "04f", "293": "052",
               "485": "053", "469": "055", "437": "056", "373": "057", "461": "059", "429": "05a", "365": "05b", "413": "05c",
               "349": "05d", "317": "05e", "509": "05f", "399": "063", "467": "065", "435": "066", "371": "067", "459": "069",
               "427": "06a", "363": "06b", "411": "06c", "347": "06d", "315": "06e", "507": "06f", "423": "072", "359": "073",
               "407": "074", "343": "075", "311": "076", "503": "077", "335": "079", "303": "07a", "495": "07b", "479": "07d"}



MQTT_SERVER = "test.mosquitto.org"
IMAGE_TOPIC = "craftsx/img"
MODE_TOPIC = "craftsx/mode"
ID_TOPIC = "craftsx/id"
TF_TOPIC = "craftsx/tf"


# Global variable to store the mode
mode = 1  # Default mode
mode_lock = threading.Lock()

# Global variable to store tf data
tf_data = None
tf_lock = threading.Lock()

from ultralytics import YOLO 

path = "../models/sumizuke_yolov8n_1.pt"
model = YOLO(path)


def update_mode():
    """ Continuously listens for mode updates in a separate thread. """
    def on_mode_message(client, userdata, msg):
        global mode
        try:
            new_mode = int(msg.payload.decode())
            with mode_lock:
                mode = new_mode
        except ValueError:
            print("Invalid mode received")
    
    mode_client = mqtt.Client()
    mode_client.on_connect = lambda c, u, f, r: c.subscribe(MODE_TOPIC)
    mode_client.on_message = on_mode_message
    mode_client.connect(MQTT_SERVER, 1883, 60)
    mode_client.loop_forever()


def update_tf():
    """ Continuously listens for tf updates in a separate thread. """
    def on_tf_message(client, userdata, msg):
        global tf_data
        try:
            new_tf = msg.payload.decode()
            new_tf = json.loads(new_tf)
            with tf_lock:
                tf_data = new_tf
        except Exception as e:
            print("Error processing TF message:", e)
    
    tf_client = mqtt.Client()
    tf_client.on_connect = lambda c, u, f, r: c.subscribe(TF_TOPIC)
    tf_client.on_message = on_tf_message
    tf_client.connect(MQTT_SERVER, 1883, 60)
    tf_client.loop_forever()

# Callback when connected to MQTT broker
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(IMAGE_TOPIC)

# Callback when image message is received
def on_image_message(client, userdata, msg):

    nparr = np.frombuffer(msg.payload, np.uint8)
    img_np_gray = cv.imdecode(nparr, cv.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR in OpenCV 3.1
    img_np = cv.cvtColor(img_np_gray, cv.COLOR_GRAY2BGR)

    with mode_lock:
        current_mode = mode
    
    print("Processing image in mode", current_mode)
    
    if current_mode == 2:
        code_table, img = cct.CCT_extract(img_np, 12, 0.85, 'black')
        print("Code Table: ", code_table)
        if code_table:
            if str(code_table[0][0]) in marker_list:
                print('yey')
                image_id = marker_list[str(code_table[0][0])]
                client.publish(ID_TOPIC, image_id)
    else:
        with tf_lock:
            current_tf = tf_data
        
        print("Current TF:", current_tf)
        if current_tf:
            point = current_tf["point"]
            xaxis = current_tf["xaxis"]
            yaxis = current_tf["yaxis"]
        
            frame = Frame(point, xaxis, yaxis)

            print("Processing image...")
            # get results
            try:
                line, tilt, depth, diameter = interpret_results(img_np)
            except Exception as e:
                print("Error interpreting results:", e)
                line = None
                depth = None
                diameter = None
                tilt = None

            line = None if line is None else line
            depth = 0 if depth is None else depth
            diameter = 0 if diameter is None else diameter
            tilt = 0 if tilt is None else tilt

            if line is not None:
                print("Line found", line)
                axis = line.direction
                rotation = angle_vectors_signed(axis, xaxis, frame.zaxis, deg=True)
                rotation = round(rotation, 1)
                
                img_id = "d" + str(depth) + "r" + str(rotation) + "t" + str(tilt) + "di" + str(diameter)
                
                client.publish(ID_TOPIC, img_id)
                print("Image ID:", img_id)
            else:
                client.publish(ID_TOPIC, "no")
                print("No image ID found")
    cv.imshow("Nicla Vision", img_np)
    cv.waitKey(1)


client = mqtt.Client()
client.on_connect = on_connect
client.message_callback_add(IMAGE_TOPIC, on_image_message)
client.connect(MQTT_SERVER, 1883, 60)

# Start mode listener in a separate thread
mode_thread = threading.Thread(target=update_mode, daemon=True)
mode_thread.start()

# Start TF listener in a separate thread
tf_thread = threading.Thread(target=update_tf, daemon=True)
tf_thread.start()

client.loop_forever()
