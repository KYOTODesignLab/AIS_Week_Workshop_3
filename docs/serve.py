"""
Local inference server — MQTT edition (no tunnel required).

Image flow:  phone → MQTT public broker (HiveMQ) ← this server (paho-mqtt)
YOLO runs:   locally on PC, results published back via MQTT
Instagram:   server uploads cached image to imgbb → posts via Graph API

Usage:
    1. pip install flask flask-cors ultralytics paho-mqtt requests
    2. Set env vars (optional — only needed for Instagram API posting):
         IG_USER_ID  - Instagram Business/Creator account numeric ID
         IG_TOKEN    - long-lived Instagram Graph API access token
         IMGBB_KEY   - free API key from imgbb.com
    3. python docs/serve.py

MQTT topics:
    Subscribe: ais-workshop/image/<session_id>   (receives base64 JPEG + message)
    Publish:   ais-workshop/result/<session_id>  (sends detections JSON)

Flask endpoints (localhost only, for Instagram API):
    POST /post-instagram/<ts>    posts cached image to Instagram, returns post id
    GET  /health                 {"status": "ok"}
"""

import base64
import json
import os
import threading
import time

import paho.mqtt.client as mqtt
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "00_Marker_Training", "models", "AIS26ws3_yolov8n_3.pt")
CACHE_DIR  = os.path.join(BASE_DIR, "cache")
PORT       = 5000

IG_USER_ID = os.environ.get("IG_USER_ID", "1711765319078442")    # Instagram account numeric ID
IG_TOKEN   = os.environ.get("IG_TOKEN",   "EAANpAC8CLA0BQ0YdLllVUtq7yFirRSrexEKe6t3l6aj5XazcuIYizfNd0ZCflNUCNlrV5sLwShRs6Mn6AYcV4aZBrZC9hdqVyn86V2oN6GrnXdzCcsk5guUeptMD3eZCeRjIharDzHwRSJP1H3W9KdZBLy9InrK23rZC6wJcZC8ppy6X6vl6RneAJnC8YTRuLaQuD5x")    # long-lived access token
IMGBB_KEY  = os.environ.get("IMGBB_KEY",  "4c23a2eebade65fb8803949474a9d8bb")    # free at imgbb.com (for IG API posting)

MQTT_BROKER    = "broker.hivemq.com"
MQTT_PORT      = 1883
MQTT_TOPIC_IMG = "ais-workshop/image"
MQTT_TOPIC_RES = "ais-workshop/result"

IG_API = "https://graph.facebook.com/v19.0"

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────

model = YOLO(MODEL_PATH)

# ── MQTT ──────────────────────────────────────────────────────────────────────

def _on_connect(client, userdata, flags, rc):
    print(f"MQTT connected (rc={rc})")
    client.subscribe(f"{MQTT_TOPIC_IMG}/+")


def _on_message(client, userdata, msg):
    try:
        payload    = json.loads(msg.payload.decode())
        session_id = payload["session_id"]
        img_data   = base64.b64decode(payload["image"])
        message    = payload.get("message", "")
        ts         = int(time.time() * 1000)

        img_path  = os.path.join(CACHE_DIR, f"{ts}.jpg")
        json_path = os.path.join(CACHE_DIR, f"{ts}.json")

        with open(img_path, "wb") as f:
            f.write(img_data)
        with open(json_path, "w") as f:
            json.dump({"timestamp": ts, "message": message, "session_id": session_id}, f, indent=2)

        results    = model(img_path)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                label      = result.names[cls_id]
                confidence = round(float(box.conf[0]), 3)
                x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
                detections.append({
                    "label": label, "confidence": confidence,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

        client.publish(
            f"{MQTT_TOPIC_RES}/{session_id}",
            json.dumps({"timestamp": ts, "detections": detections})
        )
        print(f"Session {session_id}: {len(detections)} detections published")

        # Auto-post to Instagram if credentials are configured
        if IG_USER_ID and IG_TOKEN and IMGBB_KEY:
            _post_to_instagram(ts, caption)
        elif IG_USER_ID or IG_TOKEN or IMGBB_KEY:
            print("Instagram credentials incomplete — skipping auto-post")

    except Exception as e:
        print(f"MQTT message error: {e}")


def _post_to_instagram(ts, caption):
    img_path = os.path.join(CACHE_DIR, f"{ts}.jpg")
    try:
        with open(img_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode()
        imgbb_res  = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": IMGBB_KEY, "image": b64_img, "expiration": 600},
            timeout=30,
        )
        imgbb_data = imgbb_res.json()
        if not imgbb_data.get("success"):
            print(f"imgbb upload failed: {imgbb_data}")
            return
        public_url  = imgbb_data["data"]["url"]
        create_res  = requests.post(
            f"{IG_API}/{IG_USER_ID}/media",
            params={"image_url": public_url, "caption": caption, "access_token": IG_TOKEN},
            timeout=15,
        )
        creation_id = create_res.json().get("id")
        if not creation_id:
            print(f"IG media creation failed: {create_res.json()}")
            return
        pub_res = requests.post(
            f"{IG_API}/{IG_USER_ID}/media_publish",
            params={"creation_id": creation_id, "access_token": IG_TOKEN},
            timeout=15,
        )
        post_id = pub_res.json().get("id")
        print(f"Instagram post published: {post_id}" if post_id else f"IG publish failed: {pub_res.json()}")
    except Exception as e:
        print(f"Instagram auto-post error: {e}")


def _start_mqtt():
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        c = mqtt.Client()   # paho-mqtt < 2.0
    c.on_connect = _on_connect
    c.on_message = _on_message
    c.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    c.loop_forever()


threading.Thread(target=_start_mqtt, daemon=True).start()

# ── Flask (Instagram API endpoint only) ───────────────────────────────────────

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print(f"MQTT  → {MQTT_BROKER}:{MQTT_PORT}  topics: {MQTT_TOPIC_IMG}/+")
    print(f"Flask → https://0.0.0.0:{PORT}  (Instagram API only)")
    print(f"Model: {MODEL_PATH}")
    print(f"Instagram: user={IG_USER_ID or '(not set)'}, token={'set' if IG_TOKEN else '(not set)'}")
    app.run(host="0.0.0.0", port=PORT, debug=False, ssl_context="adhoc")
