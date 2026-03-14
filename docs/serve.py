"""
Local inference server — MQTT edition (no tunnel required).

Image flow:  phone → MQTT public broker (HiveMQ) ← this server (paho-mqtt)
YOLO runs:   locally on PC, results published back via MQTT
Instagram:   server uploads cached image to Cloudinary → posts via Graph API

Usage:
    1. pip install flask flask-cors ultralytics paho-mqtt requests
    2. Set env vars (optional — only needed for Instagram API posting):
         IG_USER_ID             - Instagram Business/Creator account numeric ID
         IG_TOKEN               - long-lived Instagram Graph API access token
         CLOUDINARY_CLOUD_NAME  - Cloudinary cloud name (free at cloudinary.com)
         CLOUDINARY_UPLOAD_PRESET - unsigned upload preset name
    3. python docs/serve.py

MQTT topics:
    Subscribe: ais-workshop/image/<session_id>   (receives base64 JPEG + message)
    Publish:   ais-workshop/result/<session_id>  (sends detections JSON)

Flask endpoints (localhost only, for Instagram API):
    POST /post-instagram/<ts>    posts cached image to Instagram, returns post id
    GET  /health                 {"status": "ok"}
"""

import base64
import io
import json
import os
import threading
import time

import paho.mqtt.client as mqtt
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "00_Marker_Training", "models", "AIS26ws3_yolov8n_3.pt")
CACHE_DIR  = os.path.join(BASE_DIR, "cache")
PORT       = 5000

IG_USER_ID             = os.environ.get("IG_USER_ID",             "17841445535736185")  # Instagram Business account numeric ID
IG_TOKEN               = os.environ.get("IG_TOKEN",               "EAANpAC8CLA0BQ0YdLllVUtq7yFirRSrexEKe6t3l6aj5XazcuIYizfNd0ZCflNUCNlrV5sLwShRs6Mn6AYcV4aZBrZC9hdqVyn86V2oN6GrnXdzCcsk5guUeptMD3eZCeRjIharDzHwRSJP1H3W9KdZBLy9InrK23rZC6wJcZC8ppy6X6vl6RneAJnC8YTRuLaQuD5x")  # long-lived access token
CLOUDINARY_CLOUD_NAME  = os.environ.get("CLOUDINARY_CLOUD_NAME",  "duiypvo1n")  # free at cloudinary.com
CLOUDINARY_UPLOAD_PRESET = os.environ.get("CLOUDINARY_UPLOAD_PRESET", "mikodigi")  # unsigned upload preset

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
            json.dumps({"timestamp": ts, "detections": detections, "status": "vision received"})
        )
        print(f"Session {session_id}: {len(detections)} detections published")

        # Auto-post to Instagram if credentials are configured
        if IG_USER_ID and IG_TOKEN and CLOUDINARY_CLOUD_NAME and CLOUDINARY_UPLOAD_PRESET:
            _post_to_instagram(client, session_id, ts, message)
        elif IG_USER_ID or IG_TOKEN:
            print("Instagram credentials incomplete — skipping auto-post")

    except Exception as e:
        print(f"MQTT message error: {e}")


def _post_to_instagram(mqtt_client, session_id, ts, caption):
    img_path = os.path.join(CACHE_DIR, f"{ts}.jpg")
    def _pub_status(msg):
        mqtt_client.publish(
            f"{MQTT_TOPIC_RES}/{session_id}",
            json.dumps({"status": msg})
        )
    try:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        # Center-crop to 1:1 square (always valid aspect ratio for Instagram)
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top  = (h - side) // 2
        img  = img.crop((left, top, left + side, top + side))
        buf  = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        img_bytes = buf.getvalue()
        cloudinary_res = requests.post(
            f"https://api.cloudinary.com/v1_1/{CLOUDINARY_CLOUD_NAME}/image/upload",
            data={"upload_preset": CLOUDINARY_UPLOAD_PRESET},
            files={"file": ("capture.jpg", img_bytes, "image/jpeg")},
            timeout=30,
        )
        cloudinary_data = cloudinary_res.json()
        public_url = cloudinary_data.get("secure_url")
        if not public_url:
            print(f"Cloudinary upload failed: {cloudinary_data}")
            _pub_status("public is inaccessible")
            return
        print(f"Cloudinary URL: {public_url}")
        create_res  = requests.post(
            f"{IG_API}/{IG_USER_ID}/media",
            params={"image_url": public_url, "caption": caption, "access_token": IG_TOKEN},
            timeout=15,
        )
        creation_id = create_res.json().get("id")
        if not creation_id:
            print(f"IG media creation failed: {create_res.json()}")
            _pub_status("public is inaccessible")
            return
        pub_res = requests.post(
            f"{IG_API}/{IG_USER_ID}/media_publish",
            params={"creation_id": creation_id, "access_token": IG_TOKEN},
            timeout=15,
        )
        post_id = pub_res.json().get("id")
        if post_id:
            print(f"Instagram post published: {post_id}")
            _pub_status("multydimension vision is shared to public")
        else:
            print(f"IG publish failed: {pub_res.json()}")
            _pub_status("public is inaccessible")
    except Exception as e:
        print(f"Instagram auto-post error: {e}")
        _pub_status("public is inaccessible")


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
