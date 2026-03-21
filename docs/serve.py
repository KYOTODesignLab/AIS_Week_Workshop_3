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
import sys
import threading
import time

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import requests
from flask import Flask, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from PIL import Image
import queue

# ── Import AR pipeline from 01_Marker_Recognition ────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01_Marker_Recognition"))
from construct import process_frame, model  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "00_Marker_Training", "models", "AIS26ws3_yolov8n_3.pt")
CACHE_DIR       = os.path.join(BASE_DIR, "cache")
CACHE_HTML_DIR  = os.path.join(BASE_DIR, "cache_html")
PORT            = 5000

IG_USER_ID             = os.environ.get("IG_USER_ID",             "17841445535736185")  # Instagram Business account numeric ID
IG_TOKEN               = os.environ.get("IG_TOKEN",               "EAANpAC8CLA0BQ0YdLllVUtq7yFirRSrexEKe6t3l6aj5XazcuIYizfNd0ZCflNUCNlrV5sLwShRs6Mn6AYcV4aZBrZC9hdqVyn86V2oN6GrnXdzCcsk5guUeptMD3eZCeRjIharDzHwRSJP1H3W9KdZBLy9InrK23rZC6wJcZC8ppy6X6vl6RneAJnC8YTRuLaQuD5x")  # long-lived access token
CLOUDINARY_CLOUD_NAME  = os.environ.get("CLOUDINARY_CLOUD_NAME",  "duiypvo1n")  # free at cloudinary.com
CLOUDINARY_UPLOAD_PRESET = os.environ.get("CLOUDINARY_UPLOAD_PRESET", "mikodigi")  # unsigned upload preset

MQTT_BROKER       = "broker.hivemq.com"
MQTT_PORT         = 1883
MQTT_TOPIC_IMG    = "ais-workshop/image"
MQTT_TOPIC_CHUNK  = "ais-workshop/chunk"
MQTT_TOPIC_RES    = "ais-workshop/result"
MQTT_TOPIC_ACK    = "ais-workshop/ack"
MQTT_TOPIC_GALLERY = "ais-workshop/gallery"

IG_API = "https://graph.facebook.com/v19.0"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CACHE_HTML_DIR, exist_ok=True)

# ── Gallery helpers (index.json + MQTT thumbnail) ────────────────────────────
_index_lock = threading.Lock()


def _update_cache_index():
    """Rebuild docs/cache_html/index.json from all .jpg files in CACHE_HTML_DIR (sorted asc)."""
    index_path = os.path.join(CACHE_HTML_DIR, "index.json")
    try:
        timestamps = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(CACHE_HTML_DIR)
            if f.endswith(".jpg")
        )
        with _index_lock:
            with open(index_path, "w") as fh:
                json.dump(timestamps, fh)
    except Exception as e:
        _log(f"index.json update error: {e}", "ERROR")


WEB_MAX_W   = 1200   # max width for gallery display images
WEB_QUALITY = 82     # JPEG quality for gallery display images


def _save_web_image(src_path: str, ts: int) -> str:
    """Downsample *src_path* and save to CACHE_HTML_DIR/<ts>.jpg. Returns dest path."""
    dest = os.path.join(CACHE_HTML_DIR, f"{ts}.jpg")
    try:
        img = cv2.imread(src_path)
        if img is None:
            return ""
        h, w = img.shape[:2]
        if w > WEB_MAX_W:
            scale = WEB_MAX_W / w
            img = cv2.resize(img, (WEB_MAX_W, int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dest, img, [cv2.IMWRITE_JPEG_QUALITY, WEB_QUALITY])
    except Exception as e:
        _log(f"_save_web_image error: {e}", "ERROR")
        return ""
    return dest


def _make_thumb_b64(img_path: str, max_w: int = 600, quality: int = 72) -> str:
    """Return base64-encoded JPEG thumbnail string, or empty string on error."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return ""
        h, w = img.shape[:2]
        if w > max_w:
            scale = max_w / w
            img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode() if ok else ""
    except Exception:
        return ""


# ── SSE gallery state ─────────────────────────────────────────────────────────
_sse_clients: list = []
_sse_lock    = threading.Lock()


def _sse_broadcast(data: dict):
    """Push a JSON event to all connected SSE gallery clients."""
    payload = f"data: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


# ── Logging helper ────────────────────────────────────────────────────────────

def _log(msg, tag="INFO"):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", flush=True)

# ── Handshake state ───────────────────────────────────────────────────────────────
ACK_TIMEOUT = 10  # seconds to wait for client ack before proceeding anyway
_pending_acks: dict[str, threading.Event] = {}

# ── Chunk reassembly state ────────────────────────────────────────────────────
_chunks: dict[str, dict] = {}
_chunks_lock = threading.Lock()

# ── MQTT ──────────────────────────────────────────────────────────────────────

def _on_connect(client, userdata, flags, rc):
    if rc == 0:
        _log(f"MQTT connected to {MQTT_BROKER}:{MQTT_PORT}", "MQTT")
    else:
        _log(f"MQTT connection failed (rc={rc})", "MQTT")
    client.subscribe(f"{MQTT_TOPIC_IMG}/+")
    client.subscribe(f"{MQTT_TOPIC_CHUNK}/+")
    client.subscribe(f"{MQTT_TOPIC_ACK}/+")
    _log(f"Subscribed to image / chunk / ack topics", "MQTT")


def _wait_ack(session_id):
    """Block until client sends ack or timeout expires."""
    evt = threading.Event()
    _pending_acks[session_id] = evt
    evt.wait(timeout=ACK_TIMEOUT)
    _pending_acks.pop(session_id, None)


def _dispatch(client, session_id, img_data, message, nickname, location, orientation, crop_mode="square", camera_specs=None):
    """Save assembled image bytes and launch the processing thread."""
    ts        = int(time.time() * 1000)
    img_path  = os.path.join(CACHE_DIR, f"{ts}.jpg")
    json_path = os.path.join(CACHE_DIR, f"{ts}.json")
    _log(f"[{session_id}] Image assembled: {len(img_data)//1024} KB  nickname={nickname!r}  loc={location}", "IMG")
    with open(img_path, "wb") as f:
        f.write(img_data)
    _log(f"[{session_id}] Saved → {img_path}", "IMG")
    client.publish(
        f"{MQTT_TOPIC_RES}/{session_id}",
        json.dumps({"status": "vision received"})
    )
    threading.Thread(
        target=_process_image,
        args=(client, session_id, ts, img_path, json_path,
              message, nickname, location, orientation, crop_mode, camera_specs),
        daemon=True
    ).start()


def _on_message(client, userdata, msg):
    # ─ Handle ack messages ──────────────────────────────────────────
    if msg.topic.startswith(f"{MQTT_TOPIC_ACK}/"):
        session_id = msg.topic.split("/")[-1]
        if session_id in _pending_acks:
            _pending_acks[session_id].set()
            _log(f"[{session_id}] ACK received", "ACK")
        return

    # ─ Handle chunked image ──────────────────────────────────────────
    if msg.topic.startswith(f"{MQTT_TOPIC_CHUNK}/"):
        try:
            p          = json.loads(msg.payload.decode())
            session_id = p["session_id"]
            idx        = int(p["idx"])
            total      = int(p["total"])
            _log(f"[{session_id}] Chunk {idx+1}/{total}  ({len(p['chunk'])//1024} KB)", "CHUNK")
            with _chunks_lock:
                if session_id not in _chunks:
                    _chunks[session_id] = {"meta": {}, "parts": {}, "total": total}
                entry = _chunks[session_id]
                entry["parts"][idx] = p["chunk"]
                if idx == 0:
                    entry["meta"] = {
                        "message":      p.get("message", ""),
                        "nickname":     p.get("nickname", "anonymous"),
                        "location":     p.get("location"),
                        "orientation":  p.get("orientation"),
                        "crop_mode":    p.get("crop_mode", "square"),
                        "camera_specs": p.get("camera_specs"),
                    }
                if len(entry["parts"]) == total:
                    b64      = "".join(entry["parts"][i] for i in range(total))
                    img_data = base64.b64decode(b64)
                    meta     = entry["meta"]
                    del _chunks[session_id]
                    _log(f"[{session_id}] All {total} chunks received — reassembling {len(img_data)//1024} KB", "CHUNK")
            if len(entry["parts"]) == total:   # check outside lock
                _dispatch(client, session_id, img_data,
                          meta["message"], meta["nickname"],
                          meta["location"], meta["orientation"],
                          meta.get("crop_mode", "square"),
                          meta.get("camera_specs"))
        except Exception as e:
            _log(f"Chunk error: {e}", "ERROR")
        return

    # ─ Handle legacy single-message image ────────────────────────────
    if msg.topic.startswith(f"{MQTT_TOPIC_IMG}/"):
        try:
            raw_kb = len(msg.payload) / 1024
            _log(f"Legacy single-message image on {msg.topic}: {raw_kb:.1f} KB", "IMG")
            p        = json.loads(msg.payload.decode())
            session_id  = p["session_id"]
            img_data    = base64.b64decode(p["image"])
            _dispatch(client, session_id, img_data,
                      p.get("message", ""), p.get("nickname", "anonymous"),
                      p.get("location"), p.get("orientation"),
                      p.get("crop_mode", "square"),
                      p.get("camera_specs"))
        except Exception as e:
            _log(f"MQTT message error: {e}", "ERROR")


def _process_image(client, session_id, ts, img_path, json_path,
                   message, nickname, location, orientation, crop_mode="square", camera_specs=None):
    def _pub(status):
        client.publish(
            f"{MQTT_TOPIC_RES}/{session_id}",
            json.dumps({"status": status})
        )

    try:
        # ─ Step 1: wait for client ack before running inference ─────────
        _log(f"[{session_id}] Waiting for client ACK before inference…", "PROC")
        _wait_ack(session_id)

        # Run full AR pipeline — produces annotated image with geometry overlay
        _log(f"[{session_id}] Running AR pipeline (process_frame)…", "PROC")
        t0        = time.time()
        frame      = cv2.imread(img_path)
        annotated  = process_frame(frame, camera_specs=camera_specs)
        cv2.imwrite(img_path, annotated)   # overwrite with AR-rendered version
        _log(f"[{session_id}] AR pipeline done in {time.time()-t0:.1f}s", "PROC")
        web_path = _save_web_image(img_path, ts)
        _log(f"[{session_id}] Web image saved → {web_path}", "PROC")

        # Extract YOLO detections from the raw frame for the JSON metadata
        _log(f"[{session_id}] Running YOLO detection…", "PROC")
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
        _log(f"[{session_id}] Detection complete: {len(detections)} object(s) — " +
             ", ".join(f"{d['label']} {d['confidence']:.0%}" for d in detections) or "none", "PROC")

        meta = {
            "timestamp":    ts,
            "session_id":   session_id,
            "message":      message,
            "nickname":     nickname,
            "location":     location,
            "orientation":  orientation,
            "camera_specs": camera_specs,
            "detections":   detections
        }
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        _sse_broadcast({"ts": str(ts), "meta": meta})
        _update_cache_index()
        thumb = _make_thumb_b64(web_path or img_path)
        client.publish(
            MQTT_TOPIC_GALLERY,
            json.dumps({"ts": str(ts), "meta": meta, "thumb": thumb})
        )
        _log(f"[{session_id}] Gallery update published (thumb {len(thumb)//1024} KB b64)", "PROC")

        # Build caption: full JSON, trimming excess detections if > 2000 chars
        IG_LIMIT = 2200
        TRIM_AT  = 2000

        def _make_caption(dets):
            payload = dict(meta)
            payload["detections"] = dets
            return json.dumps(payload, separators=(",", ":"))

        caption = _make_caption(detections)
        if len(caption) > TRIM_AT:
            # Binary-search for how many detections fit
            kept = detections[:]
            while kept and len(_make_caption(kept)) > TRIM_AT:
                kept.pop()
            trimmed = len(detections) - len(kept)
            caption = _make_caption(kept)[:-1]  # strip closing }
            caption += f',"excess_recognitions":{trimmed}}}'
            _log(f"[{session_id}] Caption trimmed: kept {len(kept)}, dropped {trimmed} detections", "PROC")
        caption = caption[:IG_LIMIT]   # hard safety cap
        _log(f"[{session_id}] {len(detections)} detections  caption={len(caption)} chars", "PROC")

        # ─ Step 2: publish detections, wait for ack before Instagram ───
        client.publish(
            f"{MQTT_TOPIC_RES}/{session_id}",
            json.dumps({"detections": detections})
        )

        if IG_USER_ID and IG_TOKEN and CLOUDINARY_CLOUD_NAME and CLOUDINARY_UPLOAD_PRESET:
            _log(f"[{session_id}] Waiting for client ACK before Instagram upload…", "IG")
            _wait_ack(session_id)
            _post_to_instagram(client, session_id, ts, caption, crop_mode)
        elif IG_USER_ID or IG_TOKEN:
            _log(f"[{session_id}] Instagram credentials incomplete — skipping auto-post", "WARN")

    except Exception as e:
        _log(f"[{session_id}] Processing error: {e}", "ERROR")


def _post_to_instagram(mqtt_client, session_id, ts, caption, crop_mode="square"):
    img_path = os.path.join(CACHE_DIR, f"{ts}.jpg")
    def _pub_status(msg):
        mqtt_client.publish(
            f"{MQTT_TOPIC_RES}/{session_id}",
            json.dumps({"status": msg})
        )
    try:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        _log(f"[{session_id}] Original image size: {w}×{h}  crop_mode={crop_mode}", "IG")
        if crop_mode == "portrait":
            # Center-crop to 3:4 portrait (1080×1440)
            crop_w = min(w, h * 3 // 4)
            crop_h = crop_w * 4 // 3
            _log(f"[{session_id}] Cropping to 3:4 portrait ({crop_w}×{crop_h})…", "IG")
        else:
            # Center-crop to 1:1 square (1080×1080)
            crop_w = crop_h = min(w, h)
            _log(f"[{session_id}] Cropping to 1:1 square ({crop_w}×{crop_h})…", "IG")
        left = (w - crop_w) // 2
        top  = (h - crop_h) // 2
        img  = img.crop((left, top, left + crop_w, top + crop_h))
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
            _log(f"[{session_id}] Cloudinary upload failed: {cloudinary_data}", "ERROR")
            _pub_status("public is inaccessible")
            return
        _log(f"[{session_id}] Cloudinary URL: {public_url}", "IG")
        create_res  = requests.post(
            f"{IG_API}/{IG_USER_ID}/media",
            params={"image_url": public_url, "caption": caption, "access_token": IG_TOKEN},
            timeout=15,
        )
        creation_id = create_res.json().get("id")
        if not creation_id:
            _log(f"[{session_id}] IG media creation failed: {create_res.json()}", "ERROR")
            _pub_status("public is inaccessible")
            return
        _log(f"[{session_id}] IG media container created: {creation_id}", "IG")

        # Poll until container status is FINISHED (Instagram needs time to process)
        for attempt in range(12):
            time.sleep(5)
            status_res = requests.get(
                f"{IG_API}/{creation_id}",
                params={"fields": "status_code", "access_token": IG_TOKEN},
                timeout=10,
            )
            status_code = status_res.json().get("status_code", "")
            _log(f"[{session_id}] IG container status: {status_code} (attempt {attempt+1}/12)", "IG")
            if status_code == "FINISHED":
                break
            if status_code == "ERROR":
                _log(f"[{session_id}] IG container error: {status_res.json()}", "ERROR")
                _pub_status("public is inaccessible")
                return
        else:
            _log(f"[{session_id}] IG container did not become ready in time", "ERROR")
            _pub_status("public is inaccessible")
            return

        pub_res = requests.post(
            f"{IG_API}/{IG_USER_ID}/media_publish",
            params={"creation_id": creation_id, "access_token": IG_TOKEN},
            timeout=15,
        )
        post_id = pub_res.json().get("id")
        if post_id:
            _log(f"[{session_id}] Instagram post published: {post_id}", "IG")
            _pub_status("multydimension vision is shared to public")
        else:
            _log(f"[{session_id}] IG publish failed: {pub_res.json()}", "ERROR")
            _pub_status("public is inaccessible")
    except Exception as e:
        _log(f"[{session_id}] Instagram auto-post error: {e}", "ERROR")
        _pub_status("public is inaccessible")


def _start_mqtt():
    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        c = mqtt.Client()   # paho-mqtt < 2.0
    c.on_connect    = _on_connect
    c.on_message    = _on_message
    c.on_disconnect = lambda cl, ud, rc: _log(f"MQTT disconnected (rc={rc})", "MQTT")
    _log(f"Connecting to MQTT broker {MQTT_BROKER}:{MQTT_PORT}…", "MQTT")
    c.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    c.loop_forever()


threading.Thread(target=_start_mqtt, daemon=True).start()

# ── Flask (Instagram API endpoint only) ───────────────────────────────────────

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/cache/<path:filename>")
def serve_cache(filename):
    """Serve cached images and JSON files."""
    return send_from_directory(CACHE_DIR, filename)


@app.get("/gallery")
def gallery_api():
    """Return all cached image+metadata pairs sorted ascending by timestamp."""
    items = []
    try:
        for fname in sorted(os.listdir(CACHE_DIR)):
            if not fname.endswith(".json"):
                continue
            ts_str    = fname[:-5]
            img_path  = os.path.join(CACHE_DIR, f"{ts_str}.jpg")
            json_path = os.path.join(CACHE_DIR, fname)
            if not os.path.exists(img_path):
                continue
            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
            items.append({"ts": ts_str, "meta": meta})
    except Exception as e:
        _log(f"Gallery list error: {e}", "ERROR")
    return jsonify(items)


@app.get("/gallery/events")
def gallery_sse():
    """Server-Sent Events stream — pushes {ts, meta} whenever a new image is processed."""
    q: queue.Queue = queue.Queue(maxsize=20)
    with _sse_lock:
        _sse_clients.append(q)

    def generate():
        try:
            yield 'data: {"status":"connected"}\n\n'
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _sse_lock:
                try:
                    _sse_clients.remove(q)
                except ValueError:
                    pass

    resp = Response(stream_with_context(generate()), content_type="text/event-stream")
    resp.headers["Cache-Control"]     = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


if __name__ == "__main__":
    print(f"MQTT  → {MQTT_BROKER}:{MQTT_PORT}  topics: {MQTT_TOPIC_IMG}/+")
    print(f"Flask → https://0.0.0.0:{PORT}  (Instagram API only)")
    print(f"Model: {MODEL_PATH}")
    print(f"Instagram: user={IG_USER_ID or '(not set)'}, token={'set' if IG_TOKEN else '(not set)'}")
    app.run(host="0.0.0.0", port=PORT, debug=False, ssl_context="adhoc")
