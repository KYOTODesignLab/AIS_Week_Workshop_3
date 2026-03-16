"""
server_instagram.py
───────────────────
Upload an image via browser → run 4-marker detection (construct)
→ post the annotated result to Instagram.

Setup
-----
1. Install extra dependencies:
       pip install instagrapi python-dotenv

2. Create a .env file next to this script (never commit it to git!):
       INSTAGRAM_USERNAME=your_instagram_handle
       INSTAGRAM_PASSWORD=your_password

3. Run:
       python server_instagram.py

4. Open  http://127.0.0.1:5001  in your browser.

Notes
-----
- .ig_session.json is written next to this script to cache the login session
  so repeated requests do not trigger a full re-login.  Keep that file private.
- Instagram requires image width 320–1440 px and aspect ratio 4:5 – 1.91:1.
  If your image is outside that range, Instagram will reject it.
"""
import os
import uuid
import base64
import socket
import time
import threading
import webbrowser

import cv2
import numpy as np
import requests as _requests
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file

from construct import process_frame

# ── Configuration ─────────────────────────────────────────────────────────────
PORT = 5001

# Load credentials from .env — try python-dotenv first, then manual parse
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.join(_HERE, ".env")

try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
except ImportError:
    # python-dotenv not installed — parse the file manually
    if os.path.exists(_ENV_FILE):
        with open(_ENV_FILE) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())

INSTAGRAM_ACCESS_TOKEN = os.environ.get("INSTAGRAM_ACCESS_TOKEN", "")
INSTAGRAM_USER_ID     = os.environ.get("INSTAGRAM_USER_ID", "")
PUBLIC_BASE_URL       = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = os.urandom(24).hex()
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

# ── Temporary image store (served to the Graph API during upload) ──────────────
_temp_images: dict = {}
_temp_images_lock = threading.Lock()


def _get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def post_to_instagram(jpeg_bytes: bytes, caption: str) -> str:
    """Post *jpeg_bytes* to Instagram via the official Graph API.
    Returns the permalink of the new post.

    How it works:
      1. Temporarily expose the image at /media/<token> on this server.
      2. Ask Instagram to create a media container pointing at that URL.
      3. Wait for Instagram to finish processing the container.
      4. Publish the container.
      5. Return the post permalink.
    """
    if not INSTAGRAM_ACCESS_TOKEN or not INSTAGRAM_USER_ID:
        raise RuntimeError(
            "Instagram credentials not configured. "
            "Add INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_USER_ID to a .env file."
        )

    token = uuid.uuid4().hex
    with _temp_images_lock:
        _temp_images[token] = jpeg_bytes

    try:
        base = PUBLIC_BASE_URL or f"http://{_get_local_ip()}:{PORT}"
        image_url = f"{base}/media/{token}"

        # 1. Create media container
        r = _requests.post(
            f"https://graph.instagram.com/v21.0/{INSTAGRAM_USER_ID}/media",
            data={"image_url": image_url, "caption": caption,
                  "access_token": INSTAGRAM_ACCESS_TOKEN},
            timeout=30,
        )
        r.raise_for_status()
        container_id = r.json()["id"]

        # 2. Poll until container status is FINISHED (up to ~20 s)
        for _ in range(10):
            time.sleep(2)
            s = _requests.get(
                f"https://graph.instagram.com/v21.0/{container_id}",
                params={"fields": "status_code",
                        "access_token": INSTAGRAM_ACCESS_TOKEN},
                timeout=15,
            )
            if s.json().get("status_code") == "FINISHED":
                break

        # 3. Publish
        p = _requests.post(
            f"https://graph.instagram.com/v21.0/{INSTAGRAM_USER_ID}/media_publish",
            data={"creation_id": container_id,
                  "access_token": INSTAGRAM_ACCESS_TOKEN},
            timeout=30,
        )
        p.raise_for_status()
        media_id = p.json()["id"]

        # 4. Fetch permalink
        info = _requests.get(
            f"https://graph.instagram.com/v21.0/{media_id}",
            params={"fields": "permalink",
                    "access_token": INSTAGRAM_ACCESS_TOKEN},
            timeout=15,
        )
        return info.json().get("permalink", "https://www.instagram.com/")

    finally:
        with _temp_images_lock:
            _temp_images.pop(token, None)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/media/<token>")
def serve_temp_image(token: str):
    """Serve a temporary annotated JPEG so the Graph API can fetch it."""
    with _temp_images_lock:
        data = _temp_images.get(token)
    if data is None:
        return "", 404
    return send_file(BytesIO(data), mimetype="image/jpeg")


@app.route("/")
def index():
    return render_template("instagram_upload.html")


@app.route("/process", methods=["POST"])
def process_and_post():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    caption = request.form.get("caption", "")

    # Decode uploaded image to OpenCV BGR array
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image. Upload a valid JPEG or PNG."}), 400

    # Run 4-marker detection and 3-D overlay
    annotated = process_frame(frame)

    # Encode annotated frame as JPEG bytes
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    jpeg_bytes = buf.tobytes()
    preview_b64 = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()

    try:
        ig_url = post_to_instagram(jpeg_bytes, caption)
        return jsonify({"preview": preview_b64, "ig_url": ig_url})

    except RuntimeError as e:
        return jsonify({"preview": preview_b64, "error": str(e)})

    except Exception as e:
        return jsonify({"preview": preview_b64, "error": f"Instagram error: {e}"})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  Marker → Instagram server  →  {url}\n")
    if not INSTAGRAM_ACCESS_TOKEN:
        print("  WARNING: INSTAGRAM_ACCESS_TOKEN not set. "
              "Create a .env file with your credentials.\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)
