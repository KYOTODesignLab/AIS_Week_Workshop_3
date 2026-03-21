"""
server_phone_inference.py
─────────────────────────
Phone-side inference mode.

The phone downloads the YOLO ONNX model, runs inference entirely on-device,
draws bounding boxes in the browser, and sends the already-annotated JPEG
frames to this server.  The PC only relays those frames to the display page.

Run:  python 02_Processing_Server/server_phone_inference.py
"""
import os
import socket
import base64
import threading
import webbrowser
import qrcode
from io import BytesIO
from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

_MR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_Marker_Recognition")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT = 5000

# ---------------------------------------------------------------------------
# Flask + SocketIO setup
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))
app.config["SECRET_KEY"] = os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------------------------------------------------------------------
# Model paths  (populated at startup in __main__)
# ---------------------------------------------------------------------------
_MODEL_PT   = os.path.join(_MR_DIR, "models", "sumizuke_yolov8n_3.pt")
_MODEL_ONNX = ""   # set in __main__
_CLASS_NAMES = []  # set in __main__

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
active_cameras: set = set()
cameras_lock = threading.Lock()
SERVER_URL = ""   # set at startup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def build_qr_b64(url: str) -> str:
    qr = qrcode.QRCode(box_size=8, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ensure_onnx_model(pt_path: str) -> str:
    """Export the .pt model to ONNX for browser-side inference if not already done."""
    onnx_path = pt_path.replace(".pt", ".onnx")
    if not os.path.exists(onnx_path):
        print("Exporting YOLO model to ONNX for browser inference…")
        _m = YOLO(pt_path)
        _m.export(format="onnx", imgsz=640, simplify=True, opset=12, half=False)
        print(f"ONNX model saved: {onnx_path}")
    return onnx_path


def ensure_ssl_cert(ip: str):
    """Generate a self-signed cert with SAN. Returns (cert_path, key_path) or (None, None)."""
    base = os.path.dirname(__file__)
    cert_file = os.path.join(base, "cert.pem")
    key_file  = os.path.join(base, "key.pem")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("SSL certificate already exists — reusing it.")
        return cert_file, key_file

    try:
        import ipaddress as _ipaddress
        import datetime as _datetime
        from cryptography import x509
        from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        print(f"Generating self-signed SSL certificate for {ip} (with SAN)...")
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, ip),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "YOLO Marker Server"),
        ])
        san = x509.SubjectAlternativeName([
            x509.IPAddress(_ipaddress.ip_address(ip)),
            x509.IPAddress(_ipaddress.ip_address("127.0.0.1")),
            x509.DNSName(ip),
            x509.DNSName("localhost"),
        ])
        now = _datetime.datetime.now(_datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + _datetime.timedelta(days=365))
            .add_extension(san, critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True, key_encipherment=True,
                    content_commitment=False, key_agreement=False,
                    key_cert_sign=False, crl_sign=False,
                    encipher_only=False, decipher_only=False, data_encipherment=False,
                ),
                critical=True,
            )
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ))
        print("SSL certificate generated.")
        return cert_file, key_file
    except ImportError:
        print("WARNING: 'cryptography' package not found — running over plain HTTP.")
        print("         Mobile Chrome may block camera access over HTTP.")
        print("         Install with: pip install cryptography")
        return None, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
        filename,
        mimetype="text/css",
    )


@app.route("/model")
def serve_model():
    """Serve the ONNX model file for browser-side inference."""
    model_dir = os.path.join(_MR_DIR, "models")
    onnx_name = os.path.basename(_MODEL_ONNX)
    resp = send_from_directory(model_dir, onnx_name, mimetype="application/octet-stream")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.route("/model/names")
def serve_model_names():
    return jsonify(_CLASS_NAMES)


@app.route("/")
def index():
    camera_url = f"{SERVER_URL}/camera"
    qr_data = build_qr_b64(camera_url)
    return render_template("index.html", qr_data=qr_data, camera_url=camera_url)


@app.route("/camera")
def camera_page():
    return render_template("camera.html", server_url=SERVER_URL)


# ---------------------------------------------------------------------------
# Socket.IO events
# ---------------------------------------------------------------------------

@socketio.on("connect")
def on_connect():
    with cameras_lock:
        active = len(active_cameras) > 0
    emit("stream_status", {"active": active})


@socketio.on("disconnect")
def on_disconnect():
    with cameras_lock:
        active_cameras.discard(request.sid)
        active = len(active_cameras) > 0
    if not active:
        socketio.emit("stream_status", {"active": False})


@socketio.on("frame")
def on_frame(data: str):
    """Relay an already-annotated JPEG frame (annotated on-device) to display clients."""
    with cameras_lock:
        active_cameras.add(request.sid)
    socketio.emit("processed_frame", data)
    socketio.emit("stream_status", {"active": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Preparing model for browser inference…")
    _tmp = YOLO(_MODEL_PT)
    _CLASS_NAMES = list(_tmp.names.values())
    del _tmp
    _MODEL_ONNX = ensure_onnx_model(_MODEL_PT)

    ip = get_local_ip()
    cert_file, key_file = ensure_ssl_cert(ip)
    use_ssl = cert_file is not None

    scheme = "https" if use_ssl else "http"
    SERVER_URL = f"{scheme}://{ip}:{PORT}"

    print(f"\n{'=' * 55}")
    print(f"  Mode:        Phone-side YOLO inference (ONNX)")
    print(f"  Server URL:  {SERVER_URL}")
    print(f"  Camera page: {SERVER_URL}/camera")
    if use_ssl:
        print()
        print("  NOTE: You must accept the SSL warning in your browser")
        print("  (self-signed certificate — click Advanced > Proceed)")
    print(f"{'=' * 55}\n")

    camera_url = f"{SERVER_URL}/camera"
    print(f"  Scan this QR code with your phone to open the camera page:")
    print(f"  {camera_url}\n")
    qr_terminal = qrcode.QRCode(border=1)
    qr_terminal.add_data(camera_url)
    qr_terminal.make(fit=True)
    qr_terminal.print_ascii(invert=True)
    print()

    _local_url = f"{'https' if use_ssl else 'http'}://127.0.0.1:{PORT}"
    threading.Timer(3.0, lambda: webbrowser.open(_local_url)).start()

    kwargs = dict(host="0.0.0.0", port=PORT, debug=False, use_reloader=False,
                  allow_unsafe_werkzeug=True)
    if use_ssl:
        kwargs["ssl_context"] = (cert_file, key_file)

    socketio.run(app, **kwargs)
