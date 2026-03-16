"""
server_pc_inference.py
──────────────────────
PC-side inference mode.

The phone captures raw video and sends JPEG frames to this server via
Socket.IO.  The PC decodes each frame, runs YOLO inference, draws bounding
boxes, and streams the annotated frame to the display page.
No model download is needed on the phone.

Run:  python server_pc_inference.py
"""
import os
import socket
import base64
import threading
import webbrowser
import qrcode
from io import BytesIO
import cv2
import numpy as np
from flask import Flask, render_template, render_template_string, request, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
from construct import process_frame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT = 5000

# ---------------------------------------------------------------------------
# Flask + SocketIO setup
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
app.config["SECRET_KEY"] = os.urandom(24).hex()
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

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


def open_local_browser(url: str):
    """Open the local page with a Windows fallback and always print the manual URL."""
    try:
        if webbrowser.open_new_tab(url):
            print(f"Opened browser at: {url}")
            return
    except Exception as exc:
        print(f"Browser auto-open failed via webbrowser: {exc}")

    if os.name == "nt":
        try:
            os.startfile(url)
            print(f"Opened browser at: {url}")
            return
        except OSError as exc:
            print(f"Browser auto-open failed via os.startfile: {exc}")

    print(f"Open this URL manually: {url}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), "templates"),
        filename,
        mimetype="text/css",
    )


@app.route("/")
def index():
    camera_url = f"{SERVER_URL}/camera"
    qr_data = build_qr_b64(camera_url)
    return render_template("index.html", qr_data=qr_data, camera_url=camera_url)


@app.route("/camera")
def camera_page():
    return render_template("camera_pc.html", server_url=SERVER_URL)


@app.route("/ping")
def ping():
        return jsonify({
                "ok": True,
                "server_url": SERVER_URL,
                "remote_addr": request.remote_addr,
                "user_agent": request.headers.get("User-Agent", ""),
        })


@app.route("/debug/client")
def debug_client():
        return render_template_string(
                """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Client Debug</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            padding: 20px;
            line-height: 1.5;
            background: #111;
            color: #eee;
        }
        h1 { margin-bottom: 12px; }
        code, pre {
            background: #1d1d1d;
            color: #9fe870;
            padding: 2px 6px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .row { margin: 10px 0; }
        .bad { color: #ff7b72; }
        .good { color: #7ee787; }
        a { color: #79c0ff; }
    </style>
</head>
<body>
    <h1>Client Debug</h1>
    <div class="row">Server URL: <code>{{ server_url }}</code></div>
    <div class="row">Path: <code>/debug/client</code></div>
    <div class="row">If this page loads, basic HTTPS/network access is working.</div>
    <div class="row">Next test: <a href="/camera">Open /camera</a></div>
    <div class="row" id="protocol"></div>
    <div class="row" id="secure"></div>
    <div class="row" id="media"></div>
    <div class="row" id="ua"></div>
    <script>
        const protocol = location.protocol;
        const secureContext = window.isSecureContext;
        const hasMediaDevices = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
        document.getElementById('protocol').innerHTML = 'Protocol: <code>' + protocol + '</code>';
        document.getElementById('secure').innerHTML = 'Secure context: <span class="' + (secureContext ? 'good' : 'bad') + '">' + secureContext + '</span>';
        document.getElementById('media').innerHTML = 'Camera API available: <span class="' + (hasMediaDevices ? 'good' : 'bad') + '">' + hasMediaDevices + '</span>';
        document.getElementById('ua').innerHTML = 'User agent:<br><pre>' + navigator.userAgent + '</pre>';
    </script>
</body>
</html>
""",
                server_url=SERVER_URL,
        )


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
    """Receive a raw JPEG frame from the phone, run marker interpretation on the PC,
    and broadcast the annotated frame to all display clients."""
    with cameras_lock:
        active_cameras.add(request.sid)

    # Decode base64 data URI → numpy array
    _, b64data = data.split(",", 1)
    jpg_bytes = base64.b64decode(b64data)
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return

    # Run marker interpretation (YOLO + 3-D overlay) via the shared module
    annotated = process_frame(frame)

    # Re-encode to JPEG data URI and broadcast
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    out_uri = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    socketio.emit("processed_frame", out_uri)
    socketio.emit("stream_status", {"active": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # The YOLO model is loaded automatically when construct is imported.

    ip = get_local_ip()
    cert_file, key_file = ensure_ssl_cert(ip)
    use_ssl = cert_file is not None

    scheme = "https" if use_ssl else "http"
    SERVER_URL = f"{scheme}://{ip}:{PORT}"
    local_url = f"{scheme}://127.0.0.1:{PORT}"

    print(f"\n{'=' * 55}")
    print(f"  Mode:        PC-side YOLO inference")
    print(f"  Server URL:  {SERVER_URL}")
    print(f"  Local URL:   {local_url}")
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

    threading.Timer(1.5, lambda: open_local_browser(local_url)).start()

    kwargs = dict(host="0.0.0.0", port=PORT, debug=False, use_reloader=False,
                  allow_unsafe_werkzeug=True)
    if use_ssl:
        kwargs["ssl_context"] = (cert_file, key_file)

    socketio.run(app, **kwargs)
