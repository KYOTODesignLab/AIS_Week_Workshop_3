from ultralytics import YOLO
import math
import os
import sys
import cv2
import numpy as np

from interpreter import MarkerConfig, MarkerDetector, BaseFrame, Scene, Renderer

# ── Load model ────────────────────────────────────────────────────────────────

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_Marker_Training",
                     "models", "AIS26ws3_yolov8n_3.pt")
# ── Marker configuration ──────────────────────────────────────────────────────
# Set origin_label / x_label / y_label / corner_label to match your YOLO class names.
# dist is the physical spacing between adjacent markers.

config = MarkerConfig(
    origin_label="zinzya",
    x_label="take",
    y_label="yama",
    corner_label="kamogawa",
    dist=1.0,
    geometry_labels=["sunlight", "river", "mountain"],
    modifier_labels=["many", "big"],
)

#__________________________ designate mesh file for 3D model __________________________
mesh_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elements", "models", "20260315_AIS_Roof_Lowpoly.obj")

# Shared size constant — used by both CubeSurface and Sudare height
CUBE_SIZE = 1.0

# Minimum confidence threshold — detections below this render 1 Sudare level
MIN_CONFIDENCE = 0.7

# ── Confidence → active Sudare levels mapping ─────────────────────────────────
# ≤ 0.6 confidence → 1 level; ≥ 0.9 → 6 levels; linear in between

def _conf_to_levels(conf: float) -> int:
    if conf <= MIN_CONFIDENCE:
        return 1
    if conf >= 0.9:
        return 6
    t = (conf - MIN_CONFIDENCE) / (0.9 - MIN_CONFIDENCE)
    return max(1, min(6, round(1 + t * 5)))


# Per-geometry-marker Sudare rotation (rot_z_deg rotates surface elements around Z)
_SUDARE_PARAMS = {
    "river":    {"rot_z_deg": 0.},
    "mountain": {"rot_z_deg": 90.},
    "sunlight": {"rot_z_deg": 45.},
}


model    = YOLO(_path)
detector = MarkerDetector(model, config)

# ── Scene definition (edit here to change what appears above the markers) ─────
# All 4 base markers (o, x, tri_o, tri_x) must be visible to define the frame.
# Use anchor= to position a shape relative to a specific marker.
# e.g. scene.add_box(size=0.5, anchor="o")  →  box sits above the origin marker

def build_scene() -> Scene:
    scene = Scene()
    # The OBJ was exported from a Y-up (Rhino) coordinate system, centred at
    # roughly (299.5, 2.1, -299.5).  We remap axes OBJ(x,y,z)→world(x,−z,y)
    # so the roof sits above the marker plane (world Z = up), then scale to fit
    # within the 1×1 unit marker frame and shift the centroid to (0.5, 0.5, 0).
    #   scale  ≈ 1 / mesh_span  = 1 / 2.571 ≈ 0.389
    #   offset = target_centre − scaled_centre
    _T = np.array([[1, 0,  0],   # OBJ x  → world x
                   [0, 0, -1],   # OBJ z (negated) → world y
                   [0, 1,  0]],  # OBJ y  → world z (up)
                  dtype=float)
    _s  = 1
    _ox = 0.5  - 299.5 * _s   # ≈ -116.0
    _oy = 0.5  - 299.5 * _s   # ≈ -116.0  (−OBJ_z centre)
    _oz = 1.3  - 1.5625 * _s  # ≈ -0.608  (floor of mesh at z = 0)
    scene.add_mesh(filepath=mesh_file, scale=_s, offset=(_ox, _oy, _oz), transform=_T)
    scene.add_cube_surface(cube_size=CUBE_SIZE, offset=(0.5, 0.5, 0.5))
    # scene.add_box(size=0.3)
    return scene

SCENE = build_scene()
SCENE_ALPHA = 0.45   # 0.0 = fully transparent  ·  1.0 = fully opaque

# ── Temporal smoothers ────────────────────────────────────────────────────────
# Both operate as exponential moving averages (EMA):
#   new_value = old_value + alpha * (raw_value - old_value)
# alpha=1.0 → no smoothing; lower values → more smoothing but slower response.

_DETECT_ALPHA = 0.45   # smoothing of 2-D detection centres between frames
_POSE_ALPHA   = 0.35   # smoothing of rvec / tvec between frames
_MAX_DET_AGE  = 8      # frames before a missing marker's smoothed position expires


class _DetectionSmoother:
    """EMA on per-label 2-D detection centres.  Only updates markers that YOLO
    actually detected this frame — does NOT inject phantom markers."""

    def __init__(self, alpha: float = _DETECT_ALPHA, max_age: int = _MAX_DET_AGE):
        self.alpha   = alpha
        self.max_age = max_age
        self._pts: dict  = {}   # label → smoothed (cx, cy)
        self._age: dict  = {}   # label → frames since last seen

    def update(self, markers: list) -> list:
        # Age every known label; evict stale ones
        for k in list(self._age):
            self._age[k] += 1
            if self._age[k] > self.max_age:
                self._pts.pop(k, None)
                del self._age[k]
        # Smooth each detected marker's image point
        for m in markers:
            cx, cy = m.image_point
            if m.name in self._pts:
                ex, ey = self._pts[m.name]
                cx = ex + self.alpha * (cx - ex)
                cy = ey + self.alpha * (cy - ey)
            self._pts[m.name] = (cx, cy)
            self._age[m.name] = 0
            m.image_point = (cx, cy)
        return markers


class _PoseSmoother:
    """EMA on rvec / tvec.  When the current frame is invalid (too few markers)
    the last valid pose is held so the overlay does not flash out."""

    def __init__(self, alpha: float = _POSE_ALPHA):
        self.alpha   = alpha
        self._rvec   = None
        self._tvec   = None

    def update(self, frame: BaseFrame) -> None:
        if not frame.valid:
            # Replay the last good pose so the renderer still has something
            if self._rvec is not None:
                frame.valid         = True
                frame.rvec          = self._rvec.copy()
                frame.tvec          = self._tvec.copy()
                frame._R_mat, _     = cv2.Rodrigues(frame.rvec)
                frame._tvec_ravel   = frame.tvec.ravel()
            return
        r = frame.rvec.ravel()
        t = frame.tvec.ravel()
        if self._rvec is None:
            self._rvec = r.copy()
            self._tvec = t.copy()
        else:
            self._rvec += self.alpha * (r - self._rvec)
            self._tvec += self.alpha * (t - self._tvec)
        frame.rvec         = self._rvec.reshape(3, 1)
        frame.tvec         = self._tvec.reshape(3, 1)
        frame._R_mat, _    = cv2.Rodrigues(frame.rvec)
        frame._tvec_ravel  = self._tvec.copy()


_det_smoother  = _DetectionSmoother()
_pose_smoother = _PoseSmoother()

# ── Device FOV lookup table ───────────────────────────────────────────────────
# Each entry: (ua_substr, phys_w, phys_h, zoom_max_ge, hfov_deg, label)
#   phys_w / phys_h  = screen CSS pixels × devicePixelRatio (rounded)
#   zoom_max_ge      = minimum zoomRange.max to match this row (None = any)
#   hfov_deg         = horizontal field of view of the main (1×) wide camera
#
# HFOV derived from 35mm-equiv focal length on a 4:3 phone sensor:
#   diagonal_FOV = 2·arctan(43.27 / (2·f_equiv))
#   HFOV_4:3     ≈ diagonal_FOV × (4/5) scaled by aspect geometry ≈ 73° for 26mm
#
# Rows are tested top-to-bottom; first match wins.
_DEVICE_DB = [
    # ── Apple iPhone (wide / main camera at 1× zoom) ───────────────────────
    # Pro models: 24mm-equiv sensor  → HFOV ≈ 78°
    ("iphone", 1206, 2622, 14, 78.0, "iPhone 16 Pro"),
    ("iphone", 1320, 2868, 14, 78.0, "iPhone 16 Pro Max"),
    ("iphone", 1179, 2556, 14, 78.0, "iPhone 14/15 Pro"),
    ("iphone", 1290, 2796, 14, 78.0, "iPhone 15 Pro Max"),
    # Standard models: 26mm-equiv    → HFOV ≈ 73°
    ("iphone", 1179, 2556, None, 73.0, "iPhone 14/15/16"),
    ("iphone", 1290, 2796, None, 73.0, "iPhone 15/16 Plus"),
    ("iphone", 1170, 2532, None, 73.0, "iPhone 12/13/14"),
    ("iphone",  828, 1792, None, 73.0, "iPhone XR/11"),
    ("iphone", 1125, 2436, None, 73.0, "iPhone X/XS/11 Pro"),
    ("iphone",  750, 1334, None, 73.0, "iPhone SE"),
    # ── Samsung Galaxy (approx.) ───────────────────────────────────────────
    ("samsung", 1080, 2340, None, 80.0, "Samsung S24/S25"),
    ("samsung", 1440, 3088, None, 85.0, "Samsung S24/S25 Ultra"),
    # ── Google Pixel ───────────────────────────────────────────────────────
    ("pixel",  1080, 2400, None, 79.0, "Pixel 9"),
    ("pixel",  1344, 2992, None, 82.0, "Pixel 9 Pro XL"),
]


def _estimate_fx(camera_specs: dict, img_w: int, img_h: int) -> float:
    """Return the best focal-length estimate (in pixels) for the given frame.

    Priority:
      1. Browser-reported focalLength (Android Chrome only) converted via sensor width.
      2. Device lookup table — matched by UA + physical screen size + zoom range.
      3. Fallback: fx = 0.7 × img_w  (HFOV ≈ 71°, better than the old fx = w).
    """
    if not camera_specs:
        return img_w * 0.7

    # 1. Direct focalLength from browser (rare, Android Chrome)
    fl = camera_specs.get("focalLength")
    if fl:
        # fl is in mm; convert using the horizontal sensor width inferred from
        # the 35mm-equiv (assuming 26mm-equiv → HFOV 73° as reference)
        hfov_rad = math.radians(73.0)
        sensor_w_mm = 2 * fl * math.tan(hfov_rad / 2)
        return (img_w / 2) / math.tan(hfov_rad / 2)   # still use HFOV path

    # 2. Device lookup
    ua  = (camera_specs.get("userAgent") or "").lower()
    sw  = camera_specs.get("screenWidth")  or 0
    sh  = camera_specs.get("screenHeight") or 0
    dpr = camera_specs.get("devicePixelRatio") or 1
    zm  = (camera_specs.get("zoomRange") or {}).get("max")
    pw  = round(sw * dpr)
    ph  = round(sh * dpr)

    for ua_sub, db_pw, db_ph, zm_ge, hfov_deg, label in _DEVICE_DB:
        if ua_sub not in ua:
            continue
        if abs(pw - db_pw) > 30 or abs(ph - db_ph) > 30:
            continue
        if zm_ge is not None and (zm is None or zm < zm_ge):
            continue
        hfov_rad = math.radians(hfov_deg)
        fx = (img_w / 2) / math.tan(hfov_rad / 2)
        print(f"[camera] matched {label}  HFOV={hfov_deg}°  fx={fx:.1f}px")
        return fx

    # 3. Generic fallback (HFOV ≈ 71°)
    return img_w * 0.7


# ── Core processing function ──────────────────────────────────────────────────

def process_frame(frame: np.ndarray, camera_specs: dict = None) -> np.ndarray:
    """Run the full 4-step AR pipeline on *frame* and return the annotated image.

    This function is the shared entry-point used by server_pc_inference.py.
    Pass *camera_specs* (the dict saved in the JSON cache) to enable improved
    focal-length estimation instead of the default fx = image_width fallback.
    """
    h, w = frame.shape[:2]
    fx = _estimate_fx(camera_specs, w, h)
    K = np.array([[fx, 0,  w / 2],
                  [0,  fx, h / 2],
                  [0,  0,  1    ]], dtype=np.float64)

    annotated = frame.copy()

    # Step 1 — detect and draw bounding boxes
    markers = detector.detect(frame)
    detector.draw_detections(annotated)

    # Smooth raw 2-D detection centres before handing them to solvePnP
    markers = _det_smoother.update(markers)

    # Step 2 — build the abstract coordinate frame from detected markers
    plane = BaseFrame(markers, K, config=config)

    # Smooth rvec / tvec across frames; hold last valid pose when markers drop out
    _pose_smoother.update(plane)

    # Step 3 — build per-frame scene: static shapes + dynamic Sudare
    scene = Scene()
    for shape in SCENE.shapes:        # reuse static mesh / cube_surface
        scene.shapes.append(shape)
    for m in markers:
        if m.role_type == "geometry" and m.name in _SUDARE_PARAMS:
            scene.add_sudare(
                width=CUBE_SIZE, depth=CUBE_SIZE, height=CUBE_SIZE,
                active_levels=_conf_to_levels(m.confidence),
                offset=(0., 0., CUBE_SIZE / 2.),   # sit above the marker plane
                anchor=m.name,
                **_SUDARE_PARAMS[m.name],
            )

    # Step 4 — project and render
    renderer = Renderer(plane)
    renderer.draw_marker_dots(annotated)
    renderer.draw_axes(annotated)
    renderer.draw_scene(annotated, scene, alpha=SCENE_ALPHA)
    renderer.draw_missing_warning(annotated)

    return annotated

# ── Helpers ───────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def _collect_images(path: str) -> list[str]:
    """Return a sorted list of image file paths from a file or directory."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        )
    print(f"Path not found: {path}")
    return []


def process_images(paths: list[str], save_dir: str | None = None) -> None:
    """Run process_frame on a list of image files and display / save results.

    Press any key to advance, 'q' to quit early.
    If *save_dir* is given the annotated images are written there.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            print(f"Could not read: {p}")
            continue

        result = process_frame(frame)
        title  = os.path.basename(p)
        cv2.imshow(title, result)
        print(f"[{title}]  press any key for next, 'q' to quit")

        if save_dir:
            out_path = os.path.join(save_dir, title)
            cv2.imwrite(out_path, result)
            print(f"  saved → {out_path}")

        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# ── Standalone entry point ────────────────────────────────────────────────────
# Usage:
#   python construct.py                      → webcam
#   python construct.py image.jpg            → single image
#   python construct.py images/              → all images in folder
#   python construct.py images/ --save out/  → process & save

if __name__ == "__main__":
    args = sys.argv[1:]

    # Parse optional --save <dir>
    save_dir = None
    if "--save" in args:
        idx = args.index("--save")
        if idx + 1 < len(args):
            save_dir = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        else:
            print("--save requires a directory argument")
            sys.exit(1)

    if args:
        # Image / folder mode
        images = []
        for a in args:
            images.extend(_collect_images(a))
        if not images:
            print("No images found.")
            sys.exit(1)
        print(f"Processing {len(images)} image(s)…")
        process_images(images, save_dir=save_dir)
    else:
        # Webcam mode (original behaviour)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: could not open video stream.")
            sys.exit(1)

        print("Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            cv2.imshow("YOLOv8 Stream", process_frame(frame))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
