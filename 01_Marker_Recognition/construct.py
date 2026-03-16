from ultralytics import YOLO
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
    scene.add_mesh(filepath=mesh_file, scale=1.0, offset=(0.5, 0.5, 0.))
    scene.add_cube_surface(cube_size=CUBE_SIZE, offset=(0.5, 0.5, 0.0))
    # scene.add_box(size=0.3)
    return scene

SCENE = build_scene()
SCENE_ALPHA = 0.45   # 0.0 = fully transparent  ·  1.0 = fully opaque

# ── Core processing function ──────────────────────────────────────────────────

def process_frame(frame: np.ndarray) -> np.ndarray:
    """Run the full 4-step AR pipeline on *frame* and return the annotated image.

    This function is the shared entry-point used by server_pc_inference.py.
    """
    h, w = frame.shape[:2]
    K = np.array([[w, 0, w / 2],
                  [0, w, h / 2],
                  [0, 0, 1   ]], dtype=np.float64)

    annotated = frame.copy()

    # Step 1 — detect and draw bounding boxes
    markers = detector.detect(frame)
    detector.draw_detections(annotated)

    # Step 2 — build the abstract coordinate frame from detected markers
    plane = BaseFrame(markers, K, config=config)

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

    cap.release()
    cv2.destroyAllWindows()
