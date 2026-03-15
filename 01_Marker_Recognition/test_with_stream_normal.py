from ultralytics import YOLO
import os
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
    geometry_labels=["sunlight, river, mountain"],
    modifier_labels=["many, big"],
)

#__________________________ designate mesh file for 3D model __________________________
mesh_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elements", "models", "20260315_AIS_Roof_Lowpoly.obj")


model    = YOLO(_path)
detector = MarkerDetector(model, config)

# ── Scene definition (edit here to change what appears above the markers) ─────
# All 4 base markers (o, x, tri_o, tri_x) must be visible to define the frame.
# Use anchor= to position a shape relative to a specific marker.
# e.g. scene.add_box(size=0.5, anchor="o")  →  box sits above the origin marker

def build_scene() -> Scene:
    scene = Scene()
    scene.add_mesh(filepath=mesh_file, scale=1.0, offset=(0.5, 0.5, 0.))
    scene.add_cube_surface(cube_size=1.0, offset=(0.5, 0.5,0.0))
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

    # Step 3 — geometry lives in SCENE (defined above)

    # Step 4 — project and render
    renderer = Renderer(plane)
    renderer.draw_marker_dots(annotated)
    renderer.draw_axes(annotated)
    renderer.draw_scene(annotated, SCENE, alpha=SCENE_ALPHA)
    renderer.draw_missing_warning(annotated)

    return annotated

# ── Standalone webcam entry point ─────────────────────────────────────────────

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: could not open video stream.")
        exit()

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
