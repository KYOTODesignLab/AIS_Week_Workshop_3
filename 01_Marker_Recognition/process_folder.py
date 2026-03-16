"""
process_folder.py
─────────────────
Batch-process every image inside the  'to_process'  folder using the full
AR pipeline defined in test_with_stream_normal.py and save the annotated
results to the  'processed'  folder at maximum quality.

Quality strategy
────────────────
  • PNG  →  saved as PNG  with compression level 0  (lossless, no artefacts)
  • JPEG / JPG  →  saved as JPEG with quality 100   (near-lossless, smallest
                   artefacts theoretically possible for JPEG)
  • All other formats  →  saved as PNG (lossless)

Run from inside  01_Marker_Recognition/:
    python process_folder.py

Or supply custom paths:
    python process_folder.py --input path/to/input --output path/to/output
"""

import argparse
import os
import sys

import cv2
import numpy as np

try:
    import pillow_heif
    from PIL import Image as _PILImage
    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:
    _HEIF_AVAILABLE = False

# Import shared state and pipeline pieces from the existing module -------------
from test_with_stream_normal import detector, config, SCENE, SCENE_ALPHA
from interpreter import BaseFrame, Renderer

# ── Enhancement stages ─────────────────────────────────────────────────────────
# Each tuple: (contrast_alpha, contrast_beta, unsharp_amount, unsharp_sigma)
# Stage 1 = no change (baseline), stage 5 = strongest boost.
_STAGES = [
    (1.0,   0,  0.0, 1.0),   # 1 – original (no enhancement)
    (1.60, 20,  1.5, 0.8),   # 2 – moderate
    (2.20, 40,  3.0, 0.8),   # 3 – strong
    (2.80, 60,  5.0, 0.7),   # 4 – very strong
    (3.50, 80,  8.0, 0.6),   # 5 – extreme
]


def _enhance(frame: np.ndarray, alpha: float, beta: int,
             unsharp: float, sigma: float) -> np.ndarray:
    """Return a contrast- and sharpness-enhanced copy of *frame*."""
    img = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    if unsharp > 0:
        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        img = cv2.addWeighted(img, 1.0 + unsharp, blurred, -unsharp, 0)
    return img


def _render_on_original(original: np.ndarray,
                        best_enhanced: np.ndarray) -> np.ndarray:
    """Detect on *best_enhanced*, render AR overlay onto *original*."""
    h, w = original.shape[:2]
    K = np.array([[w, 0, w / 2],
                  [0, w, h / 2],
                  [0, 0, 1   ]], dtype=np.float64)

    # Detection on the best-enhanced frame (sets detector internal state)
    markers = detector.detect(best_enhanced)

    # Rendering is applied to a clean copy of the ORIGINAL frame
    annotated = original.copy()
    detector.draw_detections(annotated)

    plane    = BaseFrame(markers, K, config=config)
    renderer = Renderer(plane)
    renderer.draw_marker_dots(annotated)
    renderer.draw_axes(annotated)
    renderer.draw_scene(annotated, SCENE, alpha=SCENE_ALPHA)
    renderer.draw_missing_warning(annotated)

    return annotated


def _best_of_stages(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Try all enhancement stages, return (best_enhanced, stage_index, n_markers)."""
    best_enhanced = frame
    best_count    = -1
    best_stage    = 0

    for i, (a, b, u, s) in enumerate(_STAGES):
        enhanced = _enhance(frame, a, b, u, s)
        try:
            markers = detector.detect(enhanced)
        except Exception:
            markers = []
        count = len(markers)
        if count > best_count:
            best_count    = count
            best_enhanced = enhanced
            best_stage    = i + 1   # 1-based for logging

    return best_enhanced, best_stage, best_count

# ── Constants ──────────────────────────────────────────────────────────────────

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN   = os.path.join(_SCRIPT_DIR, "to_process")
DEFAULT_OUT  = os.path.join(_SCRIPT_DIR, "processed")

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def read_image(path: str) -> np.ndarray | None:
    """
    Read an image file to a BGR uint8 ndarray.
    Falls back to pillow-heif for .heic / .heif files that OpenCV cannot load.
    """
    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame is not None:
        return frame

    ext = os.path.splitext(path)[1].lower()
    if ext in {".heic", ".heif"} and _HEIF_AVAILABLE:
        try:
            pil_img = _PILImage.open(path).convert("RGB")
            arr = np.array(pil_img, dtype=np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    return None


def collect_images(folder: str) -> list[str]:
    """Return a sorted list of supported image paths inside *folder*."""
    entries = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in _IMG_EXTS
    )
    return entries


def best_write_params(filename: str) -> tuple[str, list[int]]:
    """
    Return (output_extension, cv2_imwrite_params) optimised for quality.

    - JPEG/JPG  →  keep as JPEG at quality 100
    - PNG       →  keep as PNG  at compression 0  (fastest / no compression)
    - Everything else → write as PNG (safest lossless format)
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        return ext, [cv2.IMWRITE_JPEG_QUALITY, 100]
    if ext == ".png":
        return ext, [cv2.IMWRITE_PNG_COMPRESSION, 0]
    if ext in {".heic", ".heif"}:
        # OpenCV cannot write HEIC — save as JPEG at max quality instead
        return ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 100]
    # Fallback: write as PNG regardless of the original extension
    return ".png", [cv2.IMWRITE_PNG_COMPRESSION, 0]


# ── Main batch processor ───────────────────────────────────────────────────────

def process_folder(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    images = collect_images(input_dir)
    if not images:
        print(f"No supported images found in: {input_dir}")
        sys.exit(0)

    print(f"Found {len(images)} image(s) in '{input_dir}'")
    print(f"Output will be saved to '{output_dir}'\n")

    ok = 0
    failed = 0

    for img_path in images:
        filename = os.path.basename(img_path)

        # --- Read at full quality (IMREAD_COLOR keeps 8-bit BGR, no resize) ---
        frame = read_image(img_path)
        if frame is None:
            hint = "  (install pillow-heif for HEIC support)" if os.path.splitext(filename)[1].lower() in {".heic", ".heif"} and not _HEIF_AVAILABLE else ""
            print(f"  [SKIP]  Could not read: {filename}{hint}")
            failed += 1
            continue

        # --- Try 5 enhancement stages, pick the one with most markers ----------
        best_enhanced, best_stage, n_markers = _best_of_stages(frame)
        print(f"  [DETECT] {filename}  best stage={best_stage}/5  markers={n_markers}")

        # --- Render AR on the original image using best-stage detections -------
        try:
            result: np.ndarray = _render_on_original(frame, best_enhanced)
        except Exception as exc:
            print(f"  [ERROR] {filename}: {exc}")
            failed += 1
            continue

        # --- Write at maximum quality -----------------------------------------
        out_ext, params = best_write_params(filename)
        stem            = os.path.splitext(filename)[0]

        # Find the next free 2-digit number slot
        number = 1
        while True:
            out_filename = f"{stem}_{number:02d}{out_ext}"
            out_path     = os.path.join(output_dir, out_filename)
            if not os.path.exists(out_path):
                break
            number += 1

        success = cv2.imwrite(out_path, result, params)
        if success:
            h, w = result.shape[:2]
            print(f"  [OK]    {filename}  →  {out_filename}  ({w}×{h})")
            ok += 1
        else:
            print(f"  [FAIL]  Could not write: {out_path}")
            failed += 1

    print(f"\nDone — {ok} saved, {failed} failed.")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-process images through the AR pipeline."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_IN,
        help=f"Folder with source images  (default: {DEFAULT_IN})",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUT,
        help=f"Folder for annotated output  (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    process_folder(args.input, args.output)
