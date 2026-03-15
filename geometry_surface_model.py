"""
geometry_surface_model.py
─────────────────────────
A cube whose 6 faces each carry a 5 × 5 array of nodes.
At every node two vertical surfaces are placed:

  • a circular disc    rotated +45° around the world Z axis
  • a square panel     rotated −45° around the world Z axis

The two surfaces are perpendicular to each other, forming a cross pattern.

Run the script to open a 3-D matplotlib viewer.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── Parameters ────────────────────────────────────────────────────────────────

CUBE_SIZE  = 1.0   # world units
GRID_N     = 5     # nodes per edge  (5 × 5 per face)
RADIUS     = 0.1   # circular-disc radius
HALF_SIDE  = 0.1   # square half-side (full panel) or half-side of each quadrant
N_SEGS     = 40    # polygon segments for the disc
DIVIDED    = True  # split square into 4 quadrant panels
GAP        = 0.01  # shift of each quadrant away from panel centre (world units)
ROTATE_Z_DEG = 45. # rotation of the whole surface model around world Z (for visual interest)

# ── Geometry helpers ──────────────────────────────────────────────────────────

def _Rz(deg):
    """3 × 3 rotation matrix around world Z."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0.],
                     [s,  c, 0.],
                     [0., 0., 1.]])


def _vertical_frame(rot_z_deg):
    """
    Return (hax, zax) – the two in-plane axes for a vertical surface
    whose plane normal is Rz(rot_z_deg) · X̂  (horizontal, world-Z rotation).
      hax : horizontal in-plane axis
      zax : world Z (vertical) in-plane axis
    """
    normal = _Rz(rot_z_deg) @ np.array([1., 0., 0.])
    zax    = np.array([0., 0., 1.])
    hax    = np.cross(zax, normal)
    hax   /= np.linalg.norm(hax)
    return hax, zax


def vertical_disc(center, rot_z_deg, radius, n_segs=40):
    """
    Polygon vertices for a vertical circular disc centred at *center*.
    Returns ndarray (n_segs, 3).
    """
    hax, zax = _vertical_frame(rot_z_deg)
    c = np.asarray(center, dtype=float)
    angles = np.linspace(0., 2. * math.pi, n_segs, endpoint=False)
    return c + radius * (np.outer(np.cos(angles), hax) +
                         np.outer(np.sin(angles), zax))


def vertical_square(center, rot_z_deg, half, divided=False, gap=0.0):
    """
    Return a list of vertical square polygon(s) centred at *center*.

    Division and offset are computed in the canonical frame (X = horizontal,
    Z = vertical), then the rotation rot_z_deg is applied to all vertices at
    the end.

    divided=False  →  list with one ndarray (4, 3) — the full square.
    divided=True   →  list with four ndarray (4, 3) — quadrant panels, each
                      half the size and offset outward by *gap* in X and Z.
    """
    R   = _Rz(rot_z_deg)
    c   = np.asarray(center, dtype=float)

    def _rotate_poly(local_pts):
        """Rotate local XZ-plane points and translate to world centre."""
        return (R @ local_pts.T).T + c

    if not divided:
        local = np.array([
            [-half, 0., -half], [ half, 0., -half],
            [ half, 0.,  half], [-half, 0.,  half],
        ])
        return [_rotate_poly(local)]

    # Build 4 quadrants in canonical (X, Z) space, then rotate.
    qhalf = half / 2.0
    quads = []
    for hs, vs in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:   # TL, TR, BR, BL
        cx = hs * (qhalf + gap)   # quadrant centre X
        cy = vs * gap             # quadrant centre Y (top +, bottom -)
        cz = vs * (qhalf + gap)   # quadrant centre Z
        local = np.array([
            [cx - qhalf, cy, cz - qhalf],
            [cx + qhalf, cy, cz - qhalf],
            [cx + qhalf, cy, cz + qhalf],
            [cx - qhalf, cy, cz + qhalf],
        ])
        quads.append(_rotate_poly(local))
    return quads


# ── Cube face nodes ───────────────────────────────────────────────────────────

def cube_face_nodes(size, n):
    """
    Generate (x, y, z) for every node in the 5 × 5 grid on all 6 cube faces.
    The cube is centred at the world origin.
    Edge / corner nodes appear on multiple faces (not deduplicated).
    """
    hs  = size / 2.
    lin = [-hs + size * i / (n - 1) for i in range(n)]

    for fixed in (-hs, hs):          # ±Z faces
        for u in lin:
            for v in lin:
                yield (u, v, fixed)

    for fixed in (-hs, hs):          # ±X faces
        for u in lin:
            for v in lin:
                yield (fixed, u, v)

    for fixed in (-hs, hs):          # ±Y faces
        for u in lin:
            for v in lin:
                yield (u, fixed, v)


# ── Build geometry ────────────────────────────────────────────────────────────

nodes   = list(cube_face_nodes(CUBE_SIZE, GRID_N))
discs   = [vertical_disc(p,    ROTATE_Z_DEG, RADIUS, N_SEGS) for p in nodes]
squares = [poly for p in nodes
           for poly in vertical_square(p, ROTATE_Z_DEG, HALF_SIDE,
                                       divided=DIVIDED, gap=GAP)]


# ── Visualise ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Layout: 3-D axes + slider panel below ─────────────────────────────────
    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_axes([0.05, 0.30, 0.90, 0.66], projection='3d')

    # Slider axes  [left, bottom, width, height]
    ax_radius   = fig.add_axes([0.15, 0.22, 0.70, 0.025])
    ax_halfside = fig.add_axes([0.15, 0.18, 0.70, 0.025])
    ax_gap      = fig.add_axes([0.15, 0.14, 0.70, 0.025])
    ax_rot      = fig.add_axes([0.15, 0.10, 0.70, 0.025])
    ax_divided  = fig.add_axes([0.15, 0.05, 0.15, 0.04])   # button

    sl_radius   = widgets.Slider(ax_radius,   'Disc radius',   0.01, 0.20, valinit=RADIUS,      valstep=0.005)
    sl_halfside = widgets.Slider(ax_halfside, 'Square half',   0.01, 0.20, valinit=HALF_SIDE,   valstep=0.005)
    sl_gap      = widgets.Slider(ax_gap,      'Gap',           0.00, 0.10, valinit=GAP,         valstep=0.002)
    sl_rot      = widgets.Slider(ax_rot,      'Rotation °',    0.,   90.,  valinit=ROTATE_Z_DEG, valstep=1.)
    btn_divided = widgets.Button(ax_divided,  f'Divided: {DIVIDED}')

    state = {'divided': DIVIDED}

    def _build_and_draw():
        ax.cla()
        r      = sl_radius.val
        half   = sl_halfside.val
        gap    = sl_gap.val
        rot    = sl_rot.val
        div    = state['divided']

        nodes_   = list(cube_face_nodes(CUBE_SIZE, GRID_N))
        discs_   = [vertical_disc(p, rot, r, N_SEGS) for p in nodes_]
        squares_ = [poly for p in nodes_
                    for poly in vertical_square(p, rot, half, divided=div, gap=gap)]

        ax.add_collection3d(Poly3DCollection(
            discs_,   alpha=0.35, linewidths=0.15,
            facecolor='royalblue', edgecolor='steelblue'))
        ax.add_collection3d(Poly3DCollection(
            squares_, alpha=0.40, linewidths=0.15,
            facecolor='darkorange', edgecolor='sienna'))

        # Cube wireframe
        hs = CUBE_SIZE / 2.
        v  = np.array([[-hs,-hs,-hs],[ hs,-hs,-hs],[ hs, hs,-hs],[-hs, hs,-hs],
                       [-hs,-hs, hs],[ hs,-hs, hs],[ hs, hs, hs],[-hs, hs, hs]])
        for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]:
            ax.plot(*zip(v[a], v[b]), 'k-', lw=0.9, alpha=0.35)

        xs, ys, zs = zip(*nodes_)
        ax.scatter(xs, ys, zs, c='black', s=3, alpha=0.4, depthshade=False)

        m = CUBE_SIZE * 0.78
        ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(
            f'Disc r={r:.3f}  |  Square half={half:.3f}  gap={gap:.3f}  '
            f'rot={rot:.0f}°  |  {"divided" if div else "solid"}',
            fontsize=9)
        ax.set_box_aspect([1, 1, 1])
        fig.canvas.draw_idle()

    def _on_slider(_):
        _build_and_draw()

    def _on_divided(_):
        state['divided'] = not state['divided']
        btn_divided.label.set_text(f'Divided: {state["divided"]}')
        _build_and_draw()

    sl_radius.on_changed(_on_slider)
    sl_halfside.on_changed(_on_slider)
    sl_gap.on_changed(_on_slider)
    sl_rot.on_changed(_on_slider)
    btn_divided.on_clicked(_on_divided)

    _build_and_draw()
    plt.show()
