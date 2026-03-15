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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from compas.geometry import Frame, Point, Vector

# ── Parameters ────────────────────────────────────────────────────────────────

CUBE_SIZE  = 1.0   # world units
GRID_N     = 5     # nodes per edge  (5 × 5 per face)
RADIUS     = 0.1  # circular-disc radius
HALF_SIDE  = 0.1  # square half-side
N_SEGS     = 40    # polygon segments for the disc


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


def vertical_square(center, rot_z_deg, half):
    """
    Four corner vertices for a vertical square panel centred at *center*.
    Returns ndarray (4, 3).
    """
    hax, zax = _vertical_frame(rot_z_deg)
    c = np.asarray(center, dtype=float)
    return np.array([
        c + h * hax + v * zax
        for h, v in [(-half, -half), (half, -half), (half, half), (-half, half)]
    ])


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
discs   = [vertical_disc(p,    45., RADIUS, N_SEGS) for p in nodes]
squares = [vertical_square(p, -45., HALF_SIDE)      for p in nodes]


# ── Visualise ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig = plt.figure(figsize=(11, 9))
    ax  = fig.add_subplot(111, projection='3d')

    # Circular discs — blue
    ax.add_collection3d(Poly3DCollection(
        discs,
        alpha=0.35, linewidths=0.15,
        facecolor='royalblue', edgecolor='steelblue'))

    # Square panels — orange
    ax.add_collection3d(Poly3DCollection(
        squares,
        alpha=0.40, linewidths=0.15,
        facecolor='darkorange', edgecolor='sienna'))

    # Cube wireframe
    hs = CUBE_SIZE / 2.
    v  = np.array([[-hs,-hs,-hs], [ hs,-hs,-hs], [ hs, hs,-hs], [-hs, hs,-hs],
                   [-hs,-hs, hs], [ hs,-hs, hs], [ hs, hs, hs], [-hs, hs, hs]])
    for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]:
        ax.plot(*zip(v[a], v[b]), 'k-', lw=0.9, alpha=0.35)

    # Node dots
    xs, ys, zs = zip(*nodes)
    ax.scatter(xs, ys, zs, c='black', s=3, alpha=0.4, depthshade=False)

    m = CUBE_SIZE * 0.78
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_zlim(-m, m)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(
        'Cube surface model  |  5×5 nodes per face\n'
        'Blue = vertical disc +45°   ·   Orange = vertical square −45°',
        fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()
