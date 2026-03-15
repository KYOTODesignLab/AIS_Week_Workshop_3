"""
geometry_surface_model.py
─────────────────────────
A cube whose 6 faces each carry a 5 × 5 array of nodes.
At every node two perpendicular vertical surfaces are placed:

  • VerticalDisc   – a circular disc
  • VerticalSquare – a square panel (optionally split into 4 quadrants)

Classes
-------
  VerticalDisc    – geometry for a single disc
  VerticalSquare  – geometry for a single square / quadrant set
  CubeSurface     – the full model (parameters + polygon generation)
  GeometryViewer  – interactive matplotlib viewer

Run the script to open an interactive 3-D viewer with sliders.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── Low-level math helper ─────────────────────────────────────────────────────

def _Rz(deg):
    """3 × 3 rotation matrix around world Z."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s, 0.],
                     [s,  c, 0.],
                     [0., 0., 1.]])


# ── Geometry classes ──────────────────────────────────────────────────────────

class VerticalDisc:
    """A vertical circular disc centred at a point in world space."""

    def __init__(self, center, rot_z_deg, radius, n_segs=40):
        self.center    = np.asarray(center, dtype=float)
        self.rot_z_deg = rot_z_deg
        self.radius    = radius
        self.n_segs    = n_segs

    def polygons(self):
        """Return [ndarray (n_segs, 3)] — the disc outline as a polygon."""
        normal = _Rz(self.rot_z_deg) @ np.array([1., 0., 0.])
        zax    = np.array([0., 0., 1.])
        hax    = np.cross(zax, normal)
        hax   /= np.linalg.norm(hax)
        angles = np.linspace(0., 2. * math.pi, self.n_segs, endpoint=False)
        pts = self.center + self.radius * (
            np.outer(np.cos(angles), hax) + np.outer(np.sin(angles), zax)
        )
        return [pts]


class VerticalSquare:
    """
    A vertical square panel centred at a point in world space.

    divided=False  →  polygons() returns [ndarray (4, 3)] — full square.
    divided=True   →  polygons() returns four ndarray (4, 3) — quadrants,
                      each offset outward by *gap* in X, Y and Z.
    """

    def __init__(self, center, rot_z_deg, half, divided=False, gap=0.0):
        self.center    = np.asarray(center, dtype=float)
        self.rot_z_deg = rot_z_deg
        self.half      = half
        self.divided   = divided
        self.gap       = gap

    def polygons(self):
        """Return a list of ndarray (4, 3), one per panel."""
        R = _Rz(self.rot_z_deg)
        c = self.center

        def _make(local_pts):
            return (R @ local_pts.T).T + c

        half = self.half
        if not self.divided:
            local = np.array([
                [-half, 0., -half], [ half, 0., -half],
                [ half, 0.,  half], [-half, 0.,  half],
            ])
            return [_make(local)]

        qhalf = half / 2.0
        gap   = self.gap
        quads = []
        for hs, vs in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:   # TL TR BR BL
            cx = hs * (qhalf + gap)
            cy = vs * gap
            cz = vs * (qhalf + gap)
            local = np.array([
                [cx - qhalf, cy, cz - qhalf],
                [cx + qhalf, cy, cz - qhalf],
                [cx + qhalf, cy, cz + qhalf],
                [cx - qhalf, cy, cz + qhalf],
            ])
            quads.append(_make(local))
        return quads


# ── Model ─────────────────────────────────────────────────────────────────────

class CubeSurface:
    """
    Parametric model of a cube with surface elements at every grid node.

    Parameters
    ----------
    cube_size : float   Side length of the cube.
    grid_n    : int     Nodes per edge on each face (grid_n × grid_n grid).
    rot_z_deg : float   Rotation of all surface elements around world Z.
    radius    : float   Disc radius.
    half_side : float   Square half-side (or quadrant half-side when divided).
    n_segs    : int     Polygon segments for disc approximation.
    divided   : bool    Split each square into 4 quadrant panels.
    gap       : float   Outward offset of each quadrant from the panel centre.
    """

    def __init__(self, cube_size=1.0, grid_n=5, rot_z_deg=45.,
                 radius=0.1, half_side=0.1, n_segs=40, divided=True, gap=0.01):
        self.cube_size = cube_size
        self.grid_n    = grid_n
        self.rot_z_deg = rot_z_deg
        self.radius    = radius
        self.half_side = half_side
        self.n_segs    = n_segs
        self.divided   = divided
        self.gap       = gap

    def nodes(self):
        """Generate (x, y, z) for every node on all 6 cube faces."""
        hs  = self.cube_size / 2.
        n   = self.grid_n
        lin = [-hs + self.cube_size * i / (n - 1) for i in range(n)]

        for fixed in (-hs, hs):      # ±Z faces
            for u in lin:
                for v in lin:
                    yield (u, v, fixed)

        for fixed in (-hs, hs):      # ±X faces
            for u in lin:
                for v in lin:
                    yield (fixed, u, v)

        for fixed in (-hs, hs):      # ±Y faces
            for u in lin:
                for v in lin:
                    yield (u, fixed, v)

    def make_disc(self, center):
        return VerticalDisc(center, self.rot_z_deg, self.radius, self.n_segs)

    def make_square(self, center):
        return VerticalSquare(center, self.rot_z_deg, self.half_side,
                              self.divided, self.gap)

    def disc_polygons(self):
        """All disc polygons as a flat list."""
        result = []
        for p in self.nodes():
            result.extend(self.make_disc(p).polygons())
        return result

    def square_polygons(self):
        """All square/quadrant polygons as a flat list."""
        result = []
        for p in self.nodes():
            result.extend(self.make_square(p).polygons())
        return result


# ── Viewer ────────────────────────────────────────────────────────────────────

class GeometryViewer:
    """Interactive matplotlib viewer for a CubeSurface model."""

    def __init__(self, model: CubeSurface):
        self.model = model
        self._build_figure()

    def _build_figure(self):
        m   = self.model
        fig = plt.figure(figsize=(12, 10))
        self.fig = fig
        self.ax  = fig.add_axes([0.05, 0.30, 0.90, 0.66], projection='3d')

        ax_radius   = fig.add_axes([0.15, 0.22, 0.70, 0.025])
        ax_halfside = fig.add_axes([0.15, 0.18, 0.70, 0.025])
        ax_gap      = fig.add_axes([0.15, 0.14, 0.70, 0.025])
        ax_rot      = fig.add_axes([0.15, 0.10, 0.70, 0.025])
        ax_divided  = fig.add_axes([0.15, 0.05, 0.15, 0.04])

        self.sl_radius   = widgets.Slider(ax_radius,   'Disc radius',  0.01, 0.20, valinit=m.radius,    valstep=0.005)
        self.sl_halfside = widgets.Slider(ax_halfside, 'Square half',  0.01, 0.20, valinit=m.half_side, valstep=0.005)
        self.sl_gap      = widgets.Slider(ax_gap,      'Gap',          0.00, 0.10, valinit=m.gap,       valstep=0.002)
        self.sl_rot      = widgets.Slider(ax_rot,      'Rotation °',   0.,   90.,  valinit=m.rot_z_deg, valstep=1.)
        self.btn_divided = widgets.Button(ax_divided,  f'Divided: {m.divided}')

        self.sl_radius.on_changed(self._on_slider)
        self.sl_halfside.on_changed(self._on_slider)
        self.sl_gap.on_changed(self._on_slider)
        self.sl_rot.on_changed(self._on_slider)
        self.btn_divided.on_clicked(self._on_divided)

    def _sync_model(self):
        m = self.model
        m.radius    = self.sl_radius.val
        m.half_side = self.sl_halfside.val
        m.gap       = self.sl_gap.val
        m.rot_z_deg = self.sl_rot.val

    def draw(self):
        self._sync_model()
        m  = self.model
        ax = self.ax
        ax.cla()

        ax.add_collection3d(Poly3DCollection(
            m.disc_polygons(),   alpha=0.35, linewidths=0.15,
            facecolor='royalblue', edgecolor='steelblue'))
        ax.add_collection3d(Poly3DCollection(
            m.square_polygons(), alpha=0.40, linewidths=0.15,
            facecolor='darkorange', edgecolor='sienna'))

        # Cube wireframe
        hs = m.cube_size / 2.
        v  = np.array([[-hs,-hs,-hs],[ hs,-hs,-hs],[ hs, hs,-hs],[-hs, hs,-hs],
                       [-hs,-hs, hs],[ hs,-hs, hs],[ hs, hs, hs],[-hs, hs, hs]])
        for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]:
            ax.plot(*zip(v[a], v[b]), 'k-', lw=0.9, alpha=0.35)

        node_pts = list(m.nodes())
        xs, ys, zs = zip(*node_pts)
        ax.scatter(xs, ys, zs, c='black', s=3, alpha=0.4, depthshade=False)

        lim = m.cube_size * 0.78
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(
            f'Disc r={m.radius:.3f}  |  Square half={m.half_side:.3f}  '
            f'gap={m.gap:.3f}  rot={m.rot_z_deg:.0f}°  |  '
            f'{"divided" if m.divided else "solid"}',
            fontsize=9)
        ax.set_box_aspect([1, 1, 1])
        self.fig.canvas.draw_idle()

    def _on_slider(self, _):
        self.draw()

    def _on_divided(self, _):
        self.model.divided = not self.model.divided
        self.btn_divided.label.set_text(f'Divided: {self.model.divided}')
        self.draw()

    def show(self):
        self.draw()
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model  = CubeSurface(
        cube_size=1.0, grid_n=5, rot_z_deg=45.,
        radius=0.1, half_side=0.1, n_segs=40, divided=True, gap=0.01,
    )
    viewer = GeometryViewer(model)
    viewer.show()
