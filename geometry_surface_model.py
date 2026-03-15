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
  CubeSurface     – cube model: 6-face grid of surface elements
  Sudare          – screen model: 2 rows × 6 levels on ±X or ±Y sides of a frame
  GeometryViewer  – interactive matplotlib viewer (toggle between models)

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


# ── Sudare ────────────────────────────────────────────────────────────────────

class Sudare:
    """
    A screen (sudare/簾) of surface elements placed on the ±X or ±Y sides
    of a rectangular frame.

    Nodes are arranged in *rows* × *levels* (horizontal × vertical) on each
    of the two facing sides.  Defaults: 1 row per side × 2 sides = 2 rows total,
    6 levels per side.

    Parameters
    ----------
    width       : float   Frame dimension along X.
    depth       : float   Frame dimension along Y.
    height      : float   Frame height (Z extent). Frame is centred at world origin.
    direction   : str     'x' — elements on the ±X faces;
                          'y' — elements on the ±Y faces.
    rows        : int     Horizontal node count *per face* (default 1 → 2 rows total).
    levels      : int     Vertical   node count per face (default 6).
    row_spacing   : float   Distance between the two rows (default: width for 'x',
                            depth for 'y').  Decouples row placement from frame size.
    active_levels : int     How many levels (counting from the top) to generate.
                            active_levels==levels → all; 1 → top level only.
                            Defaults to *levels* (all).
    rot_z_deg     : float   Rotation of surface elements around world Z.
    radius        : float   Disc radius.
    half_side     : float   Square half-side.
    n_segs        : int     Polygon segments for disc approximation.
    divided       : bool    Split each square into 4 quadrant panels.
    gap           : float   Outward offset of each quadrant from the panel centre.
    """

    def __init__(self, width=2.0, depth=1.0, height=2.0, direction='x',
                 rows=1, levels=6, row_spacing=None, active_levels=None,
                 rot_z_deg=45., radius=0.1, half_side=0.1,
                 n_segs=40, divided=True, gap=0.01):
        self.width     = width
        self.depth     = depth
        self.height    = height
        self.direction = direction   # 'x' or 'y'
        self.rows      = rows
        self.levels    = levels
        self.row_spacing  = row_spacing if row_spacing is not None else (
            width if direction == 'x' else depth)
        self.active_levels = active_levels if active_levels is not None else levels
        self.rot_z_deg = rot_z_deg
        self.radius    = radius
        self.half_side = half_side
        self.n_segs    = n_segs
        self.divided   = divided
        self.gap       = gap

    def nodes(self):
        """Yield (x, y, z) for every node on the two facing sides."""
        hw = self.width  / 2.
        hd = self.depth  / 2.
        hs = self.row_spacing / 2.   # half the inter-row spacing
        r  = self.rows
        lv = self.levels

        al = max(1, min(int(self.active_levels), lv))   # clamp to valid range

        if self.direction == 'x':
            # ±X — nodes vary in Y and Z; X position set by row_spacing
            lin_y = [-hd + self.depth  * i / (r  - 1) for i in range(r)]  if r  > 1 else [0.]
            lin_z = ([-self.height / 2. + self.height * i / (lv - 1) for i in range(lv)] if lv > 1 else [0.])[-al:]
            for fx in (-hs, hs):
                for y in lin_y:
                    for z in lin_z:
                        yield (fx, y, z)
        else:                           # 'y'
            # ±Y — nodes vary in X and Z; Y position set by row_spacing
            lin_x = [-hw + self.width  * i / (r  - 1) for i in range(r)]  if r  > 1 else [0.]
            lin_z = ([-self.height / 2. + self.height * i / (lv - 1) for i in range(lv)] if lv > 1 else [0.])[-al:]
            for fy in (-hs, hs):
                for x in lin_x:
                    for z in lin_z:
                        yield (x, fy, z)

    def make_disc(self, center):
        return VerticalDisc(center, self.rot_z_deg, self.radius, self.n_segs)

    def make_square(self, center):
        return VerticalSquare(center, self.rot_z_deg, self.half_side,
                              self.divided, self.gap)

    def disc_polygons(self):
        result = []
        for p in self.nodes():
            result.extend(self.make_disc(p).polygons())
        return result

    def square_polygons(self):
        result = []
        for p in self.nodes():
            result.extend(self.make_square(p).polygons())
        return result


# ── Viewer ────────────────────────────────────────────────────────────────────

class GeometryViewer:
    """Interactive matplotlib viewer — switch between CubeSurface and Sudare."""

    def __init__(self, cube_model: CubeSurface, sudare_model: Sudare):
        self.cube   = cube_model
        self.sudare = sudare_model
        self.state  = {'model_type': 'cube', 'divided': cube_model.divided}
        self._build_figure()

    def _build_figure(self):
        m   = self.cube
        fig = plt.figure(figsize=(13, 10))
        self.fig = fig
        self.ax  = fig.add_axes([0.05, 0.38, 0.90, 0.57], projection='3d')

        ax_radius     = fig.add_axes([0.15, 0.30, 0.70, 0.025])
        ax_halfside   = fig.add_axes([0.15, 0.26, 0.70, 0.025])
        ax_gap        = fig.add_axes([0.15, 0.22, 0.70, 0.025])
        ax_rot        = fig.add_axes([0.15, 0.18, 0.70, 0.025])
        ax_rowspace   = fig.add_axes([0.15, 0.14, 0.70, 0.025])
        ax_activelevs = fig.add_axes([0.15, 0.10, 0.70, 0.025])

        self.sl_radius     = widgets.Slider(ax_radius,     'Disc radius',    0.01, 0.20, valinit=m.radius,    valstep=0.005)
        self.sl_halfside   = widgets.Slider(ax_halfside,   'Square half',    0.01, 0.20, valinit=m.half_side, valstep=0.005)
        self.sl_gap        = widgets.Slider(ax_gap,        'Gap',            0.00, 0.10, valinit=m.gap,       valstep=0.002)
        self.sl_rot        = widgets.Slider(ax_rot,        'Rotation °',     0.,   90.,  valinit=m.rot_z_deg, valstep=1.)
        _rs_max            = max(self.sudare.width, self.sudare.depth) * 2.
        self.sl_rowspace   = widgets.Slider(ax_rowspace,   'Row spacing',    0.1,  _rs_max, valinit=self.sudare.row_spacing, valstep=0.05)
        _lv                = self.sudare.levels
        self.sl_activelevs = widgets.Slider(ax_activelevs, 'Active levels',  1,    _lv,  valinit=self.sudare.active_levels, valstep=1)

        ax_divided = fig.add_axes([0.08, 0.03, 0.18, 0.04])
        ax_model   = fig.add_axes([0.32, 0.03, 0.22, 0.04])
        ax_dir     = fig.add_axes([0.60, 0.03, 0.22, 0.04])

        self.btn_divided = widgets.Button(ax_divided, f'Divided: {m.divided}')
        self.btn_model   = widgets.Button(ax_model,   'Model: Cube')
        self.btn_dir     = widgets.Button(ax_dir,     f'Dir: {self.sudare.direction.upper()}  (sudare)')

        self.sl_radius.on_changed(self._on_slider)
        self.sl_halfside.on_changed(self._on_slider)
        self.sl_gap.on_changed(self._on_slider)
        self.sl_rot.on_changed(self._on_slider)
        self.sl_rowspace.on_changed(self._on_slider)
        self.sl_activelevs.on_changed(self._on_slider)
        self.btn_divided.on_clicked(self._on_divided)
        self.btn_model.on_clicked(self._on_model_type)
        self.btn_dir.on_clicked(self._on_direction)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _active_model(self):
        return self.cube if self.state['model_type'] == 'cube' else self.sudare

    def _sync_model(self):
        r    = self.sl_radius.val
        half = self.sl_halfside.val
        gap  = self.sl_gap.val
        rot  = self.sl_rot.val
        div  = self.state['divided']
        for m in (self.cube, self.sudare):
            m.radius    = r
            m.half_side = half
            m.gap       = gap
            m.rot_z_deg = rot
            m.divided   = div
        self.sudare.row_spacing   = self.sl_rowspace.val
        self.sudare.active_levels = int(self.sl_activelevs.val)

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self):
        self._sync_model()
        m  = self._active_model()
        ax = self.ax
        ax.cla()

        ax.add_collection3d(Poly3DCollection(
            m.disc_polygons(),   alpha=0.35, linewidths=0.15,
            facecolor='royalblue', edgecolor='steelblue'))
        ax.add_collection3d(Poly3DCollection(
            m.square_polygons(), alpha=0.40, linewidths=0.15,
            facecolor='darkorange', edgecolor='sienna'))

        node_pts = list(m.nodes())
        xs, ys, zs = zip(*node_pts)
        ax.scatter(xs, ys, zs, c='black', s=3, alpha=0.4, depthshade=False)

        if self.state['model_type'] == 'cube':
            self._draw_cube_frame(ax, m)
            lim = m.cube_size * 0.78
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        else:
            self._draw_sudare_frame(ax, m)
            lim = max(m.width / 2., m.depth / 2., m.height / 2.) * 1.3
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(
            f'[{self.state["model_type"].upper()}]  '
            f'Disc r={m.radius:.3f}  |  Square half={m.half_side:.3f}  '
            f'gap={m.gap:.3f}  rot={m.rot_z_deg:.0f}°  |  '
            f'{"divided" if m.divided else "solid"}',
            fontsize=9)
        ax.set_box_aspect([1, 1, 1])
        self.fig.canvas.draw_idle()

    def _draw_cube_frame(self, ax, m):
        hs = m.cube_size / 2.
        v  = np.array([[-hs,-hs,-hs],[ hs,-hs,-hs],[ hs, hs,-hs],[-hs, hs,-hs],
                       [-hs,-hs, hs],[ hs,-hs, hs],[ hs, hs, hs],[-hs, hs, hs]])
        for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]:
            ax.plot(*zip(v[a], v[b]), 'k-', lw=0.9, alpha=0.35)

    def _draw_sudare_frame(self, ax, m):
        hw, hd, hh = m.width / 2., m.depth / 2., m.height / 2.
        hs = m.row_spacing / 2.   # actual row plane positions
        if m.direction == 'x':
            # outer bounding frame
            for fx in (-hw, hw):
                corners = [(fx, -hd, -hh), (fx, hd, -hh), (fx, hd, hh), (fx, -hd, hh)]
                for i in range(4):
                    a, b = corners[i], corners[(i + 1) % 4]
                    ax.plot(*zip(a, b), 'k-', lw=0.9, alpha=0.35)
            for y, z in [(-hd, -hh), (hd, -hh), (hd, hh), (-hd, hh)]:
                ax.plot([-hw, hw], [y, y], [z, z], 'k-', lw=0.9, alpha=0.35)
            # row planes (dashed blue when different from frame edge)
            if abs(hs - hw) > 1e-6:
                for fx in (-hs, hs):
                    corners = [(fx, -hd, -hh), (fx, hd, -hh), (fx, hd, hh), (fx, -hd, hh)]
                    for i in range(4):
                        a, b = corners[i], corners[(i + 1) % 4]
                        ax.plot(*zip(a, b), 'b--', lw=0.8, alpha=0.5)
        else:
            # outer bounding frame
            for fy in (-hd, hd):
                corners = [(-hw, fy, -hh), (hw, fy, -hh), (hw, fy, hh), (-hw, fy, hh)]
                for i in range(4):
                    a, b = corners[i], corners[(i + 1) % 4]
                    ax.plot(*zip(a, b), 'k-', lw=0.9, alpha=0.35)
            for x, z in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
                ax.plot([x, x], [-hd, hd], [z, z], 'k-', lw=0.9, alpha=0.35)
            # row planes (dashed blue when different from frame edge)
            if abs(hs - hd) > 1e-6:
                for fy in (-hs, hs):
                    corners = [(-hw, fy, -hh), (hw, fy, -hh), (hw, fy, hh), (-hw, fy, hh)]
                    for i in range(4):
                        a, b = corners[i], corners[(i + 1) % 4]
                        ax.plot(*zip(a, b), 'b--', lw=0.8, alpha=0.5)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_slider(self, _):
        self.draw()

    def _on_divided(self, _):
        self.state['divided'] = not self.state['divided']
        self.btn_divided.label.set_text(f'Divided: {self.state["divided"]}')
        self.draw()

    def _on_model_type(self, _):
        self.state['model_type'] = 'sudare' if self.state['model_type'] == 'cube' else 'cube'
        self.btn_model.label.set_text(
            'Model: Cube' if self.state['model_type'] == 'cube' else 'Model: Sudare')
        self.draw()

    def _on_direction(self, _):
        self.sudare.direction = 'y' if self.sudare.direction == 'x' else 'x'
        self.btn_dir.label.set_text(f'Dir: {self.sudare.direction.upper()}  (sudare)')
        self.draw()

    def show(self):
        self.draw()
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cube = CubeSurface(
        cube_size=1.0, grid_n=5, rot_z_deg=45.,
        radius=0.1, half_side=0.1, n_segs=40, divided=True, gap=0.01,
    )
    sudare = Sudare(
        width=2.0, depth=1.0, height=2.0, direction='x',
        rows=1, levels=6, row_spacing=1.5, active_levels=6,
        rot_z_deg=45., radius=0.1, half_side=0.1, n_segs=40, divided=True, gap=0.01,
    )
    viewer = GeometryViewer(cube, sudare)
    viewer.show()
