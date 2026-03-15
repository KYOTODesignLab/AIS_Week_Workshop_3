import os
import sys
import cv2
import numpy as np
from compas.datastructures import Mesh
from compas.geometry import Box, Frame, Point, Torus, Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_surface_model import CubeSurface

_TEXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "elements", "textures", "leopard_texture.jpg")
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "elements", "models", "AIS26_03_mesh.obj")

# -- Configuration -----------------------------------------------------------

class MarkerConfig:
    """YOLO label names and physical spacing for the 4-marker layout."""

    def __init__(self, origin_label="o", x_label="x",
                 y_label="tri_o", corner_label="tri_x", dist=1.0,
                 geometry_labels=None, modifier_labels=None):
        self.origin_label = origin_label
        self.x_label      = x_label
        self.y_label      = y_label
        self.corner_label = corner_label
        self.dist         = dist

        # Abstract 3-D positions in the local marker coordinate system
        self.positions = {
            origin_label: Point(0,    0,    0),
            x_label:      Point(dist, 0,    0),
            y_label:      Point(0,    dist, 0),
            corner_label: Point(dist, dist, 0),
        }

        # Base markers define the coordinate frame (solvePnP uses these)
        self.base_roles = [origin_label, x_label, y_label, corner_label]
        self.all_roles  = self.base_roles

        # Geometry markers: each detected instance places a shape in the scene
        self.geometry_labels = geometry_labels or []   # e.g. ["cube", "sphere"]

        # Modifier markers: each detected instance alters nearby geometry
        self.modifier_labels = modifier_labels or []   # e.g. ["scale", "delete"]


DEFAULT_CONFIG = MarkerConfig()


# -- Data ---------------------------------------------------------------------

class Marker:

    def __init__(self, name: str, confidence: float, image_point: tuple, size: float,
                 role_type: str = "base"):
        self.name        = name
        self.confidence  = confidence
        self.image_point = image_point
        self.size        = size
        self.role_type   = role_type  # "base", "geometry", or "modifier"
        self.world_point: Point = None  # set by BaseFrame after pose is solved
        self.world_size: float  = None  # bounding-box side length in world units
        self.modifiers: list    = []    # modifier Markers nearby (geometry only)

    def __repr__(self):
        return f"Marker({self.name!r}, conf={self.confidence:.2f})"


# -- Step 1: detect markers --------------------------------------------------

BORDER_PAD      = 15
MIN_INNER_STD   = 30
MIN_BORDER_MEAN = 140


class MarkerDetector:

    def __init__(self, model, config: MarkerConfig = None):
        self.model         = model
        self.config        = config or DEFAULT_CONFIG
        self._last_results = None

    def detect(self, frame: np.ndarray) -> list:
        self._last_results = self.model(frame, verbose=False)[0]
        cfg  = self.config
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # computed once for all boxes

        named = {}   # role → Marker

        geo_set  = set(cfg.geometry_labels)
        mod_set  = set(cfg.modifier_labels)
        base_set = set(cfg.base_roles)

        for box, cls, conf in zip(self._last_results.boxes.xyxy,
                                   self._last_results.boxes.cls,
                                   self._last_results.boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            size = max(x2 - x1, y2 - y1)
            name   = self.model.names[int(cls.item())]
            c      = float(conf.item())

            valid, _, _ = self._stats(gray, frame.shape[:2], x1, y1, x2, y2)
            if not valid:
                continue

            if name in base_set:
                named[name] = Marker(name, c, (cx, cy), size, role_type="base")
            elif name in geo_set:
                named[name] = Marker(name, c, (cx, cy), size, role_type="geometry")
            elif name in mod_set:
                named[name] = Marker(name, c, (cx, cy), size, role_type="modifier")

        return list(named.values())

    def draw_detections(self, img: np.ndarray):
        if self._last_results is None:
            return
        overlay = img.copy()
        for box, cls, conf in zip(self._last_results.boxes.xyxy,
                                   self._last_results.boxes.cls,
                                   self._last_results.boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            name  = self.model.names[int(cls.item())]
            fh, fw = img.shape[:2]
            gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            valid, _, _ = self._stats(gray, (fh, fw), x1, y1, x2, y2)
            color = (0, 200, 0) if valid else (120, 120, 120)
            alpha = 0.6 if valid else 0.4
            label = f"{name} {float(conf.item()):.2f}"
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(overlay, (ix1, iy1 - th - 6), (ix1 + tw + 4, iy1), color, -1)
            cv2.putText(overlay, label, (ix1 + 2, iy1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    def _stats(self, gray: np.ndarray, frame_shape: tuple, x1, y1, x2, y2):
        fh, fw = frame_shape
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        ox1 = max(0,  ix1 - BORDER_PAD);  oy1 = max(0,  iy1 - BORDER_PAD)
        ox2 = min(fw, ix2 + BORDER_PAD);  oy2 = min(fh, iy2 + BORDER_PAD)
        inner  = gray[iy1:iy2, ix1:ix2].ravel()
        outer  = gray[oy1:oy2, ox1:ox2]
        mask   = np.ones(outer.shape[:2], dtype=bool)
        mask[iy1-oy1:iy2-oy1, ix1-ox1:ix2-ox1] = False
        border = outer[mask]
        if inner.size == 0 or border.size == 0:
            return False, 0, 0
        std = int(inner.std())
        bg  = int(border.mean())
        return (std >= MIN_INNER_STD and bg >= MIN_BORDER_MEAN), std, bg


# -- Step 2: coordinate frame -------------------------------------------------

class BaseFrame:

    def __init__(self, markers: list, camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray = None, config: MarkerConfig = None):
        self._by_name = {m.name: m for m in markers}
        self.K        = camera_matrix
        self.dist     = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))
        self.config   = config or DEFAULT_CONFIG
        self.valid    = False
        self.rvec     = None
        self.tvec     = None
        self.unit     = 1.0  # world distance between o and x markers
        # Cached decompositions — populated in _solve
        self._R_mat       = None   # (3,3) rotation matrix
        self._tvec_ravel  = None   # (3,) translation
        self._Kinv        = None   # inverse camera matrix
        self._solve()

    def _solve(self):
        cfg       = self.config
        available = [k for k in cfg.base_roles if k in self._by_name]
        if len(available) < 4:  # all 4 base markers required
            return

        obj_pts = np.array(
            [[cfg.positions[k].x, cfg.positions[k].y, cfg.positions[k].z]
             for k in available], dtype=np.float64
        )

        # Normalise so that 1 unit = the o-to-x distance in the config positions
        o = np.array([cfg.positions[cfg.origin_label].x,
                      cfg.positions[cfg.origin_label].y,
                      cfg.positions[cfg.origin_label].z], dtype=np.float64)
        x = np.array([cfg.positions[cfg.x_label].x,
                      cfg.positions[cfg.x_label].y,
                      cfg.positions[cfg.x_label].z], dtype=np.float64)
        ox_dist = float(np.linalg.norm(x - o))
        if ox_dist > 0:
            obj_pts = obj_pts / ox_dist
            self.unit = ox_dist

        img_pts = np.array(
            [self._by_name[k].image_point for k in available], dtype=np.float64
        )

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, self.K, self.dist, flags=cv2.SOLVEPNP_SQPNP
        )
        if ok:
            self.valid         = True
            self.rvec          = rvec
            self.tvec          = tvec
            self._R_mat, _     = cv2.Rodrigues(rvec)
            self._tvec_ravel   = tvec.ravel()
            self._Kinv         = np.linalg.inv(self.K)
            for m in self._by_name.values():
                m.world_point = self.unproject_to_plane(m.image_point, z=0.0)
                cx, cy = m.image_point
                half   = m.size / 2.0
                wp1    = self.unproject_to_plane((cx - half, cy), z=0.0)
                wp2    = self.unproject_to_plane((cx + half, cy), z=0.0)
                dx, dy, dz = wp2.x - wp1.x, wp2.y - wp1.y, wp2.z - wp1.z
                m.world_size = float((dx*dx + dy*dy + dz*dz) ** 0.5)
            # Assign modifier markers to the geometry markers they are near
            geo_markers = [m for m in self._by_name.values() if m.role_type == "geometry"]
            mod_markers = [m for m in self._by_name.values() if m.role_type == "modifier"]
            for geo in geo_markers:
                geo.modifiers = []
                radius = 2.0 * (geo.world_size or 0.0)
                for mod in mod_markers:
                    if mod.world_point is None:
                        continue
                    dx = mod.world_point.x - geo.world_point.x
                    dy = mod.world_point.y - geo.world_point.y
                    if (dx*dx + dy*dy) ** 0.5 <= radius:
                        geo.modifiers.append(mod)

    def project(self, points: np.ndarray) -> np.ndarray:
        pts2d, _ = cv2.projectPoints(points, self.rvec, self.tvec, self.K, self.dist)
        return pts2d

    def unproject_to_plane(self, image_point: tuple, z: float = 0.0) -> Point:
        """Back-project a 2D image point to the marker plane at height z."""
        u, v = image_point
        Rt   = self._R_mat.T
        d    = self._Kinv @ np.array([u, v, 1.0])
        Rtd  = Rt @ d
        Rtt  = Rt @ self._tvec_ravel
        lam  = (z + Rtt[2]) / Rtd[2]
        p    = Rt @ (lam * d - self._tvec_ravel)
        return Point(float(p[0]), float(p[1]), float(p[2]))

    @property
    def markers(self):
        return list(self._by_name.values())

    @property
    def missing_roles(self):
        return [k for k in self.config.all_roles if k not in self._by_name]

    @property
    def origin_confidence(self):
        m = self._by_name.get(self.config.origin_label)
        return m.confidence if m else 0.0

    def confidence_color(self):
        c = self.origin_confidence
        return (0, int(255 * c), int(255 * (1.0 - c)))


# -- Step 3: scene geometry --------------------------------------------------

class PlacedBox:

    def __init__(self, size=1.0, frame: Frame = None,
                 instances: int = 1, step: Vector = None,
                 anchor: str = None):
        if isinstance(size, (int, float)):
            size = (float(size), float(size), float(size))

        self.size      = size
        self.frame     = frame or Frame(Point(0.5, 0.5, 1.0), Vector(1, 0, 0), Vector(0, 1, 0))
        self.instances = instances
        self.step      = step or Vector(0.5, 0.5, 1.0)
        self.anchor    = anchor

    def compas_boxes(self, anchor_offset: Point = None) -> list:
        xsize, ysize, zsize = self.size
        dx = anchor_offset.x if anchor_offset else 0.0
        dy = anchor_offset.y if anchor_offset else 0.0
        dz = anchor_offset.z if anchor_offset else 0.0
        result = []
        for i in range(self.instances):
            origin = Point(
                dx + self.frame.point.x + self.step.x * i,
                dy + self.frame.point.y + self.step.y * i,
                dz + self.frame.point.z + self.step.z * i,
            )
            result.append(Box(xsize=xsize, ysize=ysize, zsize=zsize,
                              frame=Frame(origin, self.frame.xaxis, self.frame.yaxis)))
        return result


class PlaceTorus:

    def __init__(self, torus_radius: float = 1.0, tube_radius: float = 0.5,
                 frame: Frame = None, anchor: str = None):
        self.torus_radius = torus_radius
        self.tube_radius  = tube_radius
        self.frame        = frame or Frame(Point(0, 0, 0.5), Vector(1, 0, 0), Vector(0, 1, 0))
        self.anchor       = anchor

    def compas_tori(self, anchor_offset: Point = None) -> list:
        dx = anchor_offset.x if anchor_offset else 0.0
        dy = anchor_offset.y if anchor_offset else 0.0
        dz = anchor_offset.z if anchor_offset else 0.0
        origin = Point(
            dx + self.frame.point.x,
            dy + self.frame.point.y,
            dz + self.frame.point.z,
        )
        f = Frame(origin, self.frame.xaxis, self.frame.yaxis)
        return [Torus(self.torus_radius, self.tube_radius, frame=f)]


class PlacedMesh:
    """Load and place an OBJ mesh — pure Python / NumPy, no extra libraries.

    Parses 'v' (vertex) and 'f' (face) lines from the file.
    Triangles and quads are both handled. Vertex normals and UV lines are
    ignored. Face indices use the OBJ v/vt/vn notation — only the vertex
    component is used.
    """

    def __init__(self, filepath: str = None, scale: float = 1.0,
                 offset: tuple = (0., 0., 0.)):
        self.filepath = filepath or _MODEL_PATH
        self.scale    = scale
        self.offset   = np.array(offset, dtype=float)
        self.vertices, self.faces = self._load()

    def _load(self):
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext != '.obj':
            raise ValueError(
                f"Only OBJ files are supported (got '{ext}'). "
                "Export your mesh as OBJ from the originating application."
            )
        verts, faces = [], []
        with open(self.filepath) as fh:
            for line in fh:
                tok = line.strip().split()
                if not tok:
                    continue
                if tok[0] == 'v':
                    verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
                elif tok[0] == 'f':
                    faces.append([int(p.split('/')[0]) - 1 for p in tok[1:]])
        return np.array(verts, dtype=np.float64), faces

    def world_vertices(self):
        """Return scaled + translated vertices as ndarray (N, 3)."""
        return self.vertices * self.scale + self.offset


class PlacedCubeSurface:
    """Place a CubeSurface geometry model in the AR scene."""

    def __init__(self, cube_size=1.0, grid_n=5, rot_z_deg=45.,
                 radius=0.1, half_side=0.1, n_segs=40, divided=True, gap=0.01,
                 offset: tuple = (0., 0., 0.)):
        self.model  = CubeSurface(cube_size=cube_size, grid_n=grid_n,
                                  rot_z_deg=rot_z_deg, radius=radius,
                                  half_side=half_side, n_segs=n_segs,
                                  divided=divided, gap=gap)
        self.offset = np.array(offset, dtype=float)

    def world_polygons(self):
        """Return (disc_polys, square_polys) — lists of ndarray (N, 3) in world space."""
        discs   = [p + self.offset for p in self.model.disc_polygons()]
        squares = [p + self.offset for p in self.model.square_polygons()]
        return discs, squares


class Scene:

    def __init__(self):
        self.shapes = []

    def add_box(self, size=1.0, frame: Frame = None,
                instances: int = 1, step: Vector = None,
                anchor: str = None) -> PlacedBox:
        shape = PlacedBox(size=size, frame=frame, instances=instances,
                          step=step, anchor=anchor)
        self.shapes.append(shape)
        return shape

    def add_torus(self, torus_radius: float = 1.0, tube_radius: float = 0.5,
                  frame: Frame = None, anchor: str = None) -> PlaceTorus:
        shape = PlaceTorus(torus_radius=torus_radius, tube_radius=tube_radius,
                           frame=frame, anchor=anchor)
        self.shapes.append(shape)
        return shape

    def add_mesh(self, filepath: str = None, scale: float = 1.0,
                 offset: tuple = (0., 0., 0.)) -> PlacedMesh:
        """Place an OBJ mesh in the scene at a fixed world offset."""
        shape = PlacedMesh(filepath=filepath, scale=scale, offset=offset)
        self.shapes.append(shape)
        return shape

    def add_cube_surface(self, cube_size=1.0, grid_n=5, rot_z_deg=45.,
                         radius=0.1, half_side=0.1, n_segs=40,
                         divided=True, gap=0.01,
                         offset: tuple = (0., 0., 0.)) -> PlacedCubeSurface:
        """Place a CubeSurface geometry model in the scene."""
        shape = PlacedCubeSurface(cube_size=cube_size, grid_n=grid_n,
                                  rot_z_deg=rot_z_deg, radius=radius,
                                  half_side=half_side, n_segs=n_segs,
                                  divided=divided, gap=gap, offset=offset)
        self.shapes.append(shape)
        return shape


# -- Step 4: render ----------------------------------------------------------

class Renderer:

    AXIS_LENGTH        = 0.8
    TEXTURE_WORLD_UNITS = 2.0  # world units covered by one texture tile

    def __init__(self, marker_frame: BaseFrame):
        self.marker_frame = marker_frame
        self._texture = cv2.imread(_TEXTURE_PATH)

    def draw_scene(self, img: np.ndarray, scene: Scene,
                   alpha: float = 0.6, thickness: int = 1):
        if not self.marker_frame.valid:
            return
        color       = self.marker_frame.confidence_color()
        config      = self.marker_frame.config
        all_markers = {m.name: m for m in self.marker_frame.markers}
        R           = self.marker_frame._R_mat
        t           = self.marker_frame._tvec_ravel

        # Collect all projected faces across every shape (global painter sort)
        all_faces = []  # (depth, pts_int32, use_texture, uv_pts)
        for shape in scene.shapes:
            anchor_name   = getattr(shape, 'anchor', None)
            anchor_marker = all_markers.get(anchor_name) if anchor_name else None
            if anchor_marker is not None:
                anchor_offset = anchor_marker.world_point
            elif anchor_name and anchor_name in config.positions:
                anchor_offset = config.positions[anchor_name]
            else:
                anchor_offset = None
            modifiers   = anchor_marker.modifiers if anchor_marker is not None else []
            use_texture = "heart" in {m.name for m in modifiers} and self._texture is not None

            # ── PlacedCubeSurface: project polygons directly ──────────────────
            if isinstance(shape, PlacedCubeSurface):
                discs, squares = shape.world_polygons()
                for poly in discs:
                    pts2d = self.marker_frame.project(poly)
                    cam_z = (poly @ R.T + t)[:, 2]
                    pts_i = pts2d[:, 0, :].astype(np.int32)
                    all_faces.append((cam_z.mean(), pts_i, False, None, (225, 105, 65)))
                for poly in squares:
                    pts2d = self.marker_frame.project(poly)
                    cam_z = (poly @ R.T + t)[:, 2]
                    pts_i = pts2d[:, 0, :].astype(np.int32)
                    all_faces.append((cam_z.mean(), pts_i, False, None, (0, 140, 255)))
                continue

            # ── PlacedMesh: project vertices directly, no COMPAS ─────────────
            if isinstance(shape, PlacedMesh):
                verts_3d = shape.world_vertices()
                pts2d    = self.marker_frame.project(verts_3d)
                cam_z    = (verts_3d @ R.T + t)[:, 2]
                for face_idxs in shape.faces:
                    depth = cam_z[face_idxs].mean()
                    pts   = np.array([pts2d[i][0].astype(int) for i in face_idxs],
                                     dtype=np.int32)
                    all_faces.append((depth, pts, False, None, None))
                continue

            if isinstance(shape, PlaceTorus):
                objects = shape.compas_tori(anchor_offset=anchor_offset)
            else:
                objects = shape.compas_boxes(anchor_offset=anchor_offset)

            for obj in objects:
                mesh = obj if isinstance(obj, Mesh) else Mesh.from_shape(obj)
                vkeys    = list(mesh.vertices())
                verts_3d = np.array([mesh.vertex_coordinates(v) for v in vkeys], dtype=np.float64)
                pts2d    = self.marker_frame.project(verts_3d)
                cam_z    = (verts_3d @ R.T + t)[:, 2]
                v2i      = {v: i for i, v in enumerate(vkeys)}
                for fkey in mesh.faces():
                    idxs  = [v2i[v] for v in mesh.face_vertices(fkey)]
                    depth = cam_z[idxs].mean()
                    pts   = np.array([pts2d[i][0].astype(int) for i in idxs], dtype=np.int32)
                    if use_texture and self._texture is not None:
                        th_tex, tw_tex = self._texture.shape[:2]
                        uv_pts = np.array([
                            [verts_3d[i, 0] / self.TEXTURE_WORLD_UNITS * tw_tex,
                             verts_3d[i, 1] / self.TEXTURE_WORLD_UNITS * th_tex]
                            for i in idxs
                        ], dtype=np.float32)
                    else:
                        uv_pts = None
                    all_faces.append((depth, pts, use_texture, uv_pts, None))

        # Back-to-front sort (painter's algorithm across all shapes)
        all_faces.sort(key=lambda x: -x[0])

        # One overlay copy → one addWeighted for all flat-colour fills
        overlay = img.copy()
        for _, pts, use_tex, _, face_color in all_faces:
            if not use_tex:
                cv2.fillPoly(overlay, [pts], face_color if face_color is not None else color)
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

        # Texture fills (written directly into img)
        for _, pts, use_tex, uv_pts, _ in all_faces:
            if use_tex:
                self._apply_texture_face(img, pts, uv_pts)

        # Outlines for every face
        for _, pts, _, _, _ in all_faces:
            cv2.polylines(img, [pts], True, (0, 0, 0), thickness)

    def draw_axes(self, img: np.ndarray):
        if not self.marker_frame.valid:
            return
        L   = self.AXIS_LENGTH
        pts = self.marker_frame.project(
            np.array([[0,0,0],[L,0,0],[0,L,0],[0,0,L]], dtype=np.float64)
        )
        origin = tuple(pts[0][0].astype(int))
        for tip, color, label in [
            (pts[1], (0,   0, 255), "X"),
            (pts[2], (0, 255,   0), "Y"),
            (pts[3], (255, 100,  0), "Z"),
        ]:
            t = tuple(tip[0].astype(int))
            cv2.arrowedLine(img, origin, t, color, 2, tipLength=0.2)
            cv2.putText(img, label, t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_marker_dots(self, img: np.ndarray):
        for m in self.marker_frame.markers:
            cx, cy = m.image_point
            cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    def draw_missing_warning(self, img: np.ndarray):
        missing = self.marker_frame.missing_roles
        if missing:
            cv2.putText(img, f"Missing: {missing}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    def _apply_texture_face(self, img: np.ndarray, pts: np.ndarray, uv_pts: np.ndarray):
        """Warp self._texture onto one face using world-derived UV coords for continuity."""
        tex = self._texture
        if tex is None or uv_pts is None:
            return
        n = len(pts)
        if n < 3:
            return
        ih, iw = img.shape[:2]
        x0 = int(max(0, pts[:, 0].min()))
        y0 = int(max(0, pts[:, 1].min()))
        x1 = int(min(iw, pts[:, 0].max()))
        y1 = int(min(ih, pts[:, 1].max()))
        if x1 <= x0 or y1 <= y0:
            return
        bw, bh = x1 - x0, y1 - y0
        pts_local = (pts - np.array([x0, y0], dtype=np.int32)).astype(np.float32)
        # M maps uv (src) -> image bbox (dst); warpPerspective applies inverse per output pixel
        try:
            if n >= 4:
                M      = cv2.getPerspectiveTransform(uv_pts[:4], pts_local[:4])
                warped = cv2.warpPerspective(tex, M, (bw, bh),
                                             borderMode=cv2.BORDER_WRAP)
            else:
                M      = cv2.getAffineTransform(uv_pts[:3], pts_local[:3])
                warped = cv2.warpAffine(tex, M, (bw, bh),
                                        borderMode=cv2.BORDER_WRAP)
        except cv2.error:
            return
        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_local.astype(np.int32)], 255)
        roi = img[y0:y1, x0:x1]
        roi[mask > 0] = warped[mask > 0]

