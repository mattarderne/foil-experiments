"""
Foil board geometry definition.

The board is modeled as a 3D rectangular domain (simplified from the actual
curved shape) with key zones:
- Deck surface (top) where the rider stands and applies load
- Mast mount region (bottom center) where the foil mast bolts through
- Rails (sides) which provide structural support

Coordinate system:
  X = length (nose to tail)
  Y = width (rail to rail)
  Z = thickness (bottom to deck)

All dimensions in meters.

Reference setup:
  Board: 6'2" x 18" (1.88m x 0.457m), ~80mm thick
  Foil:  80cm mast, 800cm² front wing, 80cm fuselage, 130cm² stabilizer
  Rider: 80kg
"""

from dataclasses import dataclass, field
from typing import Optional
import os

import numpy as np

# --- Physical constants for load derivation ---
GRAVITY = 9.81
WATER_DENSITY = 1025.0  # kg/m³, seawater


@dataclass
class FoilSetup:
    """Hydrofoil geometry for deriving realistic load cases.

    All lengths in meters, areas in m².
    """
    mast_length: float = 0.80       # 80cm mast
    mast_chord: float = 0.015       # ~15mm mast profile thickness
    wing_area: float = 0.0800       # 800cm² front wing
    wing_span: float = 0.72         # ~72cm span for 800cm² high-AR wing
    fuselage_length: float = 0.80   # 80cm fuselage
    stab_area: float = 0.0130       # 130cm² stabilizer

    # Center of pressure offsets from mast base (positive = aft/below)
    wing_depth: float = 0.65        # wing sits ~65cm below mast base
    wing_aft_offset: float = 0.10   # wing COP ~10cm aft of mast axis
    stab_depth: float = 0.60        # stab ~60cm below
    stab_aft_offset: float = 0.70   # stab is at end of fuselage


@dataclass
class LoadCase:
    """A load scenario applied to the board.

    Attributes:
        name: Human-readable name for this load case.
        deck_pressure: Pressure distribution on deck (Pa) as (N_nodes,) array,
            or a scalar for uniform pressure.
        mast_force: Force vector [Fx, Fy, Fz] at mast mount (N).
        mast_torque: Torque vector [Tx, Ty, Tz] at mast mount (N·m).
        weight_rider_kg: Rider weight used to derive deck_pressure if not set.
    """

    name: str = "standing"
    deck_pressure: Optional[np.ndarray] = None
    mast_force: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -785.0]))
    mast_torque: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    weight_rider_kg: float = 85.0

    def get_deck_force_total(self) -> float:
        """Total downward force from rider (N)."""
        return self.weight_rider_kg * GRAVITY


@dataclass
class FoilBoard:
    """Foil board geometry.

    Default: 6'2" x 18" prone foil board.

    Attributes:
        length: Board length along X (m).
        width: Board width along Y (m).
        thickness: Board thickness along Z (m).
        mast_mount_x: X-position of mast mount center (fraction of length).
        mast_mount_y: Y-position of mast mount center (fraction of width).
        mast_mount_length: Length of mast mounting plate (m).
        mast_mount_width: Width of mast mounting plate (m).
        foot_zone_x: X-range for foot placement (fraction of length) [front, back].
        foot_zone_y: Y-range for foot placement (fraction of width) [left, right].
    """

    # TT60 - Ken Adgate Inspired (Shape3d design)
    # 5'5" wing foil board — all positions parsed from .s3dx
    length: float = 1.641            # 164.1cm from Shape3d
    width: float = 0.495             # 49.5cm
    thickness: float = 0.117         # 11.7cm

    # Twin Track mast mount: parsed from .s3dx Box "Twin Track"
    # Shape3d X=48.49cm from tail → optimizer X = 70.5% from nose
    # Track: 30cm long, 1.6cm wide each, spaced 9cm apart (Y=±4.5cm)
    mast_mount_x: float = 0.705     # 70.5% from nose (from .s3dx)
    mast_mount_y: float = 0.50      # centered
    mast_mount_length: float = 0.300  # 30cm twin track
    mast_mount_width: float = 0.090   # 9cm track spacing

    # Foot pads from .s3dx Box positions
    # Back footstrap: X=44.01cm from tail → 73.2% from nose, L=30cm, W=4.1cm
    # Front FS Centre: X=92.10cm from tail → 43.9% from nose, L=14.2cm, W=4.1cm
    front_foot_x: float = 0.439     # from .s3dx "Front FS Centre"
    back_foot_x: float = 0.732      # from .s3dx "back footsrap"
    foot_pad_length: float = 0.142   # 14.2cm (from Front FS Centre box)
    foot_pad_width: float = 0.12     # ~12cm rider foot contact area

    # Weight distribution: wing foiling is more back-foot weighted
    front_foot_weight: float = 0.35  # back-foot dominant in wing foiling

    def get_mast_mount_bounds(self) -> tuple:
        """Return (x_min, x_max, y_min, y_max) of the mast mount region in meters."""
        cx = self.mast_mount_x * self.length
        cy = self.mast_mount_y * self.width
        hl = self.mast_mount_length / 2
        hw = self.mast_mount_width / 2
        return (cx - hl, cx + hl, cy - hw, cy + hw)

    def get_front_foot_bounds(self) -> tuple:
        """Return (x_min, x_max, y_min, y_max) of front foot pad in meters."""
        cx = self.front_foot_x * self.length
        cy = 0.5 * self.width
        hl = self.foot_pad_length / 2
        hw = self.foot_pad_width / 2
        return (cx - hl, cx + hl, cy - hw, cy + hw)

    def get_back_foot_bounds(self) -> tuple:
        """Return (x_min, x_max, y_min, y_max) of back foot pad in meters."""
        cx = self.back_foot_x * self.length
        cy = 0.5 * self.width
        hl = self.foot_pad_length / 2
        hw = self.foot_pad_width / 2
        return (cx - hl, cx + hl, cy - hw, cy + hw)

    def get_foot_zone_bounds(self) -> tuple:
        """Return combined bounding box of both feet (for backward compat)."""
        ff = self.get_front_foot_bounds()
        bf = self.get_back_foot_bounds()
        return (
            min(ff[0], bf[0]), max(ff[1], bf[1]),
            min(ff[2], bf[2]), max(ff[3], bf[3]),
        )

    def get_domain_shape(self) -> tuple:
        """Return (Lx, Ly, Lz) domain dimensions."""
        return (self.length, self.width, self.thickness)

    def is_in_mast_mount(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask for points inside the mast mount region."""
        xmin, xmax, ymin, ymax = self.get_mast_mount_bounds()
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    def is_in_mast_mount_expanded(
        self, x: np.ndarray, y: np.ndarray, margin: float = 0.0
    ) -> np.ndarray:
        """Boolean mask with expanded bounds (for coarse meshes)."""
        xmin, xmax, ymin, ymax = self.get_mast_mount_bounds()
        return (
            (x >= xmin - margin) & (x <= xmax + margin)
            & (y >= ymin - margin) & (y <= ymax + margin)
        )

    def is_in_front_foot(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.get_front_foot_bounds()
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    def is_in_back_foot(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xmin, xmax, ymin, ymax = self.get_back_foot_bounds()
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    def is_in_foot_zone(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask for points inside either foot pad."""
        return self.is_in_front_foot(x, y) | self.is_in_back_foot(x, y)


def create_default_load_cases(
    rider_kg: float = 85.0,   # from Shape3d TT60 design
    foil: Optional[FoilSetup] = None,
) -> list:
    """Create load cases derived from actual foil physics.

    Forces at the mast base are calculated from:
    - Wing/stab lift balancing rider weight
    - Hydrodynamic drag through the mast
    - Lever arms from mast length, fuselage geometry
    - Dynamic multipliers for each scenario

    Reference setup: 80kg rider, 80cm mast, 800cm² wing, 80cm fuse, 130cm² stab.

    Returns:
        List of LoadCase objects covering typical riding scenarios.
    """
    if foil is None:
        foil = FoilSetup()

    W = rider_kg * GRAVITY  # 785 N for 80kg

    # --- Steady-state force derivation ---
    # At equilibrium foiling (~15-18 knots / 8-9 m/s):
    #   Wing lift ≈ W (supports rider weight)
    #   Drag through mast ≈ 100-200N (wing + mast + fuse parasitic)
    #   Pitch moment at mast base = wing_lift * wing_aft_offset
    #                              = 785 * 0.10 = 79 Nm

    v_cruise = 8.5   # m/s (~16.5 knots, typical prone foiling)
    q = 0.5 * WATER_DENSITY * v_cruise**2  # dynamic pressure: ~37 kPa

    # Drag estimate: CD ~0.015 for wing, plus mast and fuse
    drag_wing = q * foil.wing_area * 0.015       # ~44N
    drag_mast = q * (foil.mast_length * foil.mast_chord) * 0.01  # ~4.4N
    drag_fuse = q * 0.001 * 0.02                   # small
    total_drag = drag_wing + drag_mast + drag_fuse  # ~50N at cruise

    # Pitch moment: wing lift offset from mast axis
    pitch_moment_cruise = W * foil.wing_aft_offset  # ~79 Nm

    cases = [
        # --- Normal riding at cruise speed ---
        # Rider weight on deck, mast transmits lift + drag + pitch moment
        # Mast force at board: drag forward (X), lift pulling up on bottom
        # but board sees it as downward load on deck above mount
        LoadCase(
            name="riding_normal",
            weight_rider_kg=rider_kg,
            mast_force=np.array([
                total_drag,         # ~50N forward (drag transmitted up mast)
                0.0,                # no lateral
                -W,                 # wing lift = rider weight
            ]),
            mast_torque=np.array([
                0.0,
                pitch_moment_cruise,  # ~79 Nm pitch (nose-up from wing offset)
                0.0,
            ]),
        ),

        # --- Pumping (dynamic, peak forces) ---
        # During pump stroke, rider generates 1.5-2x body weight peaks
        # Increased pitch moment from aggressive angle changes
        LoadCase(
            name="pumping",
            weight_rider_kg=rider_kg * 1.5,  # 1.5g peak during pump
            mast_force=np.array([
                total_drag * 0.7,   # lower speed during pump
                0.0,
                -W * 1.8,           # peak lift during pump downstroke
            ]),
            mast_torque=np.array([
                0.0,
                pitch_moment_cruise * 3.0,  # ~240 Nm: aggressive pitch cycling
                0.0,
            ]),
        ),

        # --- Jump landing ---
        # Impact load: 3-4g on landing
        # Mostly vertical, minimal lateral
        LoadCase(
            name="jump_landing",
            weight_rider_kg=rider_kg * 3.5,  # 3.5g impact
            mast_force=np.array([
                0.0,                # no forward speed at impact
                0.0,
                -W * 3.5,           # impact through mast
            ]),
            mast_torque=np.array([
                0.0,
                pitch_moment_cruise * 0.5,  # reduced (no speed)
                0.0,
            ]),
        ),

        # --- Hard carve / turn ---
        # Lateral loading from banked turn, ~30-40 degree lean
        # Lateral force ≈ W * tan(35°) ≈ 550N
        # Creates roll moment through 80cm mast
        LoadCase(
            name="carving",
            weight_rider_kg=rider_kg * 1.2,  # slight increase in apparent weight
            mast_force=np.array([
                total_drag * 1.5,   # increased drag in turn
                W * 0.7,            # ~550N lateral (tan 35°)
                -W * 1.2,           # increased vertical in turn
            ]),
            mast_torque=np.array([
                W * 0.7 * foil.mast_length * 0.5,  # ~220 Nm roll from lateral
                pitch_moment_cruise,                 # pitch unchanged
                total_drag * foil.mast_length * 0.3, # ~12 Nm yaw
            ]),
        ),
    ]
    return cases


class BoardShape:
    """3D board shape parsed from a Shape3d .s3dx file.

    Provides vectorized is_inside(x, y, z) testing for the optimizer
    to mark elements as void (outside the board) or shell (on the surface).

    The board shape is defined by three Bezier curves:
    - Outline (Otl): planform, X vs half-width from centerline
    - StrBot: bottom rocker profile, X vs Z
    - StrDeck: deck profile, X vs Z

    All internal data stored in meters.
    """

    def __init__(self, s3dx_path: str):
        import xml.etree.ElementTree as ET

        tree = ET.parse(s3dx_path)
        board_el = tree.getroot().find('Board')

        self.board_length = float(board_el.find('Length').text) / 100.0
        self.board_width = float(board_el.find('Width').text) / 100.0

        # Parse the three defining curves (Shape3d has X=0 at tail)
        otl_curve = self._eval_curve(board_el.find('Otl').find('Bezier3d'), 0, 1)
        bot_curve = self._eval_curve(board_el.find('StrBot').find('Bezier3d'), 0, 2)
        deck_curve = self._eval_curve(board_el.find('StrDeck').find('Bezier3d'), 0, 2)

        # Convert to meters
        otl_x_raw = otl_curve[:, 0] / 100.0
        otl_hw_raw = otl_curve[:, 1] / 100.0
        bot_x_raw = bot_curve[:, 0] / 100.0
        bot_z_raw = bot_curve[:, 1] / 100.0
        deck_x_raw = deck_curve[:, 0] / 100.0
        deck_z_raw = deck_curve[:, 1] / 100.0

        # Flip X: Shape3d X=0 is tail, optimizer X=0 is nose
        # X_optimizer = board_length - X_shape3d, then reverse for sorted order
        self._otl_x = self.board_length - otl_x_raw[::-1]
        self._otl_hw = otl_hw_raw[::-1]
        self._bot_x = self.board_length - bot_x_raw[::-1]
        self._bot_z = bot_z_raw[::-1]
        self._deck_x = self.board_length - deck_x_raw[::-1]
        self._deck_z = deck_z_raw[::-1]

        # Z range of actual board (for mapping to optimizer coords)
        self.z_min = float(self._bot_z.min())
        self.z_max = float(self._deck_z.max())

        # Parse box positions from .s3dx
        self._parse_boxes(board_el)

    def _parse_boxes(self, board_el):
        """Parse box positions from .s3dx and convert to optimizer coords."""
        self.boxes = {}
        nb_boxes_el = board_el.find('Nb_Boxes')
        if nb_boxes_el is None:
            return

        nb_boxes = int(nb_boxes_el.text)
        for bi in range(nb_boxes):
            box_wrapper = board_el.find(f'Box_{bi}')
            if box_wrapper is None:
                continue
            box_el = box_wrapper.find('Box')
            if box_el is None:
                continue

            name = box_el.find('Name').text.strip()
            center = box_el.find('PointCenter').find('Point3d')
            cx_s3d = float(center.find('x').text) / 100.0  # cm → m
            cy_s3d = float(center.find('y').text) / 100.0
            length = float(box_el.find('Length').text) / 100.0
            width = float(box_el.find('Width').text) / 100.0
            face = int(box_el.find('Face').text)  # 0=deck, 1=bottom
            even = int(box_el.find('Even').text)  # 1=mirrored

            # Convert X from Shape3d (X=0=tail) to optimizer (X=0=nose)
            cx_opt = self.board_length - cx_s3d

            self.boxes[name] = {
                'x_center': cx_opt,
                'y_offset': cy_s3d,  # offset from centerline
                'length': length,
                'width': width,
                'face': 'deck' if face == 0 else 'bottom',
                'mirrored': even == 1,
            }

    def get_mast_mount_info(self) -> dict:
        """Get mast mount position and dimensions from parsed boxes."""
        if 'Twin Track' in self.boxes:
            tt = self.boxes['Twin Track']
            return {
                'x_center': tt['x_center'],
                'length': tt['length'],
                'track_spacing': abs(tt['y_offset']) * 2,
            }
        return None

    def get_footstrap_info(self) -> dict:
        """Get footstrap positions from parsed boxes."""
        result = {}
        # Note: "back footsrap" is a typo in the .s3dx file
        for key in ['back footsrap', 'back footstrap']:
            if key in self.boxes:
                result['back'] = self.boxes[key]
                break
        if 'Front FS Centre' in self.boxes:
            result['front'] = self.boxes['Front FS Centre']
        return result

    @staticmethod
    def _read_points(element, name):
        poly = element.find(name).find('Polygone3d')
        pts = []
        for pt in poly.findall('Point3d'):
            x = float(pt.find('x').text)
            y = float(pt.find('y').text)
            z_el = pt.find('z')
            z = float(z_el.text) if z_el is not None else 0.0
            pts.append((x, y, z))
        return pts

    def _eval_curve(self, bezier_el, dim_a, dim_b, n_per_seg=100):
        cp = self._read_points(bezier_el, 'Control_points')
        t1 = self._read_points(bezier_el, 'Tangents_1')
        t2 = self._read_points(bezier_el, 'Tangents_2')

        all_pts = []
        for i in range(len(cp) - 1):
            p0 = np.array([cp[i][dim_a], cp[i][dim_b]])
            p3 = np.array([cp[i+1][dim_a], cp[i+1][dim_b]])
            p1 = np.array([t2[i][dim_a], t2[i][dim_b]])
            p2 = np.array([t1[i+1][dim_a], t1[i+1][dim_b]])
            t = np.linspace(0, 1, n_per_seg)
            seg = (((1-t)**3)[:, None] * p0 + (3*(1-t)**2*t)[:, None] * p1 +
                   (3*(1-t)*t**2)[:, None] * p2 + (t**3)[:, None] * p3)
            if i > 0:
                seg = seg[1:]
            all_pts.append(seg)
        return np.vstack(all_pts)

    def half_width_at(self, x: np.ndarray) -> np.ndarray:
        """Interpolate half-width (m) at X positions (m)."""
        return np.interp(x, self._otl_x, self._otl_hw, left=0.0, right=0.0)

    def bot_z_at(self, x: np.ndarray) -> np.ndarray:
        """Interpolate bottom rocker Z (m, Shape3d coords) at X positions."""
        return np.interp(x, self._bot_x, self._bot_z)

    def deck_z_at(self, x: np.ndarray) -> np.ndarray:
        """Interpolate deck Z (m, Shape3d coords) at X positions."""
        return np.interp(x, self._deck_x, self._deck_z)

    def is_inside(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                  lz: float) -> np.ndarray:
        """Test if points are inside the 3D board shape.

        Args:
            x, y, z: Element center coordinates in optimizer space
                      (x: 0..length, y: 0..width, z: 0..lz).
            lz: Optimizer domain Z height (board.thickness).

        Returns:
            Boolean mask, True = inside board.
        """
        hw = self.half_width_at(x)
        center_y = self.board_width / 2.0
        dist_from_center = np.abs(y - center_y)

        # Map optimizer Z (0..lz) to Shape3d Z (z_min..z_max)
        z_shape = self.z_min + (z / lz) * (self.z_max - self.z_min)
        bot_z = self.bot_z_at(x)
        deck_z = self.deck_z_at(x)

        # Superellipse cross-section: (y/hw)^p + (z_norm)^p <= 1
        # p=2.5 gives slightly squarish rails
        y_norm = dist_from_center / np.maximum(hw, 1e-6)
        z_mid = (bot_z + deck_z) / 2.0
        z_half = (deck_z - bot_z) / 2.0
        z_norm = np.abs(z_shape - z_mid) / np.maximum(z_half, 1e-6)

        inside = (np.power(y_norm, 2.5) + np.power(z_norm, 2.5)) <= 1.0
        return inside

    def is_on_shell(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                    lz: float, thickness: float = 0.5) -> np.ndarray:
        """Test if points are on the outer shell of the board.

        Points that are inside the board but within `thickness` fraction
        of the surface are considered shell.

        Args:
            x, y, z: Element center coordinates in optimizer space.
            lz: Optimizer domain Z height.
            thickness: Shell thickness as fraction of local dimension (0-1).
                       0.5 means elements within 50% of the surface distance
                       are shell. For 1-element-thick shell on a ~12-element
                       cross-section, use ~0.15.

        Returns:
            Boolean mask, True = shell element.
        """
        hw = self.half_width_at(x)
        center_y = self.board_width / 2.0
        dist_from_center = np.abs(y - center_y)

        z_shape = self.z_min + (z / lz) * (self.z_max - self.z_min)
        bot_z = self.bot_z_at(x)
        deck_z = self.deck_z_at(x)

        y_norm = dist_from_center / np.maximum(hw, 1e-6)
        z_mid = (bot_z + deck_z) / 2.0
        z_half = (deck_z - bot_z) / 2.0
        z_norm = np.abs(z_shape - z_mid) / np.maximum(z_half, 1e-6)

        r = np.power(y_norm, 2.5) + np.power(z_norm, 2.5)
        return (r <= 1.0) & (r > (1.0 - thickness) ** 2.5)


def load_board_shape(s3dx_path: str = None) -> Optional[BoardShape]:
    """Load board shape from .s3dx file, trying default location if needed."""
    if s3dx_path and os.path.exists(s3dx_path):
        return BoardShape(s3dx_path)
    default = os.path.expanduser("~/Downloads/TT60 - Ken Adgate Inspired.s3dx")
    if os.path.exists(default):
        return BoardShape(default)
    return None
