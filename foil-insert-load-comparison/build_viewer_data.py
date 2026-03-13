#!/usr/bin/env python3
"""Build board surface and solved comparison fields for the viewer.

This is a reduced-order structural solve on the actual TT60 planform:
- top skin grid
- bottom skin grid
- through-thickness coupling springs for EPS / HD foam insert regions

It is not a full 3D continuum FE model, but it is a real solved system and is
materially more defensible than the old hand-painted shell strain proxy.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "viewer-data.json"

S3DX_CANDIDATES = [
    PROJECT_DIR.parent / "foil-board-optimizer" / "board_shape.s3dx",
    PROJECT_DIR.parent / "board-design" / "profiles" / "reference" / "TT60.s3dx",
    Path("/Users/mattbook-air/foil-experiments/foil-board-optimizer/board_shape.s3dx"),
    Path("/Users/mattbook-air/foil-experiments/board-design/profiles/reference/TT60.s3dx"),
]


def find_s3dx() -> Path:
    for path in S3DX_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate TT60 .s3dx source in expected local paths.")


def read_poly(bezier: ET.Element, name: str) -> list[tuple[float, float, float]]:
    poly = bezier.find(name).find("Polygone3d")
    points = []
    for point in poly.findall("Point3d"):
        x = float(point.find("x").text)
        y = float(point.find("y").text)
        z_el = point.find("z")
        z = float(z_el.text) if z_el is not None else 0.0
        points.append((x, y, z))
    return points


def eval_bezier_2d(
    control_points: list[tuple[float, float, float]],
    tangents_1: list[tuple[float, float, float]],
    tangents_2: list[tuple[float, float, float]],
    dim_a: int,
    dim_b: int,
    n_per_seg: int = 80,
) -> np.ndarray:
    segments = []
    for i in range(len(control_points) - 1):
        p0 = np.array([control_points[i][dim_a], control_points[i][dim_b]])
        p3 = np.array([control_points[i + 1][dim_a], control_points[i + 1][dim_b]])
        p1 = np.array([tangents_2[i][dim_a], tangents_2[i][dim_b]])
        p2 = np.array([tangents_1[i + 1][dim_a], tangents_1[i + 1][dim_b]])
        t = np.linspace(0.0, 1.0, n_per_seg)
        segment = (
            ((1 - t) ** 3)[:, None] * p0
            + (3 * (1 - t) ** 2 * t)[:, None] * p1
            + (3 * (1 - t) * t**2)[:, None] * p2
            + (t**3)[:, None] * p3
        )
        if i > 0:
            segment = segment[1:]
        segments.append(segment)
    return np.vstack(segments)


def parse_boxes(board: ET.Element, board_length: float) -> dict[str, dict[str, float]]:
    boxes: dict[str, dict[str, float]] = {}
    n_boxes_text = board.findtext("Nb_Boxes")
    n_boxes = int(n_boxes_text or "0")
    for idx in range(n_boxes):
        wrapper = board.find(f"Box_{idx}")
        if wrapper is None:
            continue
        box = wrapper.find("Box")
        if box is None:
            continue
        name = box.findtext("Name", "").strip()
        center = box.find("PointCenter").find("Point3d")
        center_x_shape = float(center.findtext("x")) / 100.0
        center_y = float(center.findtext("y")) / 100.0
        length = float(box.findtext("Length")) / 100.0
        width = float(box.findtext("Width")) / 100.0
        face = int(box.findtext("Face", "0"))
        boxes[name] = {
            "x_center": board_length - center_x_shape,
            "y_offset": center_y,
            "length": length,
            "width": width,
            "face": face,
        }
    return boxes


def parse_board_shape() -> dict:
    s3dx_path = find_s3dx()
    tree = ET.parse(s3dx_path)
    board = tree.getroot().find("Board")

    board_length = float(board.findtext("Length")) / 100.0
    board_width = float(board.findtext("Width")) / 100.0

    otl = board.find("Otl").find("Bezier3d")
    otl_cp = read_poly(otl, "Control_points")
    otl_t1 = read_poly(otl, "Tangents_1")
    otl_t2 = read_poly(otl, "Tangents_2")
    outline = eval_bezier_2d(otl_cp, otl_t1, otl_t2, 0, 1, 96)

    bot = board.find("StrBot").find("Bezier3d")
    bot_cp = read_poly(bot, "Control_points")
    bot_t1 = read_poly(bot, "Tangents_1")
    bot_t2 = read_poly(bot, "Tangents_2")
    bottom = eval_bezier_2d(bot_cp, bot_t1, bot_t2, 0, 2, 96)

    deck = board.find("StrDeck").find("Bezier3d")
    deck_cp = read_poly(deck, "Control_points")
    deck_t1 = read_poly(deck, "Tangents_1")
    deck_t2 = read_poly(deck, "Tangents_2")
    deck_curve = eval_bezier_2d(deck_cp, deck_t1, deck_t2, 0, 2, 96)

    outline_x = board_length - outline[::-1, 0] / 100.0
    outline_hw = outline[::-1, 1] / 100.0
    bottom_x = board_length - bottom[::-1, 0] / 100.0
    bottom_z = bottom[::-1, 1] / 100.0
    deck_x = board_length - deck_curve[::-1, 0] / 100.0
    deck_z = deck_curve[::-1, 1] / 100.0

    boxes = parse_boxes(board, board_length)
    mast_box = boxes.get("Twin Track")
    footstrap_front = boxes.get("Front FS Centre")
    footstrap_back = boxes.get("back footsrap") or boxes.get("back footstrap")

    return {
        "s3dx_path": str(s3dx_path),
        "length": board_length,
        "width": board_width,
        "outline_x": outline_x,
        "outline_hw": outline_hw,
        "bottom_x": bottom_x,
        "bottom_z": bottom_z,
        "deck_x": deck_x,
        "deck_z": deck_z,
        "boxes": boxes,
        "mast_x": mast_box["x_center"] if mast_box else 0.705 * board_length,
        "front_foot_x": footstrap_front["x_center"] if footstrap_front else 0.439 * board_length,
        "back_foot_x": footstrap_back["x_center"] if footstrap_back else 0.732 * board_length,
    }


def build_surface(board: dict, n_stations: int = 104, n_ring: int = 56, exponent: float = 2.55) -> dict:
    xs = np.linspace(float(board["outline_x"][2]), float(board["outline_x"][-3]), n_stations)
    outline_x = board["outline_x"]
    outline_hw = board["outline_hw"]
    bottom_x = board["bottom_x"]
    bottom_z = board["bottom_z"]
    deck_x = board["deck_x"]
    deck_z = board["deck_z"]

    center_y = board["width"] / 2.0
    hw = np.interp(xs, outline_x, outline_hw)
    z_bot = np.interp(xs, bottom_x, bottom_z)
    z_deck = np.interp(xs, deck_x, deck_z)
    z_mid = 0.5 * (z_bot + z_deck)
    z_half = 0.5 * (z_deck - z_bot)

    theta = np.linspace(0.0, 2.0 * math.pi, n_ring, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    y_profile = np.sign(cos_t) * np.abs(cos_t) ** (2.0 / exponent)
    z_profile = np.sign(sin_t) * np.abs(sin_t) ** (2.0 / exponent)

    vertices = []
    meta = []
    for station_idx, x_pos in enumerate(xs):
        local_hw = max(float(hw[station_idx]), 1e-4)
        local_mid = float(z_mid[station_idx])
        local_half = max(float(z_half[station_idx]), 1e-4)
        for ring_idx in range(n_ring):
            y = center_y + local_hw * y_profile[ring_idx]
            z = local_mid + local_half * z_profile[ring_idx]
            vertices.append([float(x_pos), float(y), float(z)])
            topness = max(float(z_profile[ring_idx]), 0.0)
            bottomness = max(-float(z_profile[ring_idx]), 0.0)
            railness = 1.0 - abs(float(z_profile[ring_idx]))
            meta.append(
                {
                    "station": station_idx,
                    "ring": ring_idx,
                    "theta": float(theta[ring_idx]),
                    "topness": topness,
                    "bottomness": bottomness,
                    "railness": railness,
                }
            )

    triangles = []
    for station_idx in range(1, n_stations):
        base = station_idx * n_ring
        prev = (station_idx - 1) * n_ring
        for ring_idx in range(n_ring):
            next_ring = (ring_idx + 1) % n_ring
            triangles.append([prev + ring_idx, base + ring_idx, base + next_ring])
            triangles.append([prev + ring_idx, base + next_ring, prev + next_ring])

    return {
        "vertices": np.array(vertices, dtype=float),
        "triangles": triangles,
        "meta": meta,
    }


def build_planform_grid(board: dict, nx: int = 96, ny: int = 34) -> dict:
    x = np.linspace(0.0, float(board["length"]), nx)
    y = np.linspace(0.0, float(board["width"]), ny)
    center_y = float(board["width"]) * 0.5
    hw = np.interp(x, board["outline_x"], board["outline_hw"], left=0.0, right=0.0)
    inside = np.abs(y[None, :] - center_y) <= hw[:, None]
    inside &= hw[:, None] > 0.01

    node_ids = -np.ones((nx, ny), dtype=int)
    node_ids[inside] = np.arange(int(np.sum(inside)))

    return {
        "x": x,
        "y": y,
        "inside": inside,
        "ids": node_ids,
        "nx": nx,
        "ny": ny,
        "n_nodes": int(np.sum(inside)),
        "dx": float(x[1] - x[0]),
        "dy": float(y[1] - y[0]),
        "center_y": center_y,
    }


def rect_mask(grid: dict, cx: float, cy: float, lx: float, ly: float) -> np.ndarray:
    xx = grid["x"][:, None]
    yy = grid["y"][None, :]
    return (
        grid["inside"]
        & (np.abs(xx - cx) <= lx * 0.5)
        & (np.abs(yy - cy) <= ly * 0.5)
    )


def distribute_load(force: np.ndarray, dofs: np.ndarray, magnitude: float, weights: np.ndarray | None = None) -> None:
    if len(dofs) == 0 or abs(magnitude) <= 1e-12:
        return
    if weights is None:
        weights = np.ones(len(dofs), dtype=float)
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-9)
    weights /= np.sum(weights)
    force[dofs] += magnitude * weights


def add_spring(rows: list[int], cols: list[int], vals: list[float], a: int, b: int, stiffness: float) -> None:
    if stiffness <= 0.0:
        return
    rows.extend([a, a, b, b])
    cols.extend([a, b, a, b])
    vals.extend([stiffness, -stiffness, -stiffness, stiffness])


def interpolate_grid(x_grid: np.ndarray, y_grid: np.ndarray, values: np.ndarray, xq: float, yq: float) -> float:
    if xq <= x_grid[0]:
        ix0 = ix1 = 0
        tx = 0.0
    elif xq >= x_grid[-1]:
        ix0 = ix1 = len(x_grid) - 1
        tx = 0.0
    else:
        ix1 = int(np.searchsorted(x_grid, xq))
        ix0 = ix1 - 1
        tx = (xq - x_grid[ix0]) / max(x_grid[ix1] - x_grid[ix0], 1e-12)

    if yq <= y_grid[0]:
        iy0 = iy1 = 0
        ty = 0.0
    elif yq >= y_grid[-1]:
        iy0 = iy1 = len(y_grid) - 1
        ty = 0.0
    else:
        iy1 = int(np.searchsorted(y_grid, yq))
        iy0 = iy1 - 1
        ty = (yq - y_grid[iy0]) / max(y_grid[iy1] - y_grid[iy0], 1e-12)

    v00 = values[ix0, iy0]
    v10 = values[ix1, iy0]
    v01 = values[ix0, iy1]
    v11 = values[ix1, iy1]

    vx0 = (1.0 - tx) * v00 + tx * v10
    vx1 = (1.0 - tx) * v01 + tx * v11
    return float((1.0 - ty) * vx0 + ty * vx1)


def solve_layered_plate(board: dict, surface: dict, variant: str, load_case: str) -> np.ndarray:
    grid = build_planform_grid(board)
    n = grid["n_nodes"]
    x = grid["x"]
    y = grid["y"]
    ids = grid["ids"]
    inside = grid["inside"]
    center_y = grid["center_y"]

    top_offset = 0
    bottom_offset = n
    ndof = 2 * n

    mast_y = center_y
    front_foot_x = float(board["front_foot_x"])
    back_foot_x = float(board["back_foot_x"])

    if variant == "tuttle":
        insert_length = 0.1905
        insert_width = 0.0508
        k_core_insert = 38.0
        top_insert_share = 0.42
        bottom_insert_share = 0.58
        top_skin_insert_boost = 2.0
        bottom_skin_insert_boost = 2.0
    else:
        insert_length = 0.5080
        insert_width = 0.1905
        k_core_insert = 7.0
        top_insert_share = 0.0
        bottom_insert_share = 1.0
        top_skin_insert_boost = 1.0
        bottom_skin_insert_boost = 2.8

    insert_mask = rect_mask(grid, float(board["mast_x"]), mast_y, insert_length, insert_width)
    front_foot_mask = rect_mask(grid, front_foot_x, center_y, 0.142, 0.12)
    back_foot_mask = rect_mask(grid, back_foot_x, center_y, 0.18, 0.12)

    case_params = {
        "vertical": {
            "mast_total": 1.0,
            "front_share": 0.50,
            "back_share": 0.50,
            "pitch_amp": 0.00,
            "roll_amp": 0.00,
        },
        "pitch": {
            "mast_total": 1.0,
            "front_share": 0.38,
            "back_share": 0.62,
            "pitch_amp": 0.65,
            "roll_amp": 0.00,
        },
        "lateral": {
            "mast_total": 0.95,
            "front_share": 0.45,
            "back_share": 0.55,
            "pitch_amp": 0.10,
            "roll_amp": 0.60,
        },
        "impulse": {
            "mast_total": 1.65,
            "front_share": 0.42,
            "back_share": 0.58,
            "pitch_amp": 0.82,
            "roll_amp": 0.08,
        },
    }[load_case]

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    x_factor = 1.0 / max(grid["dx"], 1e-6)
    y_factor = 1.0 / max(grid["dy"], 1e-6)
    k_skin_x = 6.0 * x_factor
    k_skin_y = 4.5 * y_factor
    k_core_eps = 1.8
    k_found = 0.02

    def top_dof(node_id: int) -> int:
        return top_offset + node_id

    def bottom_dof(node_id: int) -> int:
        return bottom_offset + node_id

    for ix in range(grid["nx"]):
        for iy in range(grid["ny"]):
            node_id = ids[ix, iy]
            if node_id < 0:
                continue

            in_insert = bool(insert_mask[ix, iy])
            vertical_stiffness = k_core_insert if in_insert else k_core_eps
            add_spring(rows, cols, vals, top_dof(node_id), bottom_dof(node_id), vertical_stiffness)

            rows.extend([top_dof(node_id), bottom_dof(node_id)])
            cols.extend([top_dof(node_id), bottom_dof(node_id)])
            vals.extend([k_found, k_found])

            if ix + 1 < grid["nx"] and inside[ix + 1, iy]:
                node_next = ids[ix + 1, iy]
                top_boost = top_skin_insert_boost if in_insert and insert_mask[ix + 1, iy] else 1.0
                bottom_boost = bottom_skin_insert_boost if in_insert and insert_mask[ix + 1, iy] else 1.0
                add_spring(rows, cols, vals, top_dof(node_id), top_dof(node_next), k_skin_x * top_boost)
                add_spring(rows, cols, vals, bottom_dof(node_id), bottom_dof(node_next), k_skin_x * bottom_boost)

            if iy + 1 < grid["ny"] and inside[ix, iy + 1]:
                node_next = ids[ix, iy + 1]
                top_boost = top_skin_insert_boost if in_insert and insert_mask[ix, iy + 1] else 1.0
                bottom_boost = bottom_skin_insert_boost if in_insert and insert_mask[ix, iy + 1] else 1.0
                add_spring(rows, cols, vals, top_dof(node_id), top_dof(node_next), k_skin_y * top_boost)
                add_spring(rows, cols, vals, bottom_dof(node_id), bottom_dof(node_next), k_skin_y * bottom_boost)

    K = sparse.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
    force = np.zeros(ndof, dtype=float)

    insert_nodes = np.argwhere(insert_mask)
    if len(insert_nodes) == 0:
        raise RuntimeError(f"No insert nodes found for {variant}")

    insert_node_ids = np.array([ids[ix, iy] for ix, iy in insert_nodes], dtype=int)
    insert_x = np.array([x[ix] for ix, _ in insert_nodes], dtype=float)
    insert_y = np.array([y[iy] for _, iy in insert_nodes], dtype=float)

    x_norm = (insert_x - float(board["mast_x"])) / max(insert_length * 0.5, 1e-6)
    y_norm = (insert_y - mast_y) / max(insert_width * 0.5, 1e-6)
    pitch_weights = 1.0 + case_params["pitch_amp"] * x_norm
    roll_weights = 1.0 + case_params["roll_amp"] * y_norm
    base_weights = np.clip(pitch_weights * roll_weights, 0.1, None)

    distribute_load(
        force,
        np.array([bottom_dof(node_id) for node_id in insert_node_ids], dtype=int),
        case_params["mast_total"] * bottom_insert_share,
        base_weights,
    )
    if top_insert_share > 0.0:
        top_weights = np.clip(1.0 - 0.45 * case_params["pitch_amp"] * x_norm, 0.1, None)
        distribute_load(
            force,
            np.array([top_dof(node_id) for node_id in insert_node_ids], dtype=int),
            case_params["mast_total"] * top_insert_share,
            top_weights,
        )

    front_nodes = np.argwhere(front_foot_mask)
    back_nodes = np.argwhere(back_foot_mask)
    front_ids = np.array([ids[ix, iy] for ix, iy in front_nodes], dtype=int)
    back_ids = np.array([ids[ix, iy] for ix, iy in back_nodes], dtype=int)

    distribute_load(
        force,
        np.array([top_dof(node_id) for node_id in front_ids], dtype=int),
        -case_params["mast_total"] * case_params["front_share"],
    )
    distribute_load(
        force,
        np.array([top_dof(node_id) for node_id in back_ids], dtype=int),
        -case_params["mast_total"] * case_params["back_share"],
    )

    # Remove tiny residual net load from normalization mismatch.
    residual = np.sum(force)
    if abs(residual) > 1e-12:
        active = np.concatenate(
            [
                np.array([top_dof(node_id) for node_id in front_ids], dtype=int),
                np.array([top_dof(node_id) for node_id in back_ids], dtype=int),
                np.array([bottom_dof(node_id) for node_id in insert_node_ids], dtype=int),
            ]
        )
        force[active] -= residual / max(len(active), 1)

    displacement = spsolve(K, force)
    top_u = displacement[:n]
    bottom_u = displacement[n:]

    top_energy = np.zeros(n, dtype=float)
    bottom_energy = np.zeros(n, dtype=float)

    for ix in range(grid["nx"]):
        for iy in range(grid["ny"]):
            node_id = ids[ix, iy]
            if node_id < 0:
                continue
            in_insert = bool(insert_mask[ix, iy])
            vertical_stiffness = k_core_insert if in_insert else k_core_eps
            e_vert = 0.5 * vertical_stiffness * (top_u[node_id] - bottom_u[node_id]) ** 2
            top_energy[node_id] += 0.5 * e_vert
            bottom_energy[node_id] += 0.5 * e_vert

            if ix + 1 < grid["nx"] and inside[ix + 1, iy]:
                node_next = ids[ix + 1, iy]
                top_boost = top_skin_insert_boost if in_insert and insert_mask[ix + 1, iy] else 1.0
                bottom_boost = bottom_skin_insert_boost if in_insert and insert_mask[ix + 1, iy] else 1.0
                e_top = 0.5 * k_skin_x * top_boost * (top_u[node_id] - top_u[node_next]) ** 2
                e_bottom = 0.5 * k_skin_x * bottom_boost * (bottom_u[node_id] - bottom_u[node_next]) ** 2
                top_energy[node_id] += 0.5 * e_top
                top_energy[node_next] += 0.5 * e_top
                bottom_energy[node_id] += 0.5 * e_bottom
                bottom_energy[node_next] += 0.5 * e_bottom

            if iy + 1 < grid["ny"] and inside[ix, iy + 1]:
                node_next = ids[ix, iy + 1]
                top_boost = top_skin_insert_boost if in_insert and insert_mask[ix, iy + 1] else 1.0
                bottom_boost = bottom_skin_insert_boost if in_insert and insert_mask[ix, iy + 1] else 1.0
                e_top = 0.5 * k_skin_y * top_boost * (top_u[node_id] - top_u[node_next]) ** 2
                e_bottom = 0.5 * k_skin_y * bottom_boost * (bottom_u[node_id] - bottom_u[node_next]) ** 2
                top_energy[node_id] += 0.5 * e_top
                top_energy[node_next] += 0.5 * e_top
                bottom_energy[node_id] += 0.5 * e_bottom
                bottom_energy[node_next] += 0.5 * e_bottom

    top_grid = np.zeros((grid["nx"], grid["ny"]), dtype=float)
    bottom_grid = np.zeros((grid["nx"], grid["ny"]), dtype=float)
    top_grid[inside] = top_energy
    bottom_grid[inside] = bottom_energy

    field = np.zeros(len(surface["vertices"]), dtype=float)
    for idx, vertex in enumerate(surface["vertices"]):
        vx, vy, _ = vertex
        top_val = interpolate_grid(x, y, top_grid, vx, vy)
        bottom_val = interpolate_grid(x, y, bottom_grid, vx, vy)
        topness = surface["meta"][idx]["topness"]
        bottomness = surface["meta"][idx]["bottomness"]
        railness = surface["meta"][idx]["railness"]
        field[idx] = topness * top_val + bottomness * bottom_val + railness * 0.5 * (top_val + bottom_val)

    return field


def distance_to_rectangle(x: float, y: float, cx: float, cy: float, lx: float, ly: float) -> tuple[float, float, float]:
    dx = max(abs(x - cx) - lx * 0.5, 0.0)
    dy = max(abs(y - cy) - ly * 0.5, 0.0)
    return math.sqrt(dx * dx + dy * dy), dx, dy


def distance_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.sqrt((px - qx) ** 2 + (py - qy) ** 2)


def gaussian(value: float, sigma: float) -> float:
    return math.exp(-0.5 * (value / max(sigma, 1e-6)) ** 2)


def build_field(vertices: np.ndarray, meta: list[dict], board: dict, variant: str, load_case: str) -> np.ndarray:
    mast_x = float(board["mast_x"])
    mast_y = float(board["width"]) * 0.5
    front_foot_x = float(board["front_foot_x"])
    back_foot_x = float(board["back_foot_x"])

    if variant == "tuttle":
        patch_x = 0.1905
        patch_y = 0.0508
        source_top = 1.0
        source_bottom = 1.0
        through_skin = 0.55
        spread_x = 0.15
        spread_y = 0.06
    else:
        patch_x = 0.5080
        patch_y = 0.1905
        source_top = 0.0
        source_bottom = 1.0
        through_skin = 0.24
        spread_x = 0.22
        spread_y = 0.11

    field = np.zeros(len(vertices), dtype=float)

    for idx, vertex in enumerate(vertices):
        x, y, _ = vertex
        topness = meta[idx]["topness"]
        bottomness = meta[idx]["bottomness"]
        railness = meta[idx]["railness"]

        d_rect, dx_rect, dy_rect = distance_to_rectangle(x, y, mast_x, mast_y, patch_x, patch_y)
        d_local = math.sqrt((dx_rect / spread_x) ** 2 + (dy_rect / spread_y) ** 2)
        local_core = gaussian(d_local, 1.0)

        direct_top = source_top * topness * local_core
        direct_bottom = source_bottom * bottomness * local_core
        coupled_top = source_bottom * through_skin * topness * gaussian(d_local, 1.25)
        coupled_bottom = source_top * through_skin * bottomness * gaussian(d_local, 1.25)

        front_path = gaussian(distance_to_segment(x, y, front_foot_x, mast_y, mast_x, mast_y), 0.048)
        back_path = gaussian(distance_to_segment(x, y, back_foot_x, mast_y, mast_x, mast_y), 0.052)
        deck_bridge = topness * (0.55 * front_path + 0.85 * back_path)
        hull_bridge = bottomness * gaussian(abs(x - mast_x), 0.19) * gaussian(abs(y - mast_y), 0.11)

        fore_aft = abs(x - mast_x)
        side = abs(y - mast_y)

        if load_case == "vertical":
            value = (
                1.05 * direct_top
                + 1.15 * direct_bottom
                + 0.9 * coupled_top
                + 0.7 * coupled_bottom
                + 0.9 * deck_bridge
                + 0.45 * hull_bridge
            )
        elif load_case == "pitch":
            pitch_edge = 0.45 + 0.9 * gaussian(fore_aft, 0.15)
            moment_band = 0.55 + 1.15 * min(fore_aft / 0.16, 1.0)
            value = (
                (1.0 * direct_top + 1.25 * direct_bottom + 0.95 * coupled_top) * pitch_edge
                + deck_bridge * (0.45 + 0.7 * gaussian(x - back_foot_x, 0.18))
                + hull_bridge * moment_band
            )
        elif load_case == "lateral":
            rail_amp = 0.55 + 1.45 * railness + 0.6 * min(side / 0.18, 1.0)
            value = (
                (0.85 * direct_top + 1.0 * direct_bottom + 0.8 * coupled_top) * rail_amp
                + 0.55 * deck_bridge
                + 0.7 * hull_bridge * rail_amp
            )
        elif load_case == "impulse":
            pitch_part = (
                (1.1 * direct_top + 1.3 * direct_bottom + 1.0 * coupled_top) * (0.65 + min(fore_aft / 0.18, 1.0))
            )
            value = 1.35 * (
                0.9 * direct_top
                + 1.1 * direct_bottom
                + 0.85 * coupled_top
                + 0.8 * deck_bridge
                + 0.55 * hull_bridge
            ) + 0.55 * pitch_part
        else:
            value = direct_top + direct_bottom + coupled_top + deck_bridge

        # Local singularity near the insert/bond perimeter to create realistic ring intensity.
        edge_boost = 0.22 + 0.95 * gaussian(d_rect, 0.028)
        if variant == "track":
            edge_boost *= 1.08 + 0.32 * bottomness
        else:
            edge_boost *= 1.0 + 0.18 * topness

        field[idx] = value * edge_boost

    field -= field.min()
    max_value = field.max()
    if max_value > 0:
        field /= max_value
    field = np.power(field, 0.82)
    return field


def build_payload() -> dict:
    board = parse_board_shape()
    surface = build_surface(board)
    grid = build_planform_grid(board)
    vertices = surface["vertices"]
    thickness_sample_x = np.linspace(
        max(float(np.min(board["bottom_x"])), float(np.min(board["deck_x"]))),
        min(float(np.max(board["bottom_x"])), float(np.max(board["deck_x"]))),
        512,
    )
    thickness_max = float(
        np.max(
            np.interp(thickness_sample_x, board["deck_x"], board["deck_z"])
            - np.interp(thickness_sample_x, board["bottom_x"], board["bottom_z"])
        )
    )

    raw_fields = {
        variant: {
            load_case: solve_layered_plate(board, surface, variant, load_case)
            for load_case in ("vertical", "pitch", "lateral", "impulse")
        }
        for variant in ("tuttle", "track")
    }

    fields: dict[str, dict[str, list[float]]] = {variant: {} for variant in ("tuttle", "track")}
    for load_case in ("vertical", "pitch", "lateral", "impulse"):
        peak = max(
            float(np.max(raw_fields["tuttle"][load_case])),
            float(np.max(raw_fields["track"][load_case])),
            1e-12,
        )
        for variant in ("tuttle", "track"):
            normalized = np.clip(raw_fields[variant][load_case] / peak, 0.0, None)
            normalized = np.power(normalized, 0.85)
            fields[variant][load_case] = np.round(normalized, 5).tolist()

    payload = {
        "board": {
            "source": board["s3dx_path"],
            "length": round(float(board["length"]), 5),
            "width": round(float(board["width"]), 5),
            "thickness_max": round(thickness_max, 5),
            "mast_x": round(float(board["mast_x"]), 5),
            "mast_y": round(float(board["width"] * 0.5), 5),
            "front_foot_x": round(float(board["front_foot_x"]), 5),
            "back_foot_x": round(float(board["back_foot_x"]), 5),
            "front_foot_y": round(float(board["width"] * 0.5), 5),
            "back_foot_y": round(float(board["width"] * 0.5), 5),
            "vertices": np.round(vertices, 5).tolist(),
            "triangles": surface["triangles"],
            "topness": [round(v["topness"], 5) for v in surface["meta"]],
            "bottomness": [round(v["bottomness"], 5) for v in surface["meta"]],
        },
        "model": {
            "grid": {
                "nx": grid["nx"],
                "ny": grid["ny"],
                "x": np.round(grid["x"], 5).tolist(),
                "y": np.round(grid["y"], 5).tolist(),
                "inside": grid["inside"].astype(int).ravel().tolist(),
            },
            "shell_thickness": 0.004,
            "defaults": {
                "tuttle": {
                    "size": {"x": 0.1905, "y": 0.0508, "z": 0.12065},
                    "shape": "rect",
                },
                "track": {
                    "size": {"x": 0.5080, "y": 0.1905, "z": 0.04064},
                    "shape": "rect",
                },
            },
        },
        "insert_variants": {
            "tuttle": {
                "size": {"x": 0.1905, "y": 0.0508, "z": 0.12065},
                "shape": "rect",
                "bond_faces": ["top", "bottom"],
            },
            "track": {
                "size": {"x": 0.5080, "y": 0.1905, "z": 0.04064},
                "shape": "rect",
                "bond_faces": ["bottom"],
            },
        },
        "load_cases": {
            "vertical": {
                "label": "Pure Vertical Transfer",
                "copy": "Vertical mast transfer into the shell with balanced stance reaction.",
                "mast_force": [0.0, 0.0, 1.0],
            },
            "pitch": {
                "label": "Vertical Plus Pitch Moment",
                "copy": "Fore-aft bending couple around the mast that should favor a through-thickness insert path.",
                "mast_force": [0.0, 0.0, 1.0],
            },
            "lateral": {
                "label": "Vertical Plus Lateral Carve Load",
                "copy": "Asymmetric shell demand with rail loading and roll transfer.",
                "mast_force": [0.0, 0.48, 0.82],
            },
            "impulse": {
                "label": "Pump Impulse Envelope",
                "copy": "Higher transient envelope of the pitch case.",
                "mast_force": [0.0, 0.0, 1.55],
            },
        },
        "fields": fields,
    }
    return payload


def main() -> None:
    payload = build_payload()
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
