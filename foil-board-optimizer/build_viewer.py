"""Build an interactive HTML viewer from optimization results.

Usage:
    python build_viewer.py results/modal_tt60_shell [--s3dx path/to/board.s3dx]

Expects density.bin, meta.json in the output directory.
Produces viewer.html in the project directory.
"""

import base64, json, sys, os
import xml.etree.ElementTree as ET
import numpy as np


def _eval_bezier_2d(cp, t1, t2, dim_a, dim_b, n_per_seg=80):
    """Evaluate cubic Bezier segments from Shape3d data in 2 dimensions."""
    all_pts = []
    for i in range(len(cp) - 1):
        p0 = np.array([cp[i][dim_a], cp[i][dim_b]])
        p3 = np.array([cp[i+1][dim_a], cp[i+1][dim_b]])
        p1 = np.array([t2[i][dim_a], t2[i][dim_b]])  # outgoing handle
        p2 = np.array([t1[i+1][dim_a], t1[i+1][dim_b]])  # incoming handle
        t = np.linspace(0, 1, n_per_seg)
        seg = (((1-t)**3)[:, None] * p0 + (3*(1-t)**2*t)[:, None] * p1 +
               (3*(1-t)*t**2)[:, None] * p2 + (t**3)[:, None] * p3)
        if i > 0:
            seg = seg[1:]
        all_pts.append(seg)
    return np.vstack(all_pts)


def _read_bezier_points(element):
    """Extract control points and tangent handles from a Shape3d Bezier3d element."""
    def read_poly(name):
        poly = element.find(name).find('Polygone3d')
        pts = []
        for pt in poly.findall('Point3d'):
            x = float(pt.find('x').text)
            y = float(pt.find('y').text)
            z_el = pt.find('z')
            z = float(z_el.text) if z_el is not None else 0.0
            pts.append((x, y, z))
        return pts
    return read_poly('Control_points'), read_poly('Tangents_1'), read_poly('Tangents_2')


def parse_s3dx(s3dx_path: str) -> dict:
    """Parse a Shape3d .s3dx file and return 3D board surface data.

    Returns dict with:
        outline_x, outline_hw: planform outline (X in m, half-width in m)
        bot_x, bot_z: bottom rocker (X, Z in m)
        deck_x, deck_z: deck profile (X, Z in m)
        board_verts, board_tris: 3D surface mesh (vertices and triangle indices)
        planform_x, planform_y: closed planform outline for clipping
    """
    tree = ET.parse(s3dx_path)
    board = tree.getroot().find('Board')

    # Planform outline (Otl): X vs half-width
    otl = board.find('Otl').find('Bezier3d')
    otl_cp, otl_t1, otl_t2 = _read_bezier_points(otl)
    otl_curve = _eval_bezier_2d(otl_cp, otl_t1, otl_t2, 0, 1, 80)  # X, Y in cm

    # Bottom rocker (StrBot): X vs Z
    bot = board.find('StrBot').find('Bezier3d')
    bot_cp, bot_t1, bot_t2 = _read_bezier_points(bot)
    bot_curve = _eval_bezier_2d(bot_cp, bot_t1, bot_t2, 0, 2, 80)  # X, Z in cm

    # Deck profile (StrDeck): X vs Z
    deck = board.find('StrDeck').find('Bezier3d')
    deck_cp, deck_t1, deck_t2 = _read_bezier_points(deck)
    deck_curve = _eval_bezier_2d(deck_cp, deck_t1, deck_t2, 0, 2, 80)  # X, Z in cm

    # Convert to meters
    board_length = float(board.find('Length').text) / 100.0
    board_width = float(board.find('Width').text) / 100.0
    board_thickness = float(board.find('Thickness').text) / 100.0

    # Flip X: Shape3d X=0 is tail, optimizer X=0 is nose
    otl_x_m = board_length - otl_curve[::-1, 0] / 100.0
    otl_hw_m = otl_curve[::-1, 1] / 100.0
    bot_x_m = board_length - bot_curve[::-1, 0] / 100.0
    bot_z_m = bot_curve[::-1, 1] / 100.0
    deck_x_m = board_length - deck_curve[::-1, 0] / 100.0
    deck_z_m = deck_curve[::-1, 1] / 100.0

    # Generate 3D surface mesh by lofting cross-sections
    n_stations = 80
    n_ring = 24  # points around each cross-section
    x_stations = np.linspace(otl_x_m[1], otl_x_m[-2], n_stations)

    verts = []
    tris = []

    # Z normalization: map Shape3d rocker/deck coords to optimizer's flat 0-lz box
    # The optimizer treats Z=0 as bottom, Z=lz as top (flat box)
    # Shape3d has curved bottom (rocker) and curved deck
    # We offset so the thickest cross-section centers on lz/2
    z_bot_all = np.interp(x_stations, bot_x_m, bot_z_m)
    z_deck_all = np.interp(x_stations, deck_x_m, deck_z_m)

    for si, xs in enumerate(x_stations):
        hw = float(np.interp(xs, otl_x_m, otl_hw_m))
        z_bot = z_bot_all[si]
        z_deck = z_deck_all[si]

        if hw < 0.001:
            hw = 0.001

        center_y = board_width / 2.0
        z_mid = (z_bot + z_deck) / 2.0
        z_half = (z_deck - z_bot) / 2.0

        for ri in range(n_ring):
            angle = 2 * np.pi * ri / n_ring
            y = center_y + hw * np.cos(angle)
            z = z_mid + z_half * np.sin(angle)
            verts.append([xs, y, z])

        if si > 0:
            base = si * n_ring
            prev = (si - 1) * n_ring
            for ri in range(n_ring):
                rn = (ri + 1) % n_ring
                tris.append([prev + ri, base + ri, base + rn])
                tris.append([prev + ri, base + rn, prev + rn])

    # Planform outline for clipping
    center_y = board_width / 2.0
    planform_x = np.concatenate([otl_x_m, otl_x_m[::-1]])
    planform_y = np.concatenate([center_y + otl_hw_m, center_y - otl_hw_m[::-1]])

    # Rocker/deck profiles for Z-clipping (sampled at same X stations as outline)
    # Interpolate onto the outline X stations for consistent lookup
    bot_z_at_outline = np.interp(otl_x_m, bot_x_m, bot_z_m)
    deck_z_at_outline = np.interp(otl_x_m, deck_x_m, deck_z_m)

    return {
        "outline_x": otl_x_m.tolist(),
        "outline_hw": otl_hw_m.tolist(),
        "outline_bot_z": [round(float(v), 5) for v in bot_z_at_outline],
        "outline_deck_z": [round(float(v), 5) for v in deck_z_at_outline],
        "board_length": board_length,
        "board_width": board_width,
        "board_verts": [[round(v, 5) for v in vert] for vert in verts],
        "board_tris": tris,
        "planform_x": [round(float(v), 5) for v in planform_x],
        "planform_y": [round(float(v), 5) for v in planform_y],
    }


def build(result_dir: str, output_path: str = "viewer.html", s3dx_path: str = None):
    with open(os.path.join(result_dir, "meta.json")) as f:
        meta = json.load(f)

    density_raw = np.fromfile(os.path.join(result_dir, "density.bin"), dtype=np.float32)
    density_b64 = base64.b64encode(density_raw.tobytes()).decode()

    # Load per-load-case strain energy data if available
    se_dir = os.path.join(result_dir, "strain_energy")
    strain_energy_data = {}
    if os.path.isdir(se_dir):
        for lc_info in meta.get("load_cases", []):
            name = lc_info["name"]
            se_path = os.path.join(se_dir, f"{name}.bin")
            if os.path.exists(se_path):
                se = np.fromfile(se_path, dtype=np.float32)
                strain_energy_data[name] = base64.b64encode(se.tobytes()).decode()

    se_json = json.dumps(strain_energy_data) if strain_energy_data else "{}"

    # Load safety factor data if available
    sf_path = os.path.join(result_dir, "safety_factor.bin")
    sf_b64 = ""
    if os.path.exists(sf_path):
        sf_raw = np.fromfile(sf_path, dtype=np.float32)
        sf_b64 = base64.b64encode(sf_raw.tobytes()).decode()

    vm_path = os.path.join(result_dir, "vm_max.bin")
    vm_b64 = ""
    if os.path.exists(vm_path):
        vm_raw = np.fromfile(vm_path, dtype=np.float32)
        vm_b64 = base64.b64encode(vm_raw.tobytes()).decode()

    # Parse board shape from .s3dx file
    if s3dx_path and os.path.exists(s3dx_path):
        board_shape = parse_s3dx(s3dx_path)
    else:
        # Try default location
        default_s3dx = os.path.expanduser("~/Downloads/TT60 - Ken Adgate Inspired.s3dx")
        if os.path.exists(default_s3dx):
            board_shape = parse_s3dx(default_s3dx)
        else:
            board_shape = None
    board_shape_json = json.dumps(board_shape) if board_shape else "null"

    meta_json = json.dumps(meta)
    outline_json = board_shape_json

    html = f'''<!DOCTYPE html>
<html>
<head>
<title>Foil Board Topology Optimization</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0a0a1a; color: #ddd; font-family: system-ui, -apple-system, sans-serif; overflow: hidden; }}
  canvas {{ display: block; }}
  #panel {{
    position: absolute; top: 12px; left: 12px; z-index: 10;
    background: rgba(10,10,30,0.85); backdrop-filter: blur(10px);
    padding: 16px; border-radius: 10px; width: 280px;
    border: 1px solid rgba(255,255,255,0.08);
  }}
  #panel h2 {{ font-size: 15px; margin-bottom: 10px; color: #fff; }}
  .control {{ margin: 8px 0; }}
  .control label {{ font-size: 12px; display: block; margin-bottom: 3px; opacity: 0.7; }}
  .control input[type=range] {{ width: 100%; accent-color: #e85d04; }}
  .stat {{ font-size: 12px; opacity: 0.6; margin: 2px 0; }}
  .legend {{ display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 11px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 2px; }}
  #hint {{ position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
    font-size: 12px; opacity: 0.3; }}
  .btn {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
    color: #ddd; padding: 5px 10px; border-radius: 5px; font-size: 11px; cursor: pointer; margin: 2px; }}
  .btn:hover {{ background: rgba(255,255,255,0.15); }}
  .btn.active {{ background: rgba(232,93,4,0.3); border-color: #e85d04; }}
  .btn-row {{ display: flex; flex-wrap: wrap; gap: 2px; margin: 6px 0; }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none; width: 16px; height: 16px;
    background: #fff; border-radius: 50%; border: 2px solid #e85d04;
    cursor: pointer; box-shadow: 0 1px 4px rgba(0,0,0,0.4);
  }}
</style>
</head>
<body>

<div id="panel">
  <h2>Foil Board Internal Structure</h2>
  <div class="stat" id="stats"></div>

  <div class="control">
    <label>Load intensity: <span id="thresh-val" style="font-weight:bold">0.30</span></label>
    <div style="position:relative;height:22px;margin:4px 0">
      <div style="position:absolute;top:6px;left:0;right:0;height:10px;border-radius:5px;
        background:linear-gradient(to right, #ddeeff 0%, #ffd166 25%, #e85d04 55%, #c1121f 85%, #6b0f1a 100%);
        opacity:0.7"></div>
      <input type="range" id="threshold" min="0.02" max="0.95" step="0.01" value="0.30"
        style="position:absolute;top:0;width:100%;-webkit-appearance:none;background:transparent;height:22px">
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;opacity:0.4;margin-top:2px">
      <span>foam</span><span>reinforcement</span><span>primary</span>
    </div>
  </div>

  <div class="control">
    <label>X-axis clip (length): <span id="clip-val">100%</span></label>
    <input type="range" id="clipX" min="0" max="100" step="1" value="100">
  </div>

  <div class="control">
    <label>Z-axis clip (thickness): <span id="clipZ-val">100%</span></label>
    <input type="range" id="clipZ" min="0" max="100" step="1" value="100">
  </div>

  <div class="control">
    <label>Color mode:</label>
    <div class="btn-row">
      <button class="btn active" id="btn-density">Density</button>
      <button class="btn" id="btn-lc-riding">Riding</button>
      <button class="btn" id="btn-lc-pumping">Pumping</button>
      <button class="btn" id="btn-lc-landing">Landing</button>
      <button class="btn" id="btn-lc-carving">Carving</button>
      <button class="btn" id="btn-strength">Strength</button>
    </div>
  </div>

  <div id="strength-legend" style="display:none;margin-top:8px">
    <div style="font-size:11px;opacity:0.6;margin-bottom:4px">Safety factor (yield margin)</div>
    <div style="height:8px;border-radius:4px;background:linear-gradient(to right,#c1121f,#e85d04,#ffd166,#40c057);margin-bottom:2px"></div>
    <div style="display:flex;justify-content:space-between;font-size:10px;opacity:0.4"><span>&lt;2 critical</span><span>4</span><span>6</span><span>&gt;8 safe</span></div>
  </div>

  <div class="btn-row">
    <button class="btn active" id="btn-foam">Foam core</button>
    <button class="btn active" id="btn-loads">Loads</button>
    <button class="btn active" id="btn-board">Board shape</button>
    <button class="btn" id="btn-xray">X-ray</button>
  </div>

  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#6b0f1a"></div>Primary structure</div>
    <div class="legend-item"><div class="legend-dot" style="background:#e85d04"></div>Secondary</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ffd166"></div>Reinforcement</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ddeeff;opacity:0.4"></div>Foam core</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ff3366"></div>Front foot</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ff6633"></div>Back foot</div>
    <div class="legend-item"><div class="legend-dot" style="background:#33ff88"></div>Mast mount</div>
  </div>
</div>

<div id="hint">Drag to rotate &middot; Scroll to zoom &middot; Right-drag to pan</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const meta = {meta_json};
const {{ nelx, nely, nelz, lx, ly, lz }} = meta;

// Decode density
const b64 = "{density_b64}";
const raw = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
const density = new Float32Array(raw.buffer);

// Decode safety factor and max von Mises
const sfB64 = "{sf_b64}";
const vmB64 = "{vm_b64}";
const safetyFactor = sfB64 ? new Float32Array(Uint8Array.from(atob(sfB64), c => c.charCodeAt(0)).buffer) : null;
const vmMax = vmB64 ? new Float32Array(Uint8Array.from(atob(vmB64), c => c.charCodeAt(0)).buffer) : null;

// Decode per-load-case strain energy
const seB64 = {se_json};
const strainEnergy = {{}};
for (const [name, b] of Object.entries(seB64)) {{
  const raw = Uint8Array.from(atob(b), c => c.charCodeAt(0));
  strainEnergy[name] = new Float32Array(raw.buffer);
}}
const lcNames = {{ 'riding_normal': 'riding', 'pumping': 'pumping',
                   'jump_landing': 'landing', 'carving': 'carving' }};
let activeColorMode = 'density'; // 'density' or a load case name

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a1a);

const camera = new THREE.PerspectiveCamera(40, innerWidth/innerHeight, 0.01, 100);
camera.position.set(2.0, 1.2, 1.0);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(3, 4, 2);
sun.castShadow = true;
scene.add(sun);
const fill = new THREE.DirectionalLight(0x6688cc, 0.3);
fill.position.set(-2, -1, 3);
scene.add(fill);

// Groups
const structureGroup = new THREE.Group();
const loadsGroup = new THREE.Group();
const boardGroup = new THREE.Group();
scene.add(structureGroup);
scene.add(loadsGroup);
scene.add(boardGroup);

// Center offset (so board is centered at origin)
const cx = lx/2, cy = ly/2, cz = lz/2;

// === BOARD SHAPE (parsed from Shape3d .s3dx) ===
const boardData = {outline_json};

// No JS-side clipping — the optimizer handles board shape via void masking

if (boardData) {{
  // Build 3D surface mesh from lofted cross-sections
  const bv = boardData.board_verts;
  const bt = boardData.board_tris;

  // Find Z center of the Shape3d mesh (it has rocker, so Z range differs from optimizer)
  let zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < bv.length; i++) {{
    if (bv[i][2] < zMin) zMin = bv[i][2];
    if (bv[i][2] > zMax) zMax = bv[i][2];
  }}
  const boardZCenter = (zMin + zMax) / 2;

  const positions = new Float32Array(bv.length * 3);
  for (let i = 0; i < bv.length; i++) {{
    positions[i*3]   = bv[i][0] - cx;
    positions[i*3+1] = bv[i][1] - cy;
    // Center board mesh Z to align with voxels (both centered at 0)
    positions[i*3+2] = bv[i][2] - boardZCenter;
  }}

  const indices = new Uint32Array(bt.length * 3);
  for (let i = 0; i < bt.length; i++) {{
    indices[i*3]   = bt[i][0];
    indices[i*3+1] = bt[i][1];
    indices[i*3+2] = bt[i][2];
  }}

  const surfGeo = new THREE.BufferGeometry();
  surfGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  surfGeo.setIndex(new THREE.BufferAttribute(indices, 1));
  surfGeo.computeVertexNormals();

  const surfMesh = new THREE.Mesh(surfGeo, new THREE.MeshPhysicalMaterial({{
    color: 0x4488ff, opacity: 0.12, transparent: true, side: THREE.DoubleSide,
    depthWrite: false, roughness: 0.8, metalness: 0.05,
  }}));
  boardGroup.add(surfMesh);

  // Wireframe overlay for shape definition
  const wireGeo = new THREE.WireframeGeometry(surfGeo);
  const wireMesh = new THREE.LineSegments(wireGeo, new THREE.LineBasicMaterial({{
    color: 0x4488ff, opacity: 0.08, transparent: true,
  }}));
  boardGroup.add(wireMesh);

  // Planform outline on deck and hull
  const pX = boardData.planform_x;
  const pY = boardData.planform_y;
  const pN = pX.length;
  const outlineMat = new THREE.LineBasicMaterial({{ color: 0x4488ff, opacity: 0.7, transparent: true }});
  function makeOutlineLoop(zOff) {{
    const geo = new THREE.BufferGeometry();
    const v = new Float32Array(pN * 3);
    for (let i = 0; i < pN; i++) {{
      v[i*3] = pX[i] - cx;
      v[i*3+1] = pY[i] - cy;
      v[i*3+2] = zOff;
    }}
    geo.setAttribute('position', new THREE.BufferAttribute(v, 3));
    return new THREE.LineLoop(geo, outlineMat);
  }}
  boardGroup.add(makeOutlineLoop(lz/2));
  boardGroup.add(makeOutlineLoop(-lz/2));
}}

// === LOAD VISUALIZATION ===
function addZone(bounds, color, y, label) {{
  const [xmin, xmax, ymin, ymax] = bounds;
  const w = xmax - xmin, h = ymax - ymin;
  const geo = new THREE.PlaneGeometry(w, h);
  const mat = new THREE.MeshBasicMaterial({{
    color, opacity: 0.3, transparent: true, side: THREE.DoubleSide
  }});
  const plane = new THREE.Mesh(geo, mat);
  plane.position.set((xmin+xmax)/2 - cx, (ymin+ymax)/2 - cy, y - cz);
  loadsGroup.add(plane);

  // Border
  const edge = new THREE.LineSegments(
    new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({{ color, linewidth: 2 }})
  );
  edge.position.copy(plane.position);
  loadsGroup.add(edge);
}}

// Front and back foot pads on deck (top face)
const ffb = meta.front_foot_bounds || meta.foot_bounds;
const bfb = meta.back_foot_bounds || meta.foot_bounds;
addZone(ffb, 0xff3366, lz, 'Front foot');
addZone(bfb, 0xff6633, lz, 'Back foot');

// Mast mount on bottom
addZone(meta.mast_bounds, 0x33ff88, 0, 'Mast mount');

// Mast mount on deck (where mast forces are applied)
addZone(meta.mast_bounds, 0xff33ff, lz, 'Mast force zone');

// Force arrows
function addArrow(origin, dir, length, color) {{
  const d = new THREE.Vector3(...dir).normalize();
  const arrow = new THREE.ArrowHelper(d, new THREE.Vector3(...origin), length, color, length*0.25, length*0.12);
  loadsGroup.add(arrow);
}}

// Show forces for each load case (just riding_normal as representative)
const lc = meta.load_cases[0];
const mf = lc.mast_force;
const mb = meta.mast_bounds;
const mastCenter = [(mb[0]+mb[1])/2 - cx, (mb[2]+mb[3])/2 - cy, lz - cz];

// Mast force arrow (scale for visibility)
const fMag = Math.sqrt(mf[0]**2 + mf[1]**2 + mf[2]**2);
addArrow(mastCenter, mf, 0.15, 0xff33ff);

// Deck load arrows at each foot pad
function addFootArrows(bounds, color) {{
  for (let xi = 0; xi < 2; xi++) {{
    const x = bounds[0] + (bounds[1]-bounds[0]) * (xi+0.5)/2 - cx;
    const y = (bounds[2]+bounds[3])/2 - cy;
    addArrow([x, y, lz - cz + 0.06], [0,0,-1], 0.06, color);
  }}
}}
addFootArrows(ffb, 0xff3366);  // front foot
addFootArrows(bfb, 0xff6633);  // back foot

// Fixed BC indicators (small cubes at mast mount bottom)
for (let xi = 0; xi < 3; xi++) {{
  for (let yi = 0; yi < 2; yi++) {{
    const x = mb[0] + (mb[1]-mb[0]) * (xi+0.5)/3 - cx;
    const y = mb[2] + (mb[3]-mb[2]) * (yi+0.5)/2 - cy;
    const cube = new THREE.Mesh(
      new THREE.BoxGeometry(0.015, 0.015, 0.015),
      new THREE.MeshBasicMaterial({{ color: 0x33ff88 }})
    );
    cube.position.set(x, y, -cz);
    loadsGroup.add(cube);
  }}
}}

// === VOXEL STRUCTURE ===
const dx = lx/nelx, dy = ly/nely, dz_el = lz/nelz;

// Instanced mesh for voxels
const voxelGeo = new THREE.BoxGeometry(dx*0.95, dy*0.95, dz_el*0.95);
const voxelMat = new THREE.MeshPhysicalMaterial({{
  vertexColors: true, metalness: 0.05, roughness: 0.6, clearcoat: 0.2,
}});

let instancedMesh = null;
let foamMesh = null;
let xrayMode = false;
let showFoam = true;

// Foam core material (lightweight filler)
const foamGeo = new THREE.BoxGeometry(dx*0.95, dy*0.95, dz_el*0.95);
const foamMat = new THREE.MeshPhysicalMaterial({{
  color: 0xddeeff, opacity: 0.08, transparent: true,
  depthWrite: false, roughness: 0.95, metalness: 0.0,
}});
const foamGroup = new THREE.Group();
scene.add(foamGroup);

function buildStructure(threshold, clipXPct, clipZPct) {{
  // Remove old
  if (instancedMesh) structureGroup.remove(instancedMesh);
  if (foamMesh) foamGroup.remove(foamMesh);

  const clipX = (clipXPct / 100) * nelx;
  const clipZ = (clipZPct / 100) * nelz;

  // Count structural voxels (above threshold) and foam voxels (inside board, below threshold)
  let count = 0, foamCount = 0;
  const VOID_THRESH = 0.01; // below this = outside board (void)
  for (let k = 0; k < nelz; k++) {{
    for (let j = 0; j < nely; j++) {{
      for (let i = 0; i < nelx; i++) {{
        const idx = k * nely * nelx + j * nelx + i;
        const d = density[idx];
        if (i >= clipX || k >= clipZ) continue;
        if (d >= threshold) count++;
        else if (d >= VOID_THRESH) foamCount++;
      }}
    }}
  }}

  // Build structural mesh
  if (count > 0) {{
    const iMesh = new THREE.InstancedMesh(voxelGeo, voxelMat, count);
    iMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const colorAttr = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    const matrix = new THREE.Matrix4();
    let vi = 0;

    for (let k = 0; k < nelz; k++) {{
      for (let j = 0; j < nely; j++) {{
        for (let i = 0; i < nelx; i++) {{
          const idx = k * nely * nelx + j * nelx + i;
          const d = density[idx];
          if (d < threshold || i >= clipX || k >= clipZ) continue;

          matrix.makeTranslation((i+0.5)*dx - cx, (j+0.5)*dy - cy, (k+0.5)*dz_el - cz);
          iMesh.setMatrixAt(vi, matrix);

          if (activeColorMode === 'density') {{
            const t = Math.max(0, Math.min(1, (d - threshold) / (1 - threshold)));
            colorAttr.setXYZ(vi, 0.9 + 0.1*t, 0.82 - 0.6*t, 0.39 - 0.35*t);
          }} else if (activeColorMode === 'strength') {{
            // Safety factor: red=critical(<2) → orange → yellow → green=safe(>8)
            const sf = safetyFactor ? safetyFactor[idx] : 10;
            const t = Math.max(0, Math.min(1, (sf - 1.0) / 7.0));  // 0 at SF=1, 1 at SF=8
            if (t < 0.33) {{
              colorAttr.setXYZ(vi, 0.76, 0.07 + 0.7*t*3, 0.11);
            }} else if (t < 0.67) {{
              const t2 = (t - 0.33) * 3;
              colorAttr.setXYZ(vi, 0.76 + 0.14*t2, 0.28 + 0.52*t2, 0.11 + 0.07*t2);
            }} else {{
              const t2 = (t - 0.67) * 3;
              colorAttr.setXYZ(vi, 0.9 - 0.65*t2, 0.8 + 0.15*t2, 0.18 - 0.12*t2);
            }}
          }} else {{
            // Strain energy heat map: blue → cyan → green → yellow → red
            const seArr = strainEnergy[activeColorMode];
            const se = seArr ? seArr[idx] : 0;
            const seMax = seArr ? seArr._max || 1 : 1;
            const s = Math.max(0, Math.min(1, Math.sqrt(se / seMax)));
            if (s < 0.25) {{
              colorAttr.setXYZ(vi, 0.1, 0.2 + 2.4*s, 0.6 + 1.6*s);
            }} else if (s < 0.5) {{
              const t2 = (s - 0.25) * 4;
              colorAttr.setXYZ(vi, 0.1 + 0.8*t2, 0.8, 1.0 - 0.6*t2);
            }} else if (s < 0.75) {{
              const t2 = (s - 0.5) * 4;
              colorAttr.setXYZ(vi, 0.9, 0.8 - 0.4*t2, 0.4 - 0.3*t2);
            }} else {{
              const t2 = (s - 0.75) * 4;
              colorAttr.setXYZ(vi, 0.9 + 0.1*t2, 0.4 - 0.35*t2, 0.1 - 0.05*t2);
            }}
          }}
          vi++;
        }}
      }}
    }}

    iMesh.geometry.setAttribute('color', colorAttr);
    iMesh.instanceMatrix.needsUpdate = true;
    iMesh.castShadow = true;
    iMesh.receiveShadow = true;
    instancedMesh = iMesh;
    structureGroup.add(iMesh);
  }}

  // Build foam core filler mesh
  if (foamCount > 0) {{
    const fMesh = new THREE.InstancedMesh(foamGeo, foamMat, foamCount);
    fMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const matrix = new THREE.Matrix4();
    let fi = 0;

    for (let k = 0; k < nelz; k++) {{
      for (let j = 0; j < nely; j++) {{
        for (let i = 0; i < nelx; i++) {{
          const idx = k * nely * nelx + j * nelx + i;
          const d = density[idx];
          if (i >= clipX || k >= clipZ) continue;
          if (d >= threshold || d < VOID_THRESH) continue;

          matrix.makeTranslation((i+0.5)*dx - cx, (j+0.5)*dy - cy, (k+0.5)*dz_el - cz);
          fMesh.setMatrixAt(fi, matrix);
          fi++;
        }}
      }}
    }}

    fMesh.instanceMatrix.needsUpdate = true;
    foamMesh = fMesh;
    foamGroup.add(fMesh);
  }}

  foamGroup.visible = showFoam;
}}

// Pre-compute strain energy max for each load case (for normalization)
for (const [name, se] of Object.entries(strainEnergy)) {{
  let mx = 0;
  for (let i = 0; i < se.length; i++) if (se[i] > mx) mx = se[i];
  se._max = mx;
}}

// Initial build
buildStructure(0.3, 100, 100);

// Stats
document.getElementById('stats').innerHTML =
  `${{nelx}}×${{nely}}×${{nelz}} mesh (${{nelx*nely*nelz}} elements)<br>`+
  `Volume: ${{(meta.volfrac*100).toFixed(1)}}% &middot; `+
  `Compliance: ${{meta.compliance.toFixed(4)}} &middot; `+
  `${{meta.iterations}} iterations`
  + (meta.sigma_yield ? ` · σ_yield ${{(meta.sigma_yield/1e6).toFixed(0)}} MPa` : '');

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(0, 0, 0);

// UI handlers
let currentThresh = 0.3, currentClipX = 100, currentClipZ = 100;

document.getElementById('threshold').addEventListener('input', e => {{
  currentThresh = parseFloat(e.target.value);
  document.getElementById('thresh-val').textContent = currentThresh.toFixed(2);
  buildStructure(currentThresh, currentClipX, currentClipZ);
}});

document.getElementById('clipX').addEventListener('input', e => {{
  currentClipX = parseInt(e.target.value);
  document.getElementById('clip-val').textContent = currentClipX + '%';
  buildStructure(currentThresh, currentClipX, currentClipZ);
}});

document.getElementById('clipZ').addEventListener('input', e => {{
  currentClipZ = parseInt(e.target.value);
  document.getElementById('clipZ-val').textContent = currentClipZ + '%';
  buildStructure(currentThresh, currentClipX, currentClipZ);
}});

document.getElementById('btn-loads').addEventListener('click', e => {{
  e.target.classList.toggle('active');
  loadsGroup.visible = e.target.classList.contains('active');
}});

document.getElementById('btn-board').addEventListener('click', e => {{
  e.target.classList.toggle('active');
  boardGroup.visible = e.target.classList.contains('active');
}});

document.getElementById('btn-foam').addEventListener('click', e => {{
  showFoam = !showFoam;
  e.target.classList.toggle('active');
  foamGroup.visible = showFoam;
}});

// Load case color mode buttons
const lcMap = {{
  'btn-density': 'density',
  'btn-lc-riding': 'riding_normal',
  'btn-lc-pumping': 'pumping',
  'btn-lc-landing': 'jump_landing',
  'btn-lc-carving': 'carving',
  'btn-strength': 'strength',
}};
for (const [btnId, lcName] of Object.entries(lcMap)) {{
  const btn = document.getElementById(btnId);
  if (!btn) continue;
  btn.addEventListener('click', () => {{
    // Deactivate all color mode buttons
    for (const id of Object.keys(lcMap)) {{
      const b = document.getElementById(id);
      if (b) b.classList.remove('active');
    }}
    btn.classList.add('active');
    activeColorMode = lcName;
    buildStructure(currentThresh, currentClipX, currentClipZ);
    document.getElementById('strength-legend').style.display =
      lcName === 'strength' ? 'block' : 'none';
  }});
}}

document.getElementById('btn-xray').addEventListener('click', e => {{
  xrayMode = !xrayMode;
  e.target.classList.toggle('active');
  if (instancedMesh) {{
    instancedMesh.material.transparent = xrayMode;
    instancedMesh.material.opacity = xrayMode ? 0.3 : 1.0;
    instancedMesh.material.depthWrite = !xrayMode;
    instancedMesh.material.needsUpdate = true;
  }}
}});

// Animate
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

addEventListener('resize', () => {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});
</script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Viewer written to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", nargs="?", default="results/modal_tt60_shell")
    parser.add_argument("--s3dx", default=None, help="Path to Shape3d .s3dx board file")
    parser.add_argument("-o", "--output", default="viewer.html")
    args = parser.parse_args()
    build(args.result_dir, args.output, args.s3dx)
