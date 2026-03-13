import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  INCH,
  PARAM_LIMITS_IN,
  SHAPE_OPTIONS,
  createDefaultParams,
  preprocessModel,
  solveNormalizedFields,
} from "./viewer-solver.js";

const state = {
  loadCase: "vertical",
  viewMode: "opaque",
  clip: { x: 1, y: 1, z: 1 },
  strainGain: 2.8,
  layers: {
    shell: true,
    insert: true,
    bonds: true,
    loads: true,
  },
  params: null,
  data: null,
  model: null,
  solveToken: 0,
};

const viewport = document.getElementById("viewport");
const loadingEl = document.getElementById("loading");
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, viewport.clientWidth / viewport.clientHeight, 0.01, 30);
camera.position.set(0.0, 1.0, 3.75);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.localClippingEnabled = true;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0.0, 0.02, 0.0);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(4.6, 72),
  new THREE.MeshBasicMaterial({ color: 0xe1d7c5, transparent: true, opacity: 0.65 }),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.22;
scene.add(floor);

const grid = new THREE.GridHelper(8, 20, 0xc0b39e, 0xdbd1c1);
grid.position.y = -0.2;
grid.material.transparent = true;
grid.material.opacity = 0.32;
scene.add(grid);

const boardVariants = {};
const initialFields = {};

function showLoading(text) {
  loadingEl.textContent = text;
  loadingEl.style.display = "block";
}

function hideLoading() {
  loadingEl.style.display = "none";
}

function lerpColorHex(stops, t) {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(scaled));
  const localT = scaled - idx;
  const color = new THREE.Color(stops[idx]).lerp(new THREE.Color(stops[idx + 1]), localT);
  return [color.r, color.g, color.b];
}

function colorizeGeometry(geometry, field) {
  const stops = [0x183a54, 0x2f8f83, 0xffd166, 0xd1495b];
  const colors = new Float32Array(field.length * 3);
  for (let i = 0; i < field.length; i += 1) {
    const boosted = Math.min(1, Math.pow(field[i] * state.strainGain, 0.72));
    const [r, g, b] = lerpColorHex(stops, boosted);
    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
}

function roundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function makeLabel(text, x) {
  const canvas = document.createElement("canvas");
  canvas.width = 420;
  canvas.height = 120;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(255,250,244,0.92)";
  ctx.strokeStyle = "rgba(23,33,39,0.14)";
  ctx.lineWidth = 4;
  roundedRect(ctx, 6, 6, 408, 108, 28);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#172127";
  ctx.font = "600 42px Georgia";
  ctx.fillText(text, 32, 68);
  ctx.fillStyle = "#5a666e";
  ctx.font = "24px Georgia";
  ctx.fillText("actual TT60 shell", 32, 98);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
  sprite.scale.set(1.0, 0.28, 1.0);
  sprite.position.set(x, 0.67, 0.0);
  scene.add(sprite);
}

function createArrow(origin, dir, color, scale = 0.26) {
  const v = new THREE.Vector3(dir[0], dir[1], dir[2]);
  const mag = v.length();
  if (mag < 1e-6) {
    return new THREE.Group();
  }
  v.normalize();
  return new THREE.ArrowHelper(v, origin, mag * scale, color, 0.06, 0.03);
}

function createFootprintShape(length, width, shapeName) {
  const hx = length * 0.5;
  const hy = width * 0.5;
  const shape = new THREE.Shape();

  if (shapeName === "oval") {
    shape.absellipse(0, 0, hx, hy, 0, Math.PI * 2, false, 0);
    return shape;
  }

  if (shapeName === "diamond") {
    shape.moveTo(0, hy);
    shape.lineTo(hx, 0);
    shape.lineTo(0, -hy);
    shape.lineTo(-hx, 0);
    shape.closePath();
    return shape;
  }

  if (shapeName === "capsule") {
    if (length >= width) {
      const r = hy;
      shape.moveTo(-hx + r, -hy);
      shape.lineTo(hx - r, -hy);
      shape.absarc(hx - r, 0, r, -Math.PI / 2, Math.PI / 2, false);
      shape.lineTo(-hx + r, hy);
      shape.absarc(-hx + r, 0, r, Math.PI / 2, (3 * Math.PI) / 2, false);
    } else {
      const r = hx;
      shape.moveTo(-hx, -hy + r);
      shape.lineTo(-hx, hy - r);
      shape.absarc(0, hy - r, r, Math.PI, 0, true);
      shape.lineTo(hx, -hy + r);
      shape.absarc(0, -hy + r, r, 0, Math.PI, true);
    }
    shape.closePath();
    return shape;
  }

  const radius = shapeName === "rounded" ? Math.min(hx, hy) * 0.28 : 0;
  if (radius > 1e-6) {
    shape.moveTo(-hx + radius, -hy);
    shape.lineTo(hx - radius, -hy);
    shape.quadraticCurveTo(hx, -hy, hx, -hy + radius);
    shape.lineTo(hx, hy - radius);
    shape.quadraticCurveTo(hx, hy, hx - radius, hy);
    shape.lineTo(-hx + radius, hy);
    shape.quadraticCurveTo(-hx, hy, -hx, hy - radius);
    shape.lineTo(-hx, -hy + radius);
    shape.quadraticCurveTo(-hx, -hy, -hx + radius, -hy);
    shape.closePath();
    return shape;
  }

  shape.moveTo(-hx, -hy);
  shape.lineTo(hx, -hy);
  shape.lineTo(hx, hy);
  shape.lineTo(-hx, hy);
  shape.closePath();
  return shape;
}

function createExtrudedFootprint(length, width, height, shapeName) {
  const outline = createFootprintShape(length, width, shapeName);
  const geometry = new THREE.ExtrudeGeometry(outline, {
    depth: Math.max(height, 0.001),
    bevelEnabled: false,
    curveSegments: 32,
  });
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(0, 0, width * 0.5);
  geometry.translate(0, 0, -width * 0.5);
  geometry.center();
  geometry.translate(0, height * 0.5, 0);
  return geometry;
}

function clampMeters(valueIn, range, extraMax = Infinity) {
  const min = range[0] * INCH;
  const max = Math.min(range[1] * INCH, extraMax);
  return Math.max(min, Math.min(max, valueIn));
}

function sanitizeParams(raw, variant, cavityHeight) {
  return {
    shape: raw.shape,
    size: {
      x: clampMeters(raw.size.x, PARAM_LIMITS_IN[variant].length),
      y: clampMeters(raw.size.y, PARAM_LIMITS_IN[variant].width),
      z: clampMeters(raw.size.z, PARAM_LIMITS_IN[variant].height, cavityHeight),
    },
  };
}

function buildVariant(name, data, board, offsetX) {
  const group = new THREE.Group();
  group.position.x = offsetX;
  group.position.y = -0.02;
  group.scale.setScalar(1.18);

  const positions = new Float32Array(board.vertices.length * 3);
  let zMin = Infinity;
  let zMax = -Infinity;
  for (const [, , z] of board.vertices) {
    zMin = Math.min(zMin, z);
    zMax = Math.max(zMax, z);
  }

  const shellThickness = data.model.shell_thickness;
  const bottomOuter = zMin;
  const topOuter = zMax;
  const bottomInner = bottomOuter + shellThickness;
  const topInner = topOuter - shellThickness;
  const structuralThickness = topOuter - bottomOuter;
  const cavityHeight = Math.max(topInner - bottomInner, 0.008);

  for (let i = 0; i < board.vertices.length; i += 1) {
    const [x, y, z] = board.vertices[i];
    positions[i * 3] = x - board.length / 2;
    positions[i * 3 + 1] = z - bottomOuter;
    positions[i * 3 + 2] = y - board.width / 2;
  }

  const indices = new Uint32Array(board.triangles.length * 3);
  for (let i = 0; i < board.triangles.length; i += 1) {
    indices[i * 3] = board.triangles[i][0];
    indices[i * 3 + 1] = board.triangles[i][1];
    indices[i * 3 + 2] = board.triangles[i][2];
  }

  const shellGeometry = new THREE.BufferGeometry();
  shellGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  shellGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
  colorizeGeometry(shellGeometry, initialFields[name][state.loadCase]);
  shellGeometry.computeVertexNormals();

  const shellMaterial = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.96,
  });
  const shellMesh = new THREE.Mesh(shellGeometry, shellMaterial);
  group.add(shellMesh);

  const wire = new THREE.LineSegments(
    new THREE.WireframeGeometry(shellGeometry),
    new THREE.LineBasicMaterial({ color: 0x182126, transparent: true, opacity: 0.05 }),
  );
  group.add(wire);

  const insertMat = new THREE.MeshBasicMaterial({ color: 0xc4683d, transparent: true, opacity: 0.9 });
  const bondMat = new THREE.MeshBasicMaterial({ color: 0xffe08a, transparent: true, opacity: 0.92 });
  const mastMat = new THREE.MeshBasicMaterial({ color: 0x14263a });
  const frontPadMat = new THREE.MeshBasicMaterial({ color: 0x1b6b72, transparent: true, opacity: 0.82 });
  const backPadMat = new THREE.MeshBasicMaterial({ color: 0xb6532f, transparent: true, opacity: 0.82 });

  const mastChord = 0.1143;
  const mastThickness = 0.018;
  const mastDepth = 0.62;
  const mastBlade = new THREE.Mesh(new THREE.BoxGeometry(mastChord, mastDepth, mastThickness), mastMat);
  mastBlade.position.set(board.mast_x - board.length / 2, -mastDepth * 0.5, 0.0);
  group.add(mastBlade);

  const mastHead = new THREE.Mesh(new THREE.BoxGeometry(mastChord * 0.95, 0.024, 0.085), mastMat);
  mastHead.position.set(board.mast_x - board.length / 2, -0.012, 0.0);
  group.add(mastHead);

  const loadGroup = new THREE.Group();
  group.add(loadGroup);

  const frontPad = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.006, 0.10), frontPadMat);
  frontPad.position.set(board.front_foot_x - board.length / 2, structuralThickness + 0.016, 0.0);
  group.add(frontPad);

  const backPad = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.006, 0.10), backPadMat);
  backPad.position.set(board.back_foot_x - board.length / 2, structuralThickness + 0.016, 0.0);
  group.add(backPad);

  group.userData = {
    shellGeometry,
    shellMaterial,
    wire,
    materials: [shellMaterial, insertMat, bondMat, mastMat, frontPadMat, backPadMat],
    layers: {
      shell: [shellMesh, wire],
      insert: [mastBlade, mastHead, frontPad, backPad],
      bonds: [],
      loads: [loadGroup],
    },
    loadGroup,
    board,
    cavityHeight,
    shellThickness,
    bottomInner: bottomInner - bottomOuter,
    topInner: topInner - bottomOuter,
    structuralThickness,
    bottomBondY: bottomInner - bottomOuter,
    topBondY: topInner - bottomOuter,
    bounds: {
      xMin: -board.length / 2,
      xMax: board.length / 2,
      yMin: 0,
      yMax: structuralThickness + 0.022,
      zMin: -board.width / 2,
      zMax: board.width / 2,
    },
    insertMat,
    bondMat,
    dynamicMeshes: [],
    variant: name,
  };

  rebuildInsertMeshes(group);
  scene.add(group);
  boardVariants[name] = group;
}

function rebuildInsertMeshes(group) {
  const variant = group.userData.variant;
  const params = sanitizeParams(state.params[variant], variant, group.userData.cavityHeight);
  state.params[variant] = params;

  for (const mesh of group.userData.dynamicMeshes) {
    group.remove(mesh);
  }
  group.userData.dynamicMeshes = [];
  group.userData.layers.bonds = [];
  group.userData.layers.insert = group.userData.layers.insert.filter((mesh) => !mesh.userData?.dynamicInsert);

  const height = params.size.z;
  const insertGeometry = createExtrudedFootprint(params.size.x, params.size.y, height, params.shape);
  const insert = new THREE.Mesh(insertGeometry, group.userData.insertMat);
  insert.position.set(
    group.userData.board.mast_x - group.userData.board.length / 2,
    group.userData.bottomBondY + height * 0.5,
    0.0,
  );
  insert.userData.dynamicInsert = true;
  group.add(insert);
  group.userData.dynamicMeshes.push(insert);
  group.userData.layers.insert.push(insert);

  const bondFaces = state.data.insert_variants[variant].bond_faces;
  for (const face of bondFaces) {
    const bondGeometry = createExtrudedFootprint(params.size.x, params.size.y, 0.004, params.shape);
    const bond = new THREE.Mesh(bondGeometry, group.userData.bondMat);
    bond.position.set(
      group.userData.board.mast_x - group.userData.board.length / 2,
      face === "top" ? group.userData.topBondY : group.userData.bottomBondY,
      0.0,
    );
    group.add(bond);
    group.userData.dynamicMeshes.push(bond);
    group.userData.layers.bonds.push(bond);
  }

  setLayerVisibility();
}

function updateLoadArrows() {
  const loadInfo = state.data.load_cases[state.loadCase];
  document.getElementById("case-title").textContent = loadInfo.label;
  document.getElementById("case-copy").textContent = loadInfo.copy;

  const tuttle = state.params.tuttle;
  const track = state.params.track;
  document.getElementById("comparison-copy").textContent =
    `Shared color scale for ${loadInfo.label.toLowerCase()}. `
    + `Tuttle ${formatDims(tuttle)}. Dual-track ${formatDims(track)}.`;

  Object.values(boardVariants).forEach((group) => {
    const loadLayer = group.userData.loadGroup;
    while (loadLayer.children.length) {
      loadLayer.remove(loadLayer.children[0]);
    }

    const board = group.userData.board;
    const mastOrigin = new THREE.Vector3(board.mast_x - board.length / 2, -0.055, 0.0);
    loadLayer.add(createArrow(mastOrigin, [loadInfo.mast_force[0], loadInfo.mast_force[2], loadInfo.mast_force[1]], 0x14263a));

    const frontOrigin = new THREE.Vector3(board.front_foot_x - board.length / 2, group.userData.structuralThickness + 0.045, 0.0);
    const backOrigin = new THREE.Vector3(board.back_foot_x - board.length / 2, group.userData.structuralThickness + 0.045, 0.0);
    if (state.loadCase === "pitch") {
      loadLayer.add(createArrow(frontOrigin, [0.0, -0.42, 0.0], 0x1b6b72));
      loadLayer.add(createArrow(backOrigin, [0.0, -0.74, 0.0], 0xb6532f));
    } else if (state.loadCase === "lateral") {
      loadLayer.add(createArrow(frontOrigin, [0.0, -0.4, -0.18], 0x1b6b72));
      loadLayer.add(createArrow(backOrigin, [0.0, -0.5, -0.28], 0xb6532f));
    } else if (state.loadCase === "impulse") {
      loadLayer.add(createArrow(frontOrigin, [0.0, -0.62, 0.0], 0x1b6b72));
      loadLayer.add(createArrow(backOrigin, [0.0, -0.92, 0.0], 0xb6532f));
    } else {
      loadLayer.add(createArrow(frontOrigin, [0.0, -0.46, 0.0], 0x1b6b72));
      loadLayer.add(createArrow(backOrigin, [0.0, -0.46, 0.0], 0xb6532f));
    }
  });
}

function formatDims(params) {
  const dims = [params.size.x, params.size.y, params.size.z].map((value) => (value / INCH).toFixed(1));
  const label = SHAPE_OPTIONS.find((shape) => shape.value === params.shape)?.label ?? params.shape;
  return `${dims.join(" x ")} in, ${label}`;
}

function applySolvedFields(fields) {
  for (const variant of ["tuttle", "track"]) {
    const group = boardVariants[variant];
    group.userData.currentField = fields[variant];
    colorizeGeometry(group.userData.shellGeometry, fields[variant]);
    group.userData.shellGeometry.attributes.color.needsUpdate = true;
  }
}

function updateViewMode() {
  Object.values(boardVariants).forEach((group) => {
    const shellMaterial = group.userData.shellMaterial;
    const wire = group.userData.wire;
    if (state.viewMode === "opaque") {
      shellMaterial.opacity = 0.96;
      shellMaterial.transparent = true;
      shellMaterial.depthWrite = true;
      wire.visible = false;
    } else if (state.viewMode === "xray") {
      shellMaterial.opacity = 0.28;
      shellMaterial.transparent = true;
      shellMaterial.depthWrite = false;
      wire.visible = false;
    } else {
      shellMaterial.opacity = 0.18;
      shellMaterial.transparent = true;
      shellMaterial.depthWrite = false;
      wire.visible = state.layers.shell;
    }
    shellMaterial.needsUpdate = true;
  });
}

function buildClipPlanes(group) {
  const s = group.scale.x;
  const p = group.position;
  const b = group.userData.bounds;
  const xThreshold = p.x + (b.xMin + (b.xMax - b.xMin) * state.clip.x) * s;
  const yThreshold = p.y + (b.yMin + (b.yMax - b.yMin) * state.clip.y) * s;
  const zThreshold = p.z + (b.zMin + (b.zMax - b.zMin) * state.clip.z) * s;
  return [
    new THREE.Plane(new THREE.Vector3(-1, 0, 0), xThreshold),
    new THREE.Plane(new THREE.Vector3(0, -1, 0), yThreshold),
    new THREE.Plane(new THREE.Vector3(0, 0, -1), zThreshold),
  ];
}

function updateClipping() {
  Object.values(boardVariants).forEach((group) => {
    const planes = buildClipPlanes(group);
    group.userData.materials.forEach((material) => {
      material.clippingPlanes = planes;
      material.needsUpdate = true;
    });
  });
}

function setLayerVisibility() {
  Object.values(boardVariants).forEach((group) => {
    Object.entries(state.layers).forEach(([key, visible]) => {
      for (const mesh of group.userData.layers[key] || []) {
        mesh.visible = visible;
      }
    });
  });
  updateViewMode();
}

function makeButtons() {
  const loadCaseRow = document.getElementById("load-case-row");
  Object.entries(state.data.load_cases).forEach(([key, loadInfo]) => {
    const btn = document.createElement("button");
    btn.textContent = loadInfo.label.replace(" Plus ", " + ");
    btn.className = key === state.loadCase ? "active" : "";
    btn.addEventListener("click", async () => {
      state.loadCase = key;
      [...loadCaseRow.querySelectorAll("button")].forEach((el) => el.classList.remove("active"));
      btn.classList.add("active");
      await solveAndRender();
    });
    loadCaseRow.appendChild(btn);
  });

  const viewRow = document.getElementById("view-row");
  const viewLabels = { opaque: "Opaque", xray: "X-Ray", mesh: "Mesh" };
  Object.entries(viewLabels).forEach(([key, label]) => {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.className = key === state.viewMode ? "active" : "";
    btn.addEventListener("click", () => {
      state.viewMode = key;
      [...viewRow.querySelectorAll("button")].forEach((el) => el.classList.remove("active"));
      btn.classList.add("active");
      updateViewMode();
    });
    viewRow.appendChild(btn);
  });

  const toggleRow = document.getElementById("toggle-row");
  const labels = { shell: "Shell Field", insert: "Insert", bonds: "Bond Lines", loads: "Load Arrows" };
  Object.entries(labels).forEach(([key, label]) => {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.className = state.layers[key] ? "active" : "";
    btn.addEventListener("click", () => {
      state.layers[key] = !state.layers[key];
      btn.classList.toggle("active", state.layers[key]);
      setLayerVisibility();
    });
    toggleRow.appendChild(btn);
  });

  bindSlider("slice-x", "slice-x-value", (value) => {
    state.clip.x = value / 100;
    updateClipping();
    return `${value}%`;
  });
  bindSlider("slice-y", "slice-y-value", (value) => {
    state.clip.y = value / 100;
    updateClipping();
    return `${value}%`;
  });
  bindSlider("slice-z", "slice-z-value", (value) => {
    state.clip.z = value / 100;
    updateClipping();
    return `${value}%`;
  });
  bindSlider("strain-gain", "strain-gain-value", (value) => {
    state.strainGain = value / 10;
    applySolvedFields({
      tuttle: boardVariants.tuttle.userData.currentField ?? initialFields.tuttle[state.loadCase],
      track: boardVariants.track.userData.currentField ?? initialFields.track[state.loadCase],
    });
    return `${state.strainGain.toFixed(1)}x`;
  });
}

function bindSlider(inputId, valueId, formatter) {
  const input = document.getElementById(inputId);
  const valueEl = document.getElementById(valueId);
  input.addEventListener("input", () => {
    valueEl.textContent = formatter(Number(input.value));
  });
}

function makeParamControls() {
  const root = document.getElementById("param-grid");
  root.innerHTML = "";
  root.appendChild(buildParamCard("tuttle", "Tuttle Insert", "Bonded to top and bottom skins."));
  root.appendChild(buildParamCard("track", "Dual-Track Insert", "Bonded to the bottom skin only."));
}

function buildParamCard(variant, title, copy) {
  const card = document.createElement("section");
  card.className = "param-card";
  const params = state.params[variant];
  const group = boardVariants[variant];
  const cavityIn = (group.userData.cavityHeight / INCH).toFixed(1);

  card.innerHTML = `
    <h3>${title}</h3>
    <div class="param-copy">${copy} Height is limited to the available inner cavity here: ${cavityIn} in.</div>
    <div class="param-fields">
      <div class="field wide">
        <label for="${variant}-shape">Shape</label>
        <select id="${variant}-shape"></select>
      </div>
      <div class="field">
        <label for="${variant}-length">Length (in)</label>
        <input id="${variant}-length" type="number" min="${PARAM_LIMITS_IN[variant].length[0]}" max="${PARAM_LIMITS_IN[variant].length[1]}" step="0.1" value="${(params.size.x / INCH).toFixed(1)}">
      </div>
      <div class="field">
        <label for="${variant}-width">Width (in)</label>
        <input id="${variant}-width" type="number" min="${PARAM_LIMITS_IN[variant].width[0]}" max="${PARAM_LIMITS_IN[variant].width[1]}" step="0.1" value="${(params.size.y / INCH).toFixed(1)}">
      </div>
      <div class="field wide">
        <label for="${variant}-height">Height (in)</label>
        <input id="${variant}-height" type="number" min="${PARAM_LIMITS_IN[variant].height[0]}" max="${Math.min(PARAM_LIMITS_IN[variant].height[1], group.userData.cavityHeight / INCH).toFixed(1)}" step="0.1" value="${(params.size.z / INCH).toFixed(1)}">
      </div>
    </div>
    <div class="param-actions">
      <button type="button" id="${variant}-reset">Reset ${variant === "tuttle" ? "Tuttle" : "Track"}</button>
    </div>
  `;

  const select = card.querySelector(`#${variant}-shape`);
  SHAPE_OPTIONS.forEach((option) => {
    const el = document.createElement("option");
    el.value = option.value;
    el.textContent = option.label;
    el.selected = option.value === params.shape;
    select.appendChild(el);
  });

  const sync = () => {
    const shape = card.querySelector(`#${variant}-shape`).value;
    const length = Number(card.querySelector(`#${variant}-length`).value) * INCH;
    const width = Number(card.querySelector(`#${variant}-width`).value) * INCH;
    const height = Number(card.querySelector(`#${variant}-height`).value) * INCH;
    state.params[variant] = sanitizeParams({ shape, size: { x: length, y: width, z: height } }, variant, group.userData.cavityHeight);
    for (const [key, field] of [["x", "length"], ["y", "width"], ["z", "height"]]) {
      const input = card.querySelector(`#${variant}-${field}`);
      input.value = (state.params[variant].size[key] / INCH).toFixed(1);
    }
    rebuildInsertMeshes(group);
    scheduleSolve();
  };

  card.querySelectorAll("input, select").forEach((input) => {
    input.addEventListener("change", sync);
  });

  card.querySelector(`#${variant}-reset`).addEventListener("click", () => {
    state.params[variant] = createDefaultParams(state.data)[variant];
    rootRefresh();
    scheduleSolve();
  });

  function rootRefresh() {
    makeParamControls();
    for (const key of ["tuttle", "track"]) {
      rebuildInsertMeshes(boardVariants[key]);
    }
  }

  return card;
}

function scheduleSolve() {
  const token = ++state.solveToken;
  showLoading("Solving reduced-order model...");
  window.setTimeout(async () => {
    if (token !== state.solveToken) {
      return;
    }
    await solveAndRender();
  }, 20);
}

async function solveAndRender() {
  showLoading("Solving reduced-order model...");
  await new Promise((resolve) => window.requestAnimationFrame(resolve));
  const result = solveNormalizedFields(state.data, state.model, state.params, state.loadCase);
  applySolvedFields(result.fields);
  updateLoadArrows();
  updateClipping();
  updateViewMode();
  hideLoading();
}

function resize() {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

async function init() {
  showLoading("Loading TT60 board mesh...");
  const response = await fetch("./viewer-data.json");
  if (!response.ok) {
    throw new Error(`Failed to load viewer-data.json: ${response.status}`);
  }
  state.data = await response.json();
  state.model = preprocessModel(state.data);
  state.params = createDefaultParams(state.data);
  initialFields.tuttle = state.data.fields.tuttle;
  initialFields.track = state.data.fields.track;

  buildVariant("tuttle", state.data, state.data.board, -1.12);
  buildVariant("track", state.data, state.data.board, 1.12);
  makeLabel("Tuttle", -1.12);
  makeLabel("Dual Track", 1.12);
  makeButtons();
  makeParamControls();
  updateLoadArrows();
  setLayerVisibility();
  updateClipping();
  updateViewMode();
  await solveAndRender();
}

init().catch((error) => {
  loadingEl.textContent = error.message;
  console.error(error);
});

window.addEventListener("resize", resize);
resize();

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

animate();
