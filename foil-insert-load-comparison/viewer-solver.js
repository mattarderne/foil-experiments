export const INCH = 0.0254;

export const SHAPE_OPTIONS = [
  { value: "rect", label: "Square / Rect" },
  { value: "rounded", label: "Rounded Rect" },
  { value: "diamond", label: "Diamond" },
  { value: "oval", label: "Oval" },
  { value: "capsule", label: "Rounded Oval" },
];

export const PARAM_LIMITS_IN = {
  tuttle: {
    length: [2.0, 14.0],
    width: [1.0, 8.0],
    height: [1.5, 7.0],
  },
  track: {
    length: [8.0, 28.0],
    width: [2.0, 10.0],
    height: [0.5, 4.0],
  },
};

const SHAPE_FILL = {
  rect: 1.0,
  rounded: 0.96,
  diamond: 0.82,
  oval: 0.89,
  capsule: 0.93,
};

const LOAD_CASES = {
  vertical: {
    mastTotal: 1.0,
    frontShare: 0.5,
    backShare: 0.5,
    pitchAmp: 0.0,
    rollAmp: 0.0,
  },
  pitch: {
    mastTotal: 1.0,
    frontShare: 0.38,
    backShare: 0.62,
    pitchAmp: 0.65,
    rollAmp: 0.0,
  },
  lateral: {
    mastTotal: 0.95,
    frontShare: 0.45,
    backShare: 0.55,
    pitchAmp: 0.1,
    rollAmp: 0.6,
  },
  impulse: {
    mastTotal: 1.65,
    frontShare: 0.42,
    backShare: 0.58,
    pitchAmp: 0.82,
    rollAmp: 0.08,
  },
};

export function createDefaultParams(data) {
  const defaults = data.model?.defaults ?? {};
  return {
    tuttle: {
      size: {
        x: defaults.tuttle?.size?.x ?? data.insert_variants.tuttle.size.x,
        y: defaults.tuttle?.size?.y ?? data.insert_variants.tuttle.size.y,
        z: defaults.tuttle?.size?.z ?? data.insert_variants.tuttle.size.z,
      },
      shape: defaults.tuttle?.shape ?? data.insert_variants.tuttle.shape ?? "rect",
    },
    track: {
      size: {
        x: defaults.track?.size?.x ?? data.insert_variants.track.size.x,
        y: defaults.track?.size?.y ?? data.insert_variants.track.size.y,
        z: defaults.track?.size?.z ?? data.insert_variants.track.size.z,
      },
      shape: defaults.track?.shape ?? data.insert_variants.track.shape ?? "rect",
    },
  };
}

export function preprocessModel(data) {
  const nx = data.model.grid.nx;
  const ny = data.model.grid.ny;
  const x = Float64Array.from(data.model.grid.x);
  const y = Float64Array.from(data.model.grid.y);
  const insideFlat = Uint8Array.from(data.model.grid.inside);
  const ids = new Int32Array(nx * ny);
  ids.fill(-1);

  let count = 0;
  for (let ix = 0; ix < nx; ix += 1) {
    for (let iy = 0; iy < ny; iy += 1) {
      const flat = ix * ny + iy;
      if (!insideFlat[flat]) {
        continue;
      }
      ids[flat] = count;
      count += 1;
    }
  }

  const nodeIx = new Int16Array(count);
  const nodeIy = new Int16Array(count);
  for (let ix = 0; ix < nx; ix += 1) {
    for (let iy = 0; iy < ny; iy += 1) {
      const nodeId = ids[ix * ny + iy];
      if (nodeId < 0) {
        continue;
      }
      nodeIx[nodeId] = ix;
      nodeIy[nodeId] = iy;
    }
  }

  const interpolation = buildInterpolationTable(data.board, x, y);

  return {
    nx,
    ny,
    x,
    y,
    ids,
    insideFlat,
    nNodes: count,
    nodeIx,
    nodeIy,
    dx: x.length > 1 ? x[1] - x[0] : 1,
    dy: y.length > 1 ? y[1] - y[0] : 1,
    centerY: data.board.width * 0.5,
    interpolation,
  };
}

export function solveNormalizedFields(data, model, paramsByVariant, loadCase) {
  const raw = {};
  let sharedPeak = 1e-12;

  for (const variant of ["tuttle", "track"]) {
    const solved = solveVariant(data, model, variant, paramsByVariant[variant], loadCase);
    raw[variant] = solved;
    sharedPeak = Math.max(sharedPeak, solved.peak);
  }

  return {
    fields: {
      tuttle: normalizeField(raw.tuttle.field, sharedPeak),
      track: normalizeField(raw.track.field, sharedPeak),
    },
    raw,
  };
}

function normalizeField(field, peak) {
  const out = new Float32Array(field.length);
  for (let i = 0; i < field.length; i += 1) {
    const normalized = Math.max(field[i] / peak, 0);
    out[i] = Math.pow(normalized, 0.85);
  }
  return out;
}

function solveVariant(data, model, variant, params, loadCase) {
  const board = data.board;
  const caseParams = LOAD_CASES[loadCase];
  const insertMask = buildInsertMask(model, board.mast_x, model.centerY, params.size.x, params.size.y, params.shape);
  const frontFootMask = buildRectMask(model, board.front_foot_x, model.centerY, 0.142, 0.12);
  const backFootMask = buildRectMask(model, board.back_foot_x, model.centerY, 0.18, 0.12);

  const coeff = variantCoefficients(data, variant, params);
  const system = buildSpringSystem(model, insertMask, coeff);
  const force = buildForceVector(model, board, insertMask, frontFootMask, backFootMask, coeff, caseParams);
  const displacement = solvePCG(system, force, model.nNodes * 2);
  const field = mapFieldToSurface(data.board, model, displacement, insertMask, coeff);

  let peak = 0;
  for (let i = 0; i < field.length; i += 1) {
    peak = Math.max(peak, field[i]);
  }

  return {
    field,
    peak,
    coeff,
  };
}

function variantCoefficients(data, variant, params) {
  const defaults = data.model.defaults[variant].size;
  const areaRatio = Math.max((params.size.x * params.size.y) / (defaults.x * defaults.y), 0.15);
  const shapeFill = SHAPE_FILL[params.shape] ?? 1.0;
  const internalDepth = Math.max(data.board.thickness_max - 2 * data.model.shell_thickness, 0.02);

  if (variant === "tuttle") {
    const depthRatio = clamp(params.size.z / internalDepth, 0.25, 1.45);
    const stiffnessScale = Math.pow(areaRatio * shapeFill, 0.6) * Math.pow(depthRatio, 1.15);
    const topShare = clamp(0.12 + 0.32 * depthRatio * Math.sqrt(shapeFill), 0.08, 0.48);
    return {
      kCoreInsert: 38.0 * stiffnessScale,
      topInsertShare: topShare,
      bottomInsertShare: 1.0 - topShare,
      topSkinBoost: 1.25 + 0.95 * depthRatio * Math.pow(areaRatio, 0.18) * shapeFill,
      bottomSkinBoost: 1.25 + 0.95 * depthRatio * Math.pow(areaRatio, 0.18) * shapeFill,
    };
  }

  const depthRatio = clamp(params.size.z / defaults.z, 0.35, 2.25);
  const spreadScale = Math.pow(areaRatio, 0.34) * Math.pow(shapeFill, 0.5);
  return {
    kCoreInsert: 7.0 * Math.pow(areaRatio * shapeFill, 0.55) * Math.pow(depthRatio, 0.8),
    topInsertShare: 0.0,
    bottomInsertShare: 1.0,
    topSkinBoost: clamp(1.0 + 0.08 * (spreadScale - 1.0), 1.0, 1.4),
    bottomSkinBoost: clamp(2.8 * Math.pow(spreadScale, 0.65) * Math.pow(depthRatio, 0.28), 1.6, 4.5),
  };
}

function buildSpringSystem(model, insertMask, coeff) {
  const kFound = 0.02;
  const kCoreEps = 1.8;
  const kSkinX = 6.0 / Math.max(model.dx, 1e-6);
  const kSkinY = 4.5 / Math.max(model.dy, 1e-6);
  const ndof = model.nNodes * 2;

  const diag = new Float64Array(ndof);
  diag.fill(kFound);

  const edgeA = [];
  const edgeB = [];
  const edgeK = [];
  const verticalK = new Float64Array(model.nNodes);
  const topEdges = [];
  const bottomEdges = [];

  for (let nodeId = 0; nodeId < model.nNodes; nodeId += 1) {
    const ix = model.nodeIx[nodeId];
    const iy = model.nodeIy[nodeId];
    const flat = ix * model.ny + iy;
    const vertical = insertMask[flat] ? coeff.kCoreInsert : kCoreEps;
    verticalK[nodeId] = vertical;
    const topDof = nodeId;
    const bottomDof = nodeId + model.nNodes;
    diag[topDof] += vertical;
    diag[bottomDof] += vertical;
    edgeA.push(topDof);
    edgeB.push(bottomDof);
    edgeK.push(vertical);

    if (ix + 1 < model.nx) {
      const rightFlat = (ix + 1) * model.ny + iy;
      const rightNode = model.ids[rightFlat];
      if (rightNode >= 0) {
        const boosted = insertMask[flat] && insertMask[rightFlat];
        const kTop = kSkinX * (boosted ? coeff.topSkinBoost : 1.0);
        const kBottom = kSkinX * (boosted ? coeff.bottomSkinBoost : 1.0);
        const topNext = rightNode;
        const bottomNext = rightNode + model.nNodes;
        diag[topDof] += kTop;
        diag[topNext] += kTop;
        diag[bottomDof] += kBottom;
        diag[bottomNext] += kBottom;
        edgeA.push(topDof);
        edgeB.push(topNext);
        edgeK.push(kTop);
        edgeA.push(bottomDof);
        edgeB.push(bottomNext);
        edgeK.push(kBottom);
        topEdges.push([nodeId, rightNode, kTop]);
        bottomEdges.push([nodeId, rightNode, kBottom]);
      }
    }

    if (iy + 1 < model.ny) {
      const upFlat = ix * model.ny + (iy + 1);
      const upNode = model.ids[upFlat];
      if (upNode >= 0) {
        const boosted = insertMask[flat] && insertMask[upFlat];
        const kTop = kSkinY * (boosted ? coeff.topSkinBoost : 1.0);
        const kBottom = kSkinY * (boosted ? coeff.bottomSkinBoost : 1.0);
        const topNext = upNode;
        const bottomNext = upNode + model.nNodes;
        diag[nodeId] += kTop;
        diag[topNext] += kTop;
        diag[nodeId + model.nNodes] += kBottom;
        diag[bottomNext] += kBottom;
        edgeA.push(nodeId);
        edgeB.push(topNext);
        edgeK.push(kTop);
        edgeA.push(nodeId + model.nNodes);
        edgeB.push(bottomNext);
        edgeK.push(kBottom);
        topEdges.push([nodeId, upNode, kTop]);
        bottomEdges.push([nodeId, upNode, kBottom]);
      }
    }
  }

  return {
    ndof,
    diag,
    edgeA: Int32Array.from(edgeA),
    edgeB: Int32Array.from(edgeB),
    edgeK: Float64Array.from(edgeK),
    verticalK,
    topEdges,
    bottomEdges,
  };
}

function buildForceVector(model, board, insertMask, frontFootMask, backFootMask, coeff, caseParams) {
  const force = new Float64Array(model.nNodes * 2);
  const insertNodeIds = collectNodeIds(model, insertMask);
  if (!insertNodeIds.length) {
    return force;
  }

  const xNorm = new Float64Array(insertNodeIds.length);
  const yNorm = new Float64Array(insertNodeIds.length);
  const weights = new Float64Array(insertNodeIds.length);
  const mastY = model.centerY;

  let halfLength = 0.5 * Math.max(1e-6, maxSpan(model, insertMask, "x"));
  let halfWidth = 0.5 * Math.max(1e-6, maxSpan(model, insertMask, "y"));
  if (!Number.isFinite(halfLength)) {
    halfLength = 0.05;
  }
  if (!Number.isFinite(halfWidth)) {
    halfWidth = 0.03;
  }

  for (let i = 0; i < insertNodeIds.length; i += 1) {
    const nodeId = insertNodeIds[i];
    const x = model.x[model.nodeIx[nodeId]];
    const y = model.y[model.nodeIy[nodeId]];
    xNorm[i] = (x - board.mast_x) / halfLength;
    yNorm[i] = (y - mastY) / halfWidth;
    weights[i] = Math.max((1 + caseParams.pitchAmp * xNorm[i]) * (1 + caseParams.rollAmp * yNorm[i]), 0.1);
  }

  distribute(
    force,
    insertNodeIds.map((nodeId) => nodeId + model.nNodes),
    caseParams.mastTotal * coeff.bottomInsertShare,
    weights,
  );

  if (coeff.topInsertShare > 0) {
    const topWeights = new Float64Array(insertNodeIds.length);
    for (let i = 0; i < insertNodeIds.length; i += 1) {
      topWeights[i] = Math.max(1 - 0.45 * caseParams.pitchAmp * xNorm[i], 0.1);
    }
    distribute(
      force,
      insertNodeIds,
      caseParams.mastTotal * coeff.topInsertShare,
      topWeights,
    );
  }

  const frontIds = collectNodeIds(model, frontFootMask);
  const backIds = collectNodeIds(model, backFootMask);
  distribute(force, frontIds, -caseParams.mastTotal * caseParams.frontShare);
  distribute(force, backIds, -caseParams.mastTotal * caseParams.backShare);

  let residual = 0;
  for (let i = 0; i < force.length; i += 1) {
    residual += force[i];
  }

  if (Math.abs(residual) > 1e-12) {
    const active = [
      ...frontIds,
      ...backIds,
      ...insertNodeIds.map((nodeId) => nodeId + model.nNodes),
      ...(coeff.topInsertShare > 0 ? insertNodeIds : []),
    ];
    const correction = residual / Math.max(active.length, 1);
    for (const dof of active) {
      force[dof] -= correction;
    }
  }

  return force;
}

function solvePCG(system, force, ndof) {
  const x = new Float64Array(ndof);
  const r = Float64Array.from(force);
  const z = new Float64Array(ndof);
  const p = new Float64Array(ndof);
  const ap = new Float64Array(ndof);
  const invDiag = new Float64Array(ndof);

  let normB = 0;
  for (let i = 0; i < ndof; i += 1) {
    invDiag[i] = 1 / Math.max(system.diag[i], 1e-9);
    normB += force[i] * force[i];
  }
  normB = Math.sqrt(normB) || 1;

  for (let i = 0; i < ndof; i += 1) {
    z[i] = r[i] * invDiag[i];
    p[i] = z[i];
  }

  let rzOld = dot(r, z);
  if (Math.abs(rzOld) < 1e-18) {
    return x;
  }

  for (let iter = 0; iter < 420; iter += 1) {
    multiplySystem(system, p, ap);
    const denom = dot(p, ap);
    if (Math.abs(denom) < 1e-18) {
      break;
    }
    const alpha = rzOld / denom;

    let residSq = 0;
    for (let i = 0; i < ndof; i += 1) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
      residSq += r[i] * r[i];
    }
    if (Math.sqrt(residSq) / normB < 1e-6) {
      break;
    }

    for (let i = 0; i < ndof; i += 1) {
      z[i] = r[i] * invDiag[i];
    }
    const rzNew = dot(r, z);
    const beta = rzNew / Math.max(rzOld, 1e-18);
    for (let i = 0; i < ndof; i += 1) {
      p[i] = z[i] + beta * p[i];
    }
    rzOld = rzNew;
  }

  return x;
}

function multiplySystem(system, vector, out) {
  out.fill(0);
  for (let i = 0; i < system.diag.length; i += 1) {
    out[i] = system.diag[i] * vector[i];
  }
  for (let edge = 0; edge < system.edgeA.length; edge += 1) {
    const a = system.edgeA[edge];
    const b = system.edgeB[edge];
    const k = system.edgeK[edge];
    out[a] -= k * vector[b];
    out[b] -= k * vector[a];
  }
}

function mapFieldToSurface(board, model, displacement, insertMask, coeff) {
  const n = model.nNodes;
  const topEnergy = new Float64Array(n);
  const bottomEnergy = new Float64Array(n);
  const topU = displacement.subarray(0, n);
  const bottomU = displacement.subarray(n);

  for (let nodeId = 0; nodeId < n; nodeId += 1) {
    const eVert = 0.5 * coeffForNode(insertMask, model, nodeId, coeff.kCoreInsert) * (topU[nodeId] - bottomU[nodeId]) ** 2;
    topEnergy[nodeId] += 0.5 * eVert;
    bottomEnergy[nodeId] += 0.5 * eVert;
  }

  const system = buildEnergyEdges(model, insertMask, coeff);
  for (const [a, b, k] of system.topEdges) {
    const energy = 0.5 * k * (topU[a] - topU[b]) ** 2;
    topEnergy[a] += 0.5 * energy;
    topEnergy[b] += 0.5 * energy;
  }
  for (const [a, b, k] of system.bottomEdges) {
    const energy = 0.5 * k * (bottomU[a] - bottomU[b]) ** 2;
    bottomEnergy[a] += 0.5 * energy;
    bottomEnergy[b] += 0.5 * energy;
  }

  const topGrid = new Float64Array(model.nx * model.ny);
  const bottomGrid = new Float64Array(model.nx * model.ny);
  for (let nodeId = 0; nodeId < n; nodeId += 1) {
    const flat = model.nodeIx[nodeId] * model.ny + model.nodeIy[nodeId];
    topGrid[flat] = topEnergy[nodeId];
    bottomGrid[flat] = bottomEnergy[nodeId];
  }

  const field = new Float64Array(board.vertices.length);
  for (let i = 0; i < board.vertices.length; i += 1) {
    const interp = model.interpolation[i];
    const topVal = bilerp(topGrid, model.ny, interp);
    const bottomVal = bilerp(bottomGrid, model.ny, interp);
    const topness = board.topness[i];
    const bottomness = board.bottomness[i];
    const railness = 1 - topness - bottomness;
    field[i] = topness * topVal + bottomness * bottomVal + railness * 0.5 * (topVal + bottomVal);
  }

  return field;
}

function buildEnergyEdges(model, insertMask, coeff) {
  const kSkinX = 6.0 / Math.max(model.dx, 1e-6);
  const kSkinY = 4.5 / Math.max(model.dy, 1e-6);
  const topEdges = [];
  const bottomEdges = [];

  for (let nodeId = 0; nodeId < model.nNodes; nodeId += 1) {
    const ix = model.nodeIx[nodeId];
    const iy = model.nodeIy[nodeId];
    const flat = ix * model.ny + iy;

    if (ix + 1 < model.nx) {
      const nextFlat = (ix + 1) * model.ny + iy;
      const next = model.ids[nextFlat];
      if (next >= 0) {
        const boosted = insertMask[flat] && insertMask[nextFlat];
        topEdges.push([nodeId, next, kSkinX * (boosted ? coeff.topSkinBoost : 1.0)]);
        bottomEdges.push([nodeId, next, kSkinX * (boosted ? coeff.bottomSkinBoost : 1.0)]);
      }
    }

    if (iy + 1 < model.ny) {
      const nextFlat = ix * model.ny + (iy + 1);
      const next = model.ids[nextFlat];
      if (next >= 0) {
        const boosted = insertMask[flat] && insertMask[nextFlat];
        topEdges.push([nodeId, next, kSkinY * (boosted ? coeff.topSkinBoost : 1.0)]);
        bottomEdges.push([nodeId, next, kSkinY * (boosted ? coeff.bottomSkinBoost : 1.0)]);
      }
    }
  }

  return { topEdges, bottomEdges };
}

function coeffForNode(insertMask, model, nodeId, insertValue) {
  const flat = model.nodeIx[nodeId] * model.ny + model.nodeIy[nodeId];
  return insertMask[flat] ? insertValue : 1.8;
}

function buildInsertMask(model, cx, cy, length, width, shape) {
  const mask = new Uint8Array(model.nx * model.ny);
  for (let ix = 0; ix < model.nx; ix += 1) {
    for (let iy = 0; iy < model.ny; iy += 1) {
      const flat = ix * model.ny + iy;
      if (!model.insideFlat[flat]) {
        continue;
      }
      if (containsShape(model.x[ix], model.y[iy], cx, cy, length, width, shape)) {
        mask[flat] = 1;
      }
    }
  }
  return mask;
}

function buildRectMask(model, cx, cy, length, width) {
  const mask = new Uint8Array(model.nx * model.ny);
  for (let ix = 0; ix < model.nx; ix += 1) {
    for (let iy = 0; iy < model.ny; iy += 1) {
      const flat = ix * model.ny + iy;
      if (!model.insideFlat[flat]) {
        continue;
      }
      if (Math.abs(model.x[ix] - cx) <= length * 0.5 && Math.abs(model.y[iy] - cy) <= width * 0.5) {
        mask[flat] = 1;
      }
    }
  }
  return mask;
}

function collectNodeIds(model, mask) {
  const out = [];
  for (let ix = 0; ix < model.nx; ix += 1) {
    for (let iy = 0; iy < model.ny; iy += 1) {
      const flat = ix * model.ny + iy;
      if (!mask[flat]) {
        continue;
      }
      const nodeId = model.ids[flat];
      if (nodeId >= 0) {
        out.push(nodeId);
      }
    }
  }
  return out;
}

function maxSpan(model, mask, axis) {
  let minValue = Infinity;
  let maxValue = -Infinity;
  for (let ix = 0; ix < model.nx; ix += 1) {
    for (let iy = 0; iy < model.ny; iy += 1) {
      const flat = ix * model.ny + iy;
      if (!mask[flat]) {
        continue;
      }
      const value = axis === "x" ? model.x[ix] : model.y[iy];
      minValue = Math.min(minValue, value);
      maxValue = Math.max(maxValue, value);
    }
  }
  return maxValue - minValue;
}

function containsShape(x, y, cx, cy, length, width, shape) {
  const dx = Math.abs(x - cx);
  const dy = Math.abs(y - cy);
  const hx = Math.max(length * 0.5, 1e-6);
  const hy = Math.max(width * 0.5, 1e-6);

  switch (shape) {
    case "oval":
      return (dx * dx) / (hx * hx) + (dy * dy) / (hy * hy) <= 1;
    case "diamond":
      return dx / hx + dy / hy <= 1;
    case "rounded":
      return containsRoundedRect(dx, dy, hx, hy, Math.min(hx, hy) * 0.28);
    case "capsule":
      return containsCapsule(dx, dy, hx, hy);
    case "rect":
    default:
      return dx <= hx && dy <= hy;
  }
}

function containsRoundedRect(dx, dy, hx, hy, radius) {
  if (dx > hx || dy > hy) {
    return false;
  }
  if (dx <= hx - radius || dy <= hy - radius) {
    return true;
  }
  const cx = dx - (hx - radius);
  const cy = dy - (hy - radius);
  return cx * cx + cy * cy <= radius * radius;
}

function containsCapsule(dx, dy, hx, hy) {
  if (hx >= hy) {
    const radius = hy;
    if (dx <= hx - radius) {
      return dy <= radius;
    }
    const cx = dx - (hx - radius);
    return cx * cx + dy * dy <= radius * radius;
  }
  const radius = hx;
  if (dy <= hy - radius) {
    return dx <= radius;
  }
  const cy = dy - (hy - radius);
  return dx * dx + cy * cy <= radius * radius;
}

function distribute(force, dofs, magnitude, weights = null) {
  if (!dofs.length || Math.abs(magnitude) <= 1e-12) {
    return;
  }
  let total = 0;
  if (weights) {
    for (let i = 0; i < weights.length; i += 1) {
      total += Math.max(weights[i], 1e-9);
    }
  } else {
    total = dofs.length;
  }
  for (let i = 0; i < dofs.length; i += 1) {
    const weight = weights ? Math.max(weights[i], 1e-9) / total : 1 / total;
    force[dofs[i]] += magnitude * weight;
  }
}

function buildInterpolationTable(board, xGrid, yGrid) {
  return board.vertices.map(([vx, vy]) => {
    const xRef = bracket(xGrid, vx);
    const yRef = bracket(yGrid, vy);
    return {
      ix0: xRef.i0,
      ix1: xRef.i1,
      iy0: yRef.i0,
      iy1: yRef.i1,
      tx: xRef.t,
      ty: yRef.t,
    };
  });
}

function bracket(values, query) {
  if (query <= values[0]) {
    return { i0: 0, i1: 0, t: 0 };
  }
  if (query >= values[values.length - 1]) {
    const last = values.length - 1;
    return { i0: last, i1: last, t: 0 };
  }
  let lo = 0;
  let hi = values.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (values[mid] <= query) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return {
    i0: lo,
    i1: hi,
    t: (query - values[lo]) / Math.max(values[hi] - values[lo], 1e-12),
  };
}

function bilerp(values, ny, interp) {
  const idx00 = interp.ix0 * ny + interp.iy0;
  const idx10 = interp.ix1 * ny + interp.iy0;
  const idx01 = interp.ix0 * ny + interp.iy1;
  const idx11 = interp.ix1 * ny + interp.iy1;
  const v00 = values[idx00];
  const v10 = values[idx10];
  const v01 = values[idx01];
  const v11 = values[idx11];
  const vx0 = (1 - interp.tx) * v00 + interp.tx * v10;
  const vx1 = (1 - interp.tx) * v01 + interp.tx * v11;
  return (1 - interp.ty) * vx0 + interp.ty * vx1;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
