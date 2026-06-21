/**
 * exclusion.js — per-face exclusion masking
 *
 * Provides three capabilities:
 *  1. buildAdjacency   – builds an inter-triangle adjacency list with dihedral
 *                        angles and precomputes per-triangle centroids.
 *  2. bucketFill       – BFS flood fill that respects a max dihedral-angle
 *                        threshold (stops at "sharp" edges).
 *  3. buildExclusionOverlayGeo – compact geometry for the orange preview overlay.
 *  4. buildFaceWeights – per-vertex exclusion weights for the subdivision pass.
 */

import * as THREE from 'three';
import { QuantizedPointMap } from './meshIndex.js';

const QUANT = 1e4;

// ── Adjacency & centroids ─────────────────────────────────────────────────────

/**
 * Build inter-triangle adjacency data for a non-indexed BufferGeometry.
 *
 * @param {THREE.BufferGeometry} geometry  – non-indexed
 * @returns {{
 *   adjacency:   Array<Array<{neighbor:number, angle:number}>>,
 *   centroids:   Float32Array   (triCount × 3, geometry-local centroid per triangle),
 *   boundRadii:  Float32Array   (triCount, max vertex-to-centroid distance per triangle),
 *   faceNormals: Float32Array   (triCount × 3, geometry-local unit face normal per triangle)
 * }}
 */
export function buildAdjacency(geometry) {
  const posAttr  = geometry.attributes.position;
  const triCount = posAttr.count / 3;

  // Pre-allocate face normals, centroids, and per-triangle bounding radii
  const faceNormals = new Float32Array(triCount * 3);
  const centroids   = new Float32Array(triCount * 3);
  const boundRadii  = new Float32Array(triCount); // max vertex-to-centroid distance

  const vA = new THREE.Vector3();
  const vB = new THREE.Vector3();
  const vC = new THREE.Vector3();
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const fn = new THREE.Vector3();

  for (let t = 0; t < triCount; t++) {
    const i = t * 3;
    vA.fromBufferAttribute(posAttr, i);
    vB.fromBufferAttribute(posAttr, i + 1);
    vC.fromBufferAttribute(posAttr, i + 2);

    e1.subVectors(vB, vA);
    e2.subVectors(vC, vA);
    fn.crossVectors(e1, e2).normalize();

    faceNormals[i]     = fn.x;
    faceNormals[i + 1] = fn.y;
    faceNormals[i + 2] = fn.z;

    const cx = (vA.x + vB.x + vC.x) / 3;
    const cy = (vA.y + vB.y + vC.y) / 3;
    const cz = (vA.z + vB.z + vC.z) / 3;
    centroids[i]     = cx;
    centroids[i + 1] = cy;
    centroids[i + 2] = cz;
    const dA = (vA.x-cx)**2 + (vA.y-cy)**2 + (vA.z-cz)**2;
    const dB = (vB.x-cx)**2 + (vB.y-cy)**2 + (vB.z-cz)**2;
    const dC = (vC.x-cx)**2 + (vC.y-cy)**2 + (vC.z-cz)**2;
    boundRadii[t] = Math.sqrt(Math.max(dA, dB, dC));
  }

  // Build edge → triangle list (two triangles share an edge iff they share two
  // vertex positions after quantization-based deduplication).
  // Vertex-dedup pass: assign a numeric ID to each unique quantised position.
  const posToId = new QuantizedPointMap(QUANT, Math.min(triCount * 3, 1 << 22));
  let nextId = 0;
  const vertId = new Uint32Array(triCount * 3);
  for (let i = 0; i < triCount * 3; i++) {
    const id = posToId.getOrSet(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i), nextId);
    if (posToId.inserted) nextId++;
    vertId[i] = id;
  }
  // nextId^2 < MAX_SAFE_INTEGER → safe up to ~94M unique vertices
  const numEdgeKey = (a, b) => a < b ? a * nextId + b : b * nextId + a;

  const edgeMap = new Map();
  const edgePairs = [0, 1, 0, 2, 1, 2]; // vertex-index pairs within triangle

  for (let t = 0; t < triCount; t++) {
    const base = t * 3;
    for (let e = 0; e < 6; e += 2) {
      const ek = numEdgeKey(vertId[base + edgePairs[e]], vertId[base + edgePairs[e + 1]]);
      const entry = edgeMap.get(ek);
      if (entry) entry.push(t);
      else edgeMap.set(ek, [t]);
    }
  }

  // Convert edge map to adjacency list with per-edge dihedral angle
  // Array from buildAdjacency
  const adjacency = new Array(triCount);
  for (let t = 0; t < triCount; t++) adjacency[t] = [];

  let openEdgeCount = 0;
  let nonManifoldEdgeCount = 0;

  for (const [, tris] of edgeMap) {
    if (tris.length === 1) { openEdgeCount++; continue; }
    if (tris.length > 2) nonManifoldEdgeCount++;
    const [a, b] = tris;
    const nAx = faceNormals[a * 3], nAy = faceNormals[a * 3 + 1], nAz = faceNormals[a * 3 + 2];
    const nBx = faceNormals[b * 3], nBy = faceNormals[b * 3 + 1], nBz = faceNormals[b * 3 + 2];
    const dot      = Math.max(-1, Math.min(1, nAx * nBx + nAy * nBy + nAz * nBz));
    const angleDeg = Math.acos(dot) * (180 / Math.PI);
    adjacency[a].push({ neighbor: b, angle: angleDeg });
    adjacency[b].push({ neighbor: a, angle: angleDeg });
  }

  return { adjacency, centroids, boundRadii, faceNormals, openEdgeCount, nonManifoldEdgeCount };
}

// ── Bucket fill ───────────────────────────────────────────────────────────────

/**
 * BFS flood fill starting from seedTriIdx.
 * Spreads across edges whose dihedral angle ≤ thresholdDeg.
 *
 * @param {number} seedTriIdx
 * @param {Array<Array<{neighbor:number, angle:number}>>} adjacency
 * @param {number} thresholdDeg
 * @returns {Set<number>}  set of triangle indices in the filled region
 */
export function bucketFill(seedTriIdx, adjacency, thresholdDeg) {
  const visited = new Set([seedTriIdx]);
  const queue   = [seedTriIdx];
  let head = 0;
  while (head < queue.length) {
    const cur       = queue[head++];
    const neighbors = adjacency[cur];
    if (!neighbors) continue;
    for (const { neighbor, angle } of neighbors) {
      if (!visited.has(neighbor) && angle <= thresholdDeg) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
  }
  return visited;
}

// ── Overlay geometry ──────────────────────────────────────────────────────────

/**
 * Build a compact non-indexed BufferGeometry for an overlay.
 *
 * @param {THREE.BufferGeometry} geometry   – non-indexed source geometry
 * @param {Set<number>}          faceSet
 * @param {boolean}              [invert=false]  when true, include faces NOT in faceSet
 * @returns {THREE.BufferGeometry}
 */
export function buildExclusionOverlayGeo(geometry, faceSet, invert = false) {
  const srcPos   = geometry.attributes.position.array;
  const srcNrm   = geometry.attributes.normal ? geometry.attributes.normal.array : null;
  const total    = srcPos.length / 9; // total triangle count
  const isArr    = faceSet instanceof Uint8Array;

  // Count included faces
  let setSize;
  if (isArr) {
    setSize = 0;
    for (let i = 0; i < faceSet.length; i++) if (faceSet[i]) setSize++;
  } else {
    setSize = faceSet.size;
  }
  const count    = invert ? total - setSize : setSize;
  const outPos   = new Float32Array(count * 9);
  const outNrm   = srcNrm ? new Float32Array(count * 9) : null;
  let dst = 0;
  if (invert) {
    for (let t = 0; t < total; t++) {
      if (isArr ? faceSet[t] : faceSet.has(t)) continue;
      const src = t * 9;
      outPos.set(srcPos.subarray(src, src + 9), dst);
      if (outNrm) outNrm.set(srcNrm.subarray(src, src + 9), dst);
      dst += 9;
    }
  } else {
    if (isArr) {
      for (let t = 0; t < faceSet.length; t++) {
        if (!faceSet[t]) continue;
        const src = t * 9;
        outPos.set(srcPos.subarray(src, src + 9), dst);
        if (outNrm) outNrm.set(srcNrm.subarray(src, src + 9), dst);
        dst += 9;
      }
    } else {
      for (const t of faceSet) {
        const src = t * 9;
        outPos.set(srcPos.subarray(src, src + 9), dst);
        if (outNrm) outNrm.set(srcNrm.subarray(src, src + 9), dst);
        dst += 9;
      }
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(outPos, 3));
  if (outNrm) geo.setAttribute('normal', new THREE.BufferAttribute(outNrm, 3));
  return geo;
}

// ── Per-layer weight channels (for subdivision threading) ────────────────────

/**
 * Build one-hot per-vertex layer-membership weights for a non-indexed geometry.
 *
 * Channel k corresponds to layers[k]. Every layer (Layer 1 included) is
 * paint-driven: a face is textured only by the layer whose faceSet contains it.
 * Faces not painted by ANY layer get all-zero weights → no texture (the bake /
 * preview produce zero displacement there, so the model stays flat). Regions are
 * disjoint, so the last layer that claims a face wins.
 * subdivision.js threads these channels through edge splits via linear
 * interpolation, producing a smooth 0→1 blend band one original-edge wide at
 * each region boundary (sharp dihedral edges stay hard).
 *
 * @param {THREE.BufferGeometry} geometry  – non-indexed
 * @param {Array<{faceSet:Set<number>}>} layers
 * @returns {{ weights: Float32Array, layerCount: number }}
 *          weights length = position.count × layerCount (K channels per vertex)
 */
/**
 * Per-original-face channel map: faceChannel[f] = the layer index that paints
 * face f, or -1 if unpainted. Disjoint regions, last layer wins.
 */
export function buildFaceChannels(layers, triCount) {
  const fc = new Int32Array(triCount).fill(-1);
  for (let ch = 0; ch < layers.length; ch++) {
    const fs = layers[ch].faceSet;
    if (!fs) continue;
    for (const f of fs) if (f >= 0 && f < triCount) fc[f] = ch;
  }
  return fc;
}

/**
 * Build HARD per-face layer weights for a SUBDIVIDED geometry, from the
 * subdivided→original face map (faceParentId) and the original faceChannel.
 * Each subdivided face's 3 vertices get the one-hot of its parent's layer
 * (or all-zero if the parent is unpainted). This is bleed-free: a painted
 * region textures exactly its faces and no neighbours, unlike the welded
 * vertex-interpolation path which spreads weight across shared vertices on
 * coarse meshes.
 *
 * @returns {Float32Array} length = subTriCount × 3 × K
 */
export function buildLayerWeightsFromParents(subTriCount, faceParentId, faceChannel, K) {
  const weights = new Float32Array(subTriCount * 3 * K);
  for (let f = 0; f < subTriCount; f++) {
    const parent = faceParentId ? faceParentId[f] : f;
    const ch = (parent >= 0 && parent < faceChannel.length) ? faceChannel[parent] : -1;
    if (ch < 0) continue;
    const base = f * 3 * K;
    weights[base + ch]         = 1;
    weights[base + K + ch]     = 1;
    weights[base + 2 * K + ch] = 1;
  }
  return weights;
}

export function buildLayerWeights(geometry, layers) {
  const count    = geometry.attributes.position.count;
  const triCount = count / 3;
  const K        = Math.max(1, layers.length);
  const weights  = new Float32Array(count * K);

  // Resolve each face to its owning channel; -1 = unpainted → no texture.
  const faceChannel = new Int32Array(triCount).fill(-1);
  for (let ch = 0; ch < layers.length; ch++) {
    const fs = layers[ch].faceSet;
    if (!fs) continue;
    for (const f of fs) if (f >= 0 && f < triCount) faceChannel[f] = ch;
  }

  for (let f = 0; f < triCount; f++) {
    const ch = faceChannel[f];
    if (ch < 0) continue;            // unpainted face → all-zero weights (flat)
    const base = f * 3 * K;
    weights[base + ch]         = 1; // vertex 0
    weights[base + K + ch]     = 1; // vertex 1
    weights[base + 2 * K + ch] = 1; // vertex 2
  }
  return { weights, layerCount: K };
}

// ── Layer overlay geometry ────────────────────────────────────────────────────

/**
 * Build a vertex-colored overlay geometry highlighting each face by the color
 * of the layer that owns it. Faces not claimed by any layer's faceSet are
 * skipped (they belong to the implicit base layer and stay unhighlighted).
 *
 * @param {THREE.BufferGeometry} geometry  – non-indexed source geometry
 * @param {Array<{color:number, faceSet:Set<number>}>} layers
 * @returns {THREE.BufferGeometry}
 */
export function buildLayerOverlayGeo(geometry, layers) {
  const srcPos = geometry.attributes.position.array;
  // Count claimed faces (faceSets are disjoint, but guard against double-count).
  const claimed = new Map(); // faceIdx → color
  for (const layer of layers) {
    if (!layer.faceSet || layer.faceSet.size === 0) continue;
    for (const f of layer.faceSet) claimed.set(f, layer.color);
  }
  const count  = claimed.size;
  const outPos = new Float32Array(count * 9);
  const outCol = new Float32Array(count * 9);
  let dst = 0;
  for (const [f, color] of claimed) {
    const src = f * 9;
    outPos.set(srcPos.subarray(src, src + 9), dst);
    const r = ((color >> 16) & 0xff) / 255;
    const g = ((color >> 8)  & 0xff) / 255;
    const b = (color & 0xff) / 255;
    for (let v = 0; v < 3; v++) {
      outCol[dst + v * 3]     = r;
      outCol[dst + v * 3 + 1] = g;
      outCol[dst + v * 3 + 2] = b;
    }
    dst += 9;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(outPos, 3));
  geo.setAttribute('color',    new THREE.BufferAttribute(outCol, 3));
  return geo;
}

// ── Face-weight array for subdivision ────────────────────────────────────────

/**
 * Build a per-non-indexed-vertex exclusion weight array.
 * Vertex i (in the non-indexed buffer) belongs to triangle floor(i/3).
 * Excluded triangles get weight 1.0, all others 0.0.
 * subdivision.js threads these through edge splits via linear interpolation,
 * producing smooth 0→1 transitions at exclusion boundaries.
 *
 * @param {THREE.BufferGeometry} geometry
 * @param {Set<number>}          excludedFaces
 * @returns {Float32Array}  length = geometry.attributes.position.count
 */
export function buildFaceWeights(geometry, excludedFaces, invert = false) {
  const count   = geometry.attributes.position.count;
  const weights = new Float32Array(count); // default 0.0 (included)
  if (invert) {
    // Include-only mode: all faces start excluded (1.0); painted faces are included (0.0)
    weights.fill(1.0);
    for (const t of excludedFaces) {
      weights[t * 3]     = 0.0;
      weights[t * 3 + 1] = 0.0;
      weights[t * 3 + 2] = 0.0;
    }
  } else {
    for (const t of excludedFaces) {
      weights[t * 3]     = 1.0;
      weights[t * 3 + 1] = 1.0;
      weights[t * 3 + 2] = 1.0;
    }
  }
  return weights;
}
