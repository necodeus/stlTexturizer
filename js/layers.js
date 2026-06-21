/**
 * layers.js — multi-texture layer model.
 *
 * BumpMesh historically applied ONE texture with ONE set of UV/projection
 * parameters and a single binary face mask. This module introduces "layers":
 * each layer carries its own texture (mapEntry), its own UV/projection
 * parameters (uv) and the set of original-mesh faces it covers (faceSet).
 *
 * The global `settings` object in main.js stays the single source of truth for
 * mesh-wide parameters AND mirrors the *active* layer's UV parameters, so the
 * entire existing slider wiring keeps writing into `settings.*`. On layer
 * switch we copy `layer.uv → settings` (and refresh the sliders); on every
 * slider change we copy the relevant `settings.* → activeLayer.uv`.
 */

// Hard cap — bounded by GPU uniform-array / vertex-attribute budget in the
// multi-texture preview shader (two vec4 weight attributes = 8 channels).
export const MAX_LAYERS = 8;

/**
 * The subset of `settings` fields that are PER-LAYER (texture + projection/UV).
 * Everything not listed here (refineLength, maxTriangles, regularize*,
 * bottom/topAngleLimit, smoothBottom, harvest*, noDownwardZ, lockScale,
 * snapSeamlessWrap, blendNormalSmoothing, …) stays mesh-global in `settings`.
 */
export const LAYER_UV_FIELDS = [
  'mappingMode',
  'scaleU', 'scaleV',
  'offsetU', 'offsetV',
  'rotation',
  'amplitude', 'textureHeight', 'invertDisplacement',
  'mappingBlend', 'seamBandWidth', 'capAngle',
  'textureSmoothing',
  'cylinderCenterX', 'cylinderCenterY', 'cylinderRadius',
];

// Distinct, high-contrast colors for the paint overlay (one per layer slot).
export const LAYER_COLORS = [
  0x4f9dff, // blue
  0xff8a3d, // orange
  0x46d39a, // green
  0xc98bff, // purple
  0xffd23f, // yellow
  0xff5d8f, // pink
  0x33d6e0, // cyan
  0xa0d24a, // lime
];

let _nextLayerId = 1;

/** Allocate a process-unique layer id. */
export function nextLayerId() { return _nextLayerId++; }

/**
 * Pull the per-layer UV defaults out of a settings snapshot. Used to seed a
 * new layer's uv from the current global settings so a freshly added layer
 * starts from whatever the user last had on screen.
 */
export function extractUV(settings) {
  const uv = {};
  for (const k of LAYER_UV_FIELDS) uv[k] = settings[k];
  return uv;
}

/**
 * Create a layer.
 *
 * @param {object}      opts
 * @param {string}      opts.name
 * @param {object|null} opts.mapEntry   { name, texture, imageData, fullCanvas, width, height, isCustom? }
 * @param {object}      opts.uv         per-layer UV params (see LAYER_UV_FIELDS)
 * @param {Iterable<number>} [opts.faceSet]  original-mesh face indices
 * @param {number}      [opts.colorIndex]    palette slot (defaults by id)
 */
export function createLayer({ name, mapEntry = null, uv, faceSet = [], colorIndex } = {}) {
  const id = nextLayerId();
  const ci = colorIndex ?? ((id - 1) % LAYER_COLORS.length);
  return {
    id,
    name: name || `Layer ${id}`,
    color: LAYER_COLORS[ci],
    colorIndex: ci,
    mapEntry,
    uv: { ...uv },
    faceSet: new Set(faceSet),
  };
}

/** Copy a layer's UV params into the global settings object (in place). */
export function syncSettingsFromLayer(settings, layer) {
  if (!layer) return;
  for (const k of LAYER_UV_FIELDS) {
    if (layer.uv[k] !== undefined) settings[k] = layer.uv[k];
  }
}

/** Copy the per-layer UV params from global settings back into the layer. */
export function syncLayerFromSettings(layer, settings) {
  if (!layer) return;
  for (const k of LAYER_UV_FIELDS) layer.uv[k] = settings[k];
}

/** Find a layer by id within a layers array. */
export function findLayer(layers, id) {
  return layers.find(l => l.id === id) || null;
}

/**
 * Remove `faceIdx` from every layer except `keepId`. Enforces the invariant
 * that a face belongs to at most one layer (disjoint regions).
 */
export function claimFace(layers, faceIdx, keepId) {
  for (const l of layers) {
    if (l.id !== keepId) l.faceSet.delete(faceIdx);
  }
}
