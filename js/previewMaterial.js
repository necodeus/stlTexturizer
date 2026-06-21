import * as THREE from 'three';

// Mapping mode constants (must match index.html <option value="…">)
export const MODE_PLANAR_XY   = 0;
export const MODE_PLANAR_XZ   = 1;
export const MODE_PLANAR_YZ   = 2;
export const MODE_CYLINDRICAL = 3;
export const MODE_SPHERICAL   = 4;
export const MODE_TRIPLANAR   = 5;
export const MODE_CUBIC       = 6;

// Hard cap — must match js/layers.js MAX_LAYERS. Sized to fit two vec4 weight
// attributes and a sampler2D[MAX_LAYERS] array within GLSL1 / texture-unit budget.
export const MAX_LAYERS = 8;

// ── GLSL source ──────────────────────────────────────────────────────────────
//
// Preview strategy, two modes:
//   1. Bump-only (default):  UV projection & bump mapping in the fragment shader.
//   2. Displacement preview: the vertex shader samples the height and moves
//      each vertex along its smooth normal.
//
// Multi-texture: when layerCount > 1 the height at a point is the weighted sum
// of every layer's sampled grey × that layer's amplitude (mirrors the CPU bake
// in displacement.js). The per-vertex layer weights arrive in two vec4
// attributes (layerWeightsA/B = 8 channels). When layerCount <= 1 the original
// single-texture path runs UNCHANGED.

const sharedGLSL = /* glsl */`
  uniform sampler2D displacementMap;
  uniform int       mappingMode;
  uniform vec2      scaleUV;
  uniform float     amplitude;
  uniform vec2      offsetUV;
  uniform float     rotation;
  uniform vec3      boundsMin;
  uniform vec3      boundsSize;
  uniform vec3      boundsCenter;
  uniform vec2      cylinderCenter;
  uniform float     cylinderRadius;
  uniform float     bottomAngleLimit;
  uniform float     topAngleLimit;
  uniform float     mappingBlend;
  uniform float     seamBandWidth;
  uniform float     capAngle;
  uniform int       symmetricDisplacement;
  uniform int       noDownwardZ;
  uniform int       useDisplacement;
  uniform vec2      textureAspect;

  // Multi-texture layers (active when layerCount > 1)
  #define MAX_LAYERS ${MAX_LAYERS}
  uniform int       layerCount;
  uniform sampler2D layerMaps[MAX_LAYERS];
  uniform vec2      layerScale[MAX_LAYERS];
  uniform vec2      layerOffset[MAX_LAYERS];
  uniform float     layerRotation[MAX_LAYERS];
  uniform float     layerAmplitude[MAX_LAYERS];
  uniform vec2      layerAspect[MAX_LAYERS];

  const float PI     = 3.14159265358979;
  const float TWO_PI = 6.28318530717959;
  const float CUBIC_AXIS_EPSILON = 1e-4;

  int dominantCubicAxis(vec3 n) {
    vec3 absN = abs(n);
    if (absN.x >= absN.y - CUBIC_AXIS_EPSILON && absN.x >= absN.z - CUBIC_AXIS_EPSILON) return 0;
    if (absN.y >= absN.z - CUBIC_AXIS_EPSILON) return 1;
    return 2;
  }

  vec3 cubicBlendWeights(vec3 n) {
    vec3 absN = abs(n);
    int axis = dominantCubicAxis(n);
    float primary = axis == 0 ? absN.x : axis == 1 ? absN.y : absN.z;
    float secondary = axis == 0 ? max(absN.y, absN.z)
                    : axis == 1 ? max(absN.x, absN.z)
                                : max(absN.x, absN.y);

    if (mappingBlend < 0.001) {
      if (axis == 0) return vec3(1.0, 0.0, 0.0);
      if (axis == 1) return vec3(0.0, 1.0, 0.0);
      return vec3(0.0, 0.0, 1.0);
    }

    vec3 oneHot = axis == 0 ? vec3(1.0, 0.0, 0.0)
                : axis == 1 ? vec3(0.0, 1.0, 0.0)
                            : vec3(0.0, 0.0, 1.0);

    float seamWidth = max(seamBandWidth, CUBIC_AXIS_EPSILON * 2.0);
    float seamMixRaw = 1.0 - clamp((primary - secondary) / seamWidth, 0.0, 1.0);
    float seamMix = mappingBlend * seamMixRaw * seamMixRaw * (3.0 - 2.0 * seamMixRaw);
    if (seamMix <= 0.001) return oneHot;

    float power = 1.0 + (1.0 - seamMix) * 11.0;
    vec3 softWeights = pow(absN, vec3(power));
    softWeights /= dot(softWeights, vec3(1.0)) + 1e-6;

    vec3 blendedWeights = mix(oneHot, softWeights, seamMix);
    return blendedWeights / (dot(blendedWeights, vec3(1.0)) + 1e-6);
  }

  // ── Parametrized sampler + height (used by BOTH single and multi paths) ─────
  // Passing the indexed sampler/transform in lets the multi path call this once
  // per layer inside a constant-bounded loop (GLSL1 requires the sampler-array
  // index to be a loop index / constant-index-expression — satisfied at the
  // call site layerMaps[k]).
  float sampleMapP(sampler2D map, vec2 rawUV, vec2 sc, vec2 off, float rot, vec2 asp) {
    vec2 uv = (rawUV * asp) / sc + off;
    float c = cos(rot); float s = sin(rot);
    uv -= 0.5;
    uv  = vec2(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
    uv += 0.5;
    return texture2D(map, uv).r;
  }

  float computeHeightP(sampler2D map, vec2 sc, vec2 off, float rot, vec2 asp,
                       vec3 pos, vec3 projN, vec3 blendN) {
    vec3 rel = pos - boundsCenter;
    float maxDim = max(boundsSize.x, max(boundsSize.y, boundsSize.z));
    float md = max(maxDim, 1e-4);

    if (mappingMode == 0) {
      return sampleMapP(map, vec2((pos.x - boundsMin.x) / md, (pos.y - boundsMin.y) / md), sc, off, rot, asp);

    } else if (mappingMode == 1) {
      return sampleMapP(map, vec2((pos.x - boundsMin.x) / md, (pos.z - boundsMin.z) / md), sc, off, rot, asp);

    } else if (mappingMode == 2) {
      return sampleMapP(map, vec2((pos.y - boundsMin.y) / md, (pos.z - boundsMin.z) / md), sc, off, rot, asp);

    } else if (mappingMode == 3) {
      vec2 cylRel2 = pos.xy - cylinderCenter;
      float r = max(cylinderRadius, 1e-4);
      float C = TWO_PI * r;
      float u_cyl = atan(cylRel2.y, cylRel2.x) / TWO_PI + 0.5;
      float v_cyl = (pos.z - boundsMin.z) / C;

      float seamBand = seamBandWidth * 0.1;
      float seamDist = min(u_cyl, 1.0 - u_cyl);
      float hSide;
      if (seamBand > 0.001 && seamDist < seamBand) {
        float d = u_cyl < 0.5 ? u_cyl : u_cyl - 1.0;
        float t = smoothstep(0.0, 1.0, (d + seamBand) / (2.0 * seamBand));
        float hLeft  = sampleMapP(map, vec2(1.0 + d, v_cyl), sc, off, rot, asp);
        float hRight = sampleMapP(map, vec2(d, v_cyl), sc, off, rot, asp);
        hSide = mix(hLeft, hRight, t);
      } else {
        hSide = sampleMapP(map, vec2(u_cyl, v_cyl), sc, off, rot, asp);
      }

      if (mappingBlend < 0.001) return hSide;
      float capThreshold = cos(radians(capAngle));
      float blendHalf = seamBandWidth * 0.5;
      float capW = smoothstep(capThreshold - blendHalf, capThreshold + blendHalf, abs(blendN.z));
      float hCap  = sampleMapP(map, vec2(cylRel2.x / C + 0.5, cylRel2.y / C + 0.5), sc, off, rot, asp);
      return mix(hSide, hCap, capW);

    } else if (mappingMode == 4) {
      float r     = length(rel);
      float phi   = acos(clamp(rel.z / max(r, 1e-4), -1.0, 1.0));
      float u_sph = atan(rel.y, rel.x) / TWO_PI + 0.5;
      float v_sph = phi / PI;

      float seamBand = seamBandWidth * 0.1;
      float seamDist = min(u_sph, 1.0 - u_sph);
      if (seamBand > 0.001 && seamDist < seamBand) {
        float d = u_sph < 0.5 ? u_sph : u_sph - 1.0;
        float t = smoothstep(0.0, 1.0, (d + seamBand) / (2.0 * seamBand));
        float hLeft  = sampleMapP(map, vec2(1.0 + d, v_sph), sc, off, rot, asp);
        float hRight = sampleMapP(map, vec2(d, v_sph), sc, off, rot, asp);
        return mix(hLeft, hRight, t);
      }
      return sampleMapP(map, vec2(u_sph, v_sph), sc, off, rot, asp);

    } else if (mappingMode == 5) {
      vec3 blend = abs(projN);
      blend = pow(blend, vec3(4.0));
      blend /= dot(blend, vec3(1.0)) + 1e-4;
      float yzU = (pos.y - boundsMin.y) / md;
      if (projN.x < 0.0) yzU = -yzU;
      float xzU = (pos.x - boundsMin.x) / md;
      if (projN.y > 0.0) xzU = -xzU;
      float xyU = (pos.x - boundsMin.x) / md;
      if (projN.z < 0.0) xyU = -xyU;
      float hXY = sampleMapP(map, vec2(xyU, (pos.y - boundsMin.y) / md), sc, off, rot, asp);
      float hXZ = sampleMapP(map, vec2(xzU, (pos.z - boundsMin.z) / md), sc, off, rot, asp);
      float hYZ = sampleMapP(map, vec2(yzU, (pos.z - boundsMin.z) / md), sc, off, rot, asp);
      return hXY * blend.z + hXZ * blend.y + hYZ * blend.x;

    } else {
      float yzU = (pos.y - boundsMin.y) / md;
      if (projN.x < 0.0) yzU = -yzU;
      float xzU = (pos.x - boundsMin.x) / md;
      if (projN.y > 0.0) xzU = -xzU;
      float xyU = (pos.x - boundsMin.x) / md;
      if (projN.z < 0.0) xyU = -xyU;
      float hYZ = sampleMapP(map, vec2(yzU, (pos.z - boundsMin.z) / md), sc, off, rot, asp);
      float hXZ = sampleMapP(map, vec2(xzU, (pos.z - boundsMin.z) / md), sc, off, rot, asp);
      float hXY = sampleMapP(map, vec2(xyU, (pos.y - boundsMin.y) / md), sc, off, rot, asp);
      vec3 bN = blendN;
      vec3 absFaceN = abs(projN);
      float facePrimary = max(absFaceN.x, max(absFaceN.y, absFaceN.z));
      float faceSecondary = absFaceN.x + absFaceN.y + absFaceN.z - facePrimary
                          - min(absFaceN.x, min(absFaceN.y, absFaceN.z));
      if (facePrimary - faceSecondary <= CUBIC_AXIS_EPSILON) bN = projN;
      vec3 wts = cubicBlendWeights(bN);
      return hYZ * wts.x + hXZ * wts.y + hXY * wts.z;
    }
  }

  // Single-texture height (raw grey) — global uniforms.
  float computeHeightAtPoint(vec3 pos, vec3 projN, vec3 blendN) {
    return computeHeightP(displacementMap, scaleUV, offsetUV, rotation, textureAspect, pos, projN, blendN);
  }

  // Fetch the k-th weight from the two vec4 attribute halves (GLSL1: no dynamic
  // component indexing, so an explicit ladder).
  float layerWeight(int k, vec4 a, vec4 b) {
    if (k == 0) return a.x; if (k == 1) return a.y; if (k == 2) return a.z; if (k == 3) return a.w;
    if (k == 4) return b.x; if (k == 5) return b.y; if (k == 6) return b.z; return b.w;
  }

  // Sampler arrays must be indexed by a CONSTANT integral expression (GLSL ES
  // 1.00 / ANGLE). A loop index does not qualify on strict drivers, so dispatch
  // to a constant index here. Non-sampler uniform arrays could be variable-
  // indexed, but we keep them constant too for a single clean ladder.
  float heightForLayer(int k, vec3 pos, vec3 projN, vec3 blendN) {
    if (k == 0) return computeHeightP(layerMaps[0], layerScale[0], layerOffset[0], layerRotation[0], layerAspect[0], pos, projN, blendN);
    if (k == 1) return computeHeightP(layerMaps[1], layerScale[1], layerOffset[1], layerRotation[1], layerAspect[1], pos, projN, blendN);
    if (k == 2) return computeHeightP(layerMaps[2], layerScale[2], layerOffset[2], layerRotation[2], layerAspect[2], pos, projN, blendN);
    if (k == 3) return computeHeightP(layerMaps[3], layerScale[3], layerOffset[3], layerRotation[3], layerAspect[3], pos, projN, blendN);
    if (k == 4) return computeHeightP(layerMaps[4], layerScale[4], layerOffset[4], layerRotation[4], layerAspect[4], pos, projN, blendN);
    if (k == 5) return computeHeightP(layerMaps[5], layerScale[5], layerOffset[5], layerRotation[5], layerAspect[5], pos, projN, blendN);
    if (k == 6) return computeHeightP(layerMaps[6], layerScale[6], layerOffset[6], layerRotation[6], layerAspect[6], pos, projN, blendN);
    return computeHeightP(layerMaps[7], layerScale[7], layerOffset[7], layerRotation[7], layerAspect[7], pos, projN, blendN);
  }

  // Final displacement scalar at a point (symmetric centring + amplitude folded
  // in). Single path: one sample × amplitude. Multi path: normalised weighted
  // sum of each layer's grey × its amplitude — matches displacement.js Pass 3.
  float displacementScalar(vec3 pos, vec3 projN, vec3 blendN, vec4 lwA, vec4 lwB) {
    // layerCount < 1 only when no layer data is bound (legacy single-texture);
    // the app always binds layers now, so the weighted path runs even for one
    // layer — unpainted faces (all-zero weights) yield zero displacement.
    if (layerCount < 1) {
      float h = computeHeightAtPoint(pos, projN, blendN);
      if (symmetricDisplacement == 1) h -= 0.5;
      return h * amplitude;
    }
    float wsum = 0.0;
    for (int k = 0; k < MAX_LAYERS; k++) { if (k >= layerCount) break; wsum += layerWeight(k, lwA, lwB); }
    if (wsum < 1e-6) return 0.0;
    float acc = 0.0;
    for (int k = 0; k < MAX_LAYERS; k++) {
      if (k >= layerCount) break;
      float wk = layerWeight(k, lwA, lwB);
      if (wk <= 0.0) continue;
      float hk = heightForLayer(k, pos, projN, blendN);
      if (symmetricDisplacement == 1) hk -= 0.5;
      acc += (wk / wsum) * hk * layerAmplitude[k];
    }
    return acc;
  }
`;

const vertexShader = /* glsl */`
  precision highp float;
  ${sharedGLSL}

  attribute vec3  smoothNormal;
  attribute vec3  faceNormal;
  attribute float faceMask;
  attribute float boundaryFalloffAttr;
  attribute float boundaryMaskTypeAttr;
  attribute vec4  layerWeightsA;
  attribute vec4  layerWeightsB;

  varying vec3  vModelPos;
  varying vec3  vModelNormal;
  varying vec3  vViewPos;
  varying vec3  vNormal;
  varying vec3  vSmoothNormal;
  varying float vFaceMask;
  varying float vUserMask;
  varying float vMaskType;
  varying vec4  vLayerWeightsA;
  varying vec4  vLayerWeightsB;

  void main() {
    vec3 safeN = length(normal) > 1e-6 ? normalize(normal) : vec3(0.0, 0.0, 1.0);
    vec3 fN = length(faceNormal) > 1e-6 ? normalize(faceNormal) : safeN;
    vec3 pos = position;

    float surfaceAngle = degrees(acos(clamp(abs(fN.z), 0.0, 1.0)));
    float angleMask = 1.0;
    if (fN.z <  0.0 && bottomAngleLimit >= 1.0)
      angleMask = min(angleMask, surfaceAngle > bottomAngleLimit ? 1.0 : 0.0);
    if (fN.z >= 0.0 && topAngleLimit >= 1.0)
      angleMask = min(angleMask, surfaceAngle > topAngleLimit ? 1.0 : 0.0);
    float totalMask = angleMask * faceMask * boundaryFalloffAttr;
    vFaceMask = totalMask;
    vUserMask = faceMask;
    vMaskType = boundaryMaskTypeAttr;
    vLayerWeightsA = layerWeightsA;
    vLayerWeightsB = layerWeightsB;

    if (useDisplacement == 1) {
      // displacementScalar already folds symmetric centring + per-layer amplitude.
      float d = displacementScalar(position, safeN, safeN, layerWeightsA, layerWeightsB);
      d *= totalMask;

      vec3 sN = length(smoothNormal) > 1e-6 ? normalize(smoothNormal) : safeN;
      pos = position + sN * d;
      if (noDownwardZ == 1 && pos.z < position.z) pos.z = position.z;
    }

    vModelPos    = position;
    vModelNormal = fN;
    vec4 mvPos   = modelViewMatrix * vec4(pos, 1.0);
    vViewPos     = mvPos.xyz;
    vNormal      = normalize(normalMatrix * fN);
    vec3 sN = length(smoothNormal) > 1e-6 ? normalize(smoothNormal) : safeN;
    vSmoothNormal = normalize(normalMatrix * sN);
    gl_Position  = projectionMatrix * mvPos;
  }
`;

const fragmentShader = /* glsl */`
  precision highp float;
  ${sharedGLSL}

  uniform sampler2D boundaryEdgeTex;
  uniform int       boundaryEdgeCount;
  uniform float     boundaryEdgeTexWidth;
  uniform float     boundaryFalloffDist;

  varying vec3  vModelPos;
  varying vec3  vModelNormal;
  varying vec3  vViewPos;
  varying vec3  vNormal;
  varying vec3  vSmoothNormal;
  varying float vFaceMask;
  varying float vUserMask;
  varying float vMaskType;
  varying vec4  vLayerWeightsA;
  varying vec4  vLayerWeightsB;

  // Face-stable projection normal via screen-space derivatives.
  vec3 projNormal() {
    vec3 _fN = cross(dFdx(vModelPos), dFdy(vModelPos));
    return length(_fN) > 1e-10 ? normalize(_fN) : vModelNormal;
  }

  void main() {
    vec3 N = normalize(vNormal) * (gl_FrontFacing ? 1.0 : -1.0);
    vec3 PN = projNormal();

    // Height field + bump strength.  Single path keeps the original behaviour
    // (raw grey, amplitude applied via bumpStr).  Multi path uses the blended
    // scalar (amplitude already folded) so bumpStr drops the amplitude factor.
    float h;
    float bumpAmp;
    if (layerCount >= 1) {
      h = displacementScalar(vModelPos, PN, vModelNormal, vLayerWeightsA, vLayerWeightsB);
      bumpAmp = 1.0;
    } else {
      h = computeHeightAtPoint(vModelPos, PN, vModelNormal);
      if (symmetricDisplacement == 1) h = h - 0.5;
      bumpAmp = amplitude;
    }

    float dhx = dFdx(h);
    float dhy = dFdy(h);

    float maskBlend = vFaceMask;

    if (useDisplacement == 0 && boundaryFalloffDist > 0.001 && boundaryEdgeCount > 0) {
      float minDist = boundaryFalloffDist;
      for (int i = 0; i < 64; i++) {
        if (i >= boundaryEdgeCount) break;
        float uA = (float(i * 2) + 0.5) / boundaryEdgeTexWidth;
        float uB = (float(i * 2 + 1) + 0.5) / boundaryEdgeTexWidth;
        vec3 ea = texture2D(boundaryEdgeTex, vec2(uA, 0.5)).xyz;
        vec3 eb = texture2D(boundaryEdgeTex, vec2(uB, 0.5)).xyz;
        vec3 ab = eb - ea;
        float abLen2 = dot(ab, ab);
        float t = clamp(dot(vModelPos - ea, ab) / max(abLen2, 1e-10), 0.0, 1.0);
        float d = length(vModelPos - (ea + t * ab));
        if (d < minDist) { minDist = d; if (d < 1e-4) break; }
      }
      maskBlend *= clamp(minDist / boundaryFalloffDist, 0.0, 1.0);
    }

    h *= maskBlend;
    dhx *= maskBlend;
    dhy *= maskBlend;

    vec3 dp1 = dFdx(vViewPos);
    vec3 dp2 = dFdy(vViewPos);

    vec3 T = dp1 - dot(dp1, N) * N;
    vec3 B = dp2 - dot(dp2, N) * N;
    float lenT = length(T);
    float lenB = length(B);
    T = lenT > 1e-5 ? T / lenT : vec3(1.0, 0.0, 0.0);
    B = lenB > 1e-5 ? B / lenB : vec3(0.0, 1.0, 0.0);

    float posScale = max(length(dp1) + length(dp2), 1e-6);
    float bumpStr  = useDisplacement == 1
      ? bumpAmp * 2.0 / posScale
      : bumpAmp * 6.0 / posScale;

    vec3 bumpVec = N - bumpStr * (dhx * T + dhy * B);
    vec3 bumpN = length(bumpVec) > 1e-6 ? normalize(bumpVec) : N;

    vec3 smoothN = normalize(vSmoothNormal) * (gl_FrontFacing ? 1.0 : -1.0);
    bumpN = mix(smoothN, bumpN, maskBlend);

    vec3 tealBase      = vec3(0.22, 0.68, 0.68);
    vec3 userMaskColor = vec3(0.85, 0.40, 0.15);
    vec3 angleMaskColor = vec3(0.45, 0.48, 0.50);

    vec3 L1 = normalize(vec3( 0.5,  0.8,  1.0));
    vec3 L2 = normalize(vec3(-0.5, -0.2, -0.6));
    vec3 V  = normalize(-vViewPos);

    float diff1 = max(dot(bumpN, L1), 0.0);
    float diff2 = max(dot(bumpN, L2), 0.0) * 0.35;

    vec3 H1   = normalize(L1 + V);
    float spec = pow(max(dot(bumpN, H1), 0.0), 64.0) * 0.60;

    vec3 litTeal = tealBase * 0.55
                 + tealBase * diff1 * vec3(1.00, 0.96, 0.88) * 0.55
                 + tealBase * diff2 * vec3(0.80, 0.60, 0.50) * 0.15
                 + vec3(spec);

    float maskEffect = 1.0 - maskBlend;
    float effectiveMaskType = mix(vMaskType, 0.0, step(0.5, 1.0 - vUserMask));
    vec3 maskBase = mix(userMaskColor, angleMaskColor, effectiveMaskType);
    vec3 litMask = maskBase * 0.55
                 + maskBase * diff1 * vec3(1.00, 0.96, 0.88) * 0.55
                 + maskBase * diff2 * vec3(0.80, 0.60, 0.50) * 0.15
                 + vec3(spec);

    vec3 color = mix(litTeal, litMask, maskEffect);

    gl_FragColor = vec4(color, 1.0);
  }
`;

// ── Public API ────────────────────────────────────────────────────────────────

let _fallbackTex = null;

/**
 * Create a ShaderMaterial for the displacement preview.
 * @param {THREE.Texture|null} displacementTexture
 * @param {object} settings
 */
export function createPreviewMaterial(displacementTexture, settings) {
  const mat = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: buildUniforms(displacementTexture, settings),
    side: THREE.DoubleSide,
  });
  return mat;
}

/**
 * Update existing ShaderMaterial uniforms in-place (no recreate).
 *
 * @param {object} [layerUniforms] optional multi-texture data:
 *   { count, maps:[THREE.Texture], scale:[[u,v]], offset:[[u,v]], rotation:[rad],
 *     amplitude:[n], aspect:[[u,v]] }. When omitted or count<=1, the single-
 *     texture path is used.
 */
export function updateMaterial(material, displacementTexture, settings, layerUniforms = null) {
  const u = material.uniforms;
  if (displacementTexture && u.displacementMap.value !== displacementTexture) {
    u.displacementMap.value = displacementTexture;
  }
  u.mappingMode.value   = settings.mappingMode;
  u.scaleUV.value.set(settings.scaleU, settings.scaleV);
  u.amplitude.value     = settings.amplitude;
  u.offsetUV.value.set(settings.offsetU, settings.offsetV);
  u.rotation.value      = (settings.rotation ?? 0) * Math.PI / 180;
  if (settings.bounds) {
    u.boundsMin.value.copy(settings.bounds.min);
    u.boundsSize.value.copy(settings.bounds.size);
    u.boundsCenter.value.copy(settings.bounds.center);
    const cx = settings.cylinderCenterX ?? settings.bounds.center.x;
    const cy = settings.cylinderCenterY ?? settings.bounds.center.y;
    const cr = settings.cylinderRadius
      ?? Math.max(settings.bounds.size.x, settings.bounds.size.y) * 0.5;
    u.cylinderCenter.value.set(cx, cy);
    u.cylinderRadius.value = cr;
  }
  u.bottomAngleLimit.value = settings.bottomAngleLimit ?? 5.0;
  u.topAngleLimit.value    = settings.topAngleLimit    ?? 0.0;
  u.mappingBlend.value            = settings.mappingBlend            ?? 0.0;
  u.seamBandWidth.value           = settings.seamBandWidth           ?? 0.35;
  u.capAngle.value                = settings.capAngle                ?? 20.0;
  u.symmetricDisplacement.value   = settings.symmetricDisplacement   ? 1 : 0;
  u.noDownwardZ.value             = settings.noDownwardZ             ? 1 : 0;
  u.useDisplacement.value         = settings.useDisplacement         ? 1 : 0;
  u.textureAspect.value.set(settings.textureAspectU ?? 1, settings.textureAspectV ?? 1);
  u.boundaryFalloffDist.value       = settings.boundaryFalloff           ?? 0.0;

  applyLayerUniforms(u, layerUniforms);
}

/** Write the per-layer uniform arrays (or set layerCount<=1 for the single path). */
export function applyLayerUniforms(u, layerUniforms) {
  if (!layerUniforms || !layerUniforms.count || layerUniforms.count < 1) {
    // No layer data → legacy single-texture path (shader uses layerCount < 1).
    u.layerCount.value = 0;
    return;
  }
  const n = Math.min(layerUniforms.count, MAX_LAYERS);
  u.layerCount.value = n;
  const fb = getFallbackTexture();
  for (let k = 0; k < MAX_LAYERS; k++) {
    u.layerMaps.value[k]      = (k < n && layerUniforms.maps[k]) ? layerUniforms.maps[k] : fb;
    u.layerScale.value[k].set(layerUniforms.scale?.[k]?.[0] ?? 1, layerUniforms.scale?.[k]?.[1] ?? 1);
    u.layerOffset.value[k].set(layerUniforms.offset?.[k]?.[0] ?? 0, layerUniforms.offset?.[k]?.[1] ?? 0);
    u.layerRotation.value[k]  = layerUniforms.rotation?.[k] ?? 0;
    u.layerAmplitude.value[k] = layerUniforms.amplitude?.[k] ?? 0;
    u.layerAspect.value[k].set(layerUniforms.aspect?.[k]?.[0] ?? 1, layerUniforms.aspect?.[k]?.[1] ?? 1);
  }
}

// ── Internal ──────────────────────────────────────────────────────────────────

function buildUniforms(tex, settings) {
  const b = settings.bounds || {
    min:    new THREE.Vector3(),
    size:   new THREE.Vector3(1, 1, 1),
    center: new THREE.Vector3(),
  };
  const fb = getFallbackTexture();
  return {
    displacementMap: { value: tex || fb },
    mappingMode:     { value: settings.mappingMode ?? MODE_TRIPLANAR },
    scaleUV:         { value: new THREE.Vector2(settings.scaleU ?? 1, settings.scaleV ?? 1) },
    amplitude:       { value: settings.amplitude ?? 1.0 },
    offsetUV:        { value: new THREE.Vector2(settings.offsetU ?? 0, settings.offsetV ?? 0) },
    rotation:        { value: ((settings.rotation ?? 0) * Math.PI / 180) },
    boundsMin:        { value: b.min.clone() },
    boundsSize:       { value: b.size.clone() },
    boundsCenter:     { value: b.center.clone() },
    cylinderCenter:   { value: new THREE.Vector2(
                          settings.cylinderCenterX ?? b.center.x,
                          settings.cylinderCenterY ?? b.center.y) },
    cylinderRadius:   { value: settings.cylinderRadius
                          ?? Math.max(b.size.x, b.size.y) * 0.5 },
    bottomAngleLimit: { value: settings.bottomAngleLimit ?? 5.0 },
    topAngleLimit:    { value: settings.topAngleLimit    ?? 0.0 },
    mappingBlend:             { value: settings.mappingBlend            ?? 0.0 },
    seamBandWidth:            { value: settings.seamBandWidth            ?? 0.35 },
    capAngle:                 { value: settings.capAngle                 ?? 20.0 },
    symmetricDisplacement:    { value: settings.symmetricDisplacement   ? 1 : 0 },
    noDownwardZ:              { value: settings.noDownwardZ             ? 1 : 0 },
    useDisplacement:          { value: settings.useDisplacement         ? 1 : 0 },
    textureAspect:            { value: new THREE.Vector2(settings.textureAspectU ?? 1, settings.textureAspectV ?? 1) },
    // Multi-texture layer uniforms (layerCount<1 → legacy single-texture path
    // until applyLayerUniforms binds real layers)
    layerCount:      { value: 0 },
    layerMaps:       { value: Array.from({ length: MAX_LAYERS }, () => fb) },
    layerScale:      { value: Array.from({ length: MAX_LAYERS }, () => new THREE.Vector2(1, 1)) },
    layerOffset:     { value: Array.from({ length: MAX_LAYERS }, () => new THREE.Vector2(0, 0)) },
    layerRotation:   { value: new Array(MAX_LAYERS).fill(0) },
    layerAmplitude:  { value: new Array(MAX_LAYERS).fill(0) },
    layerAspect:     { value: Array.from({ length: MAX_LAYERS }, () => new THREE.Vector2(1, 1)) },
    boundaryEdgeTex:          { value: createFallbackDataTexture() },
    boundaryEdgeCount:        { value: 0 },
    boundaryEdgeTexWidth:     { value: 1.0 },
    boundaryFalloffDist:        { value: settings.boundaryFalloff ?? 0.0 },
  };
}

function getFallbackTexture() {
  if (_fallbackTex) return _fallbackTex;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 4;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, 4, 4);
  _fallbackTex = new THREE.CanvasTexture(canvas);
  _fallbackTex.wrapS = _fallbackTex.wrapT = THREE.RepeatWrapping;
  return _fallbackTex;
}

function createFallbackDataTexture() {
  const data = new Float32Array(4);
  const t = new THREE.DataTexture(data, 1, 1, THREE.RGBAFormat, THREE.FloatType);
  t.minFilter = THREE.NearestFilter;
  t.magFilter = THREE.NearestFilter;
  t.needsUpdate = true;
  return t;
}
