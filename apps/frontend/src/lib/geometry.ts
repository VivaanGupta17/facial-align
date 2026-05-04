/**
 * 3D geometry utilities for the Facial Align surgical planning platform.
 *
 * All coordinates follow two conventions:
 *  - DICOM LPS (Left-Posterior-Superior): used in medical data
 *  - Three.js RHS (Right-Hand, Y-up): used in the 3D viewer
 *
 * Matrices are stored in column-major order (matching WebGL / THREE.Matrix4).
 */

import type { Vector3, BoundingBox, Transform3D } from '../types/medical'

// =============================================================================
// Type aliases
// =============================================================================

/** Column-major 4×4 matrix — 16 floats */
export type Mat4 = [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
]

/** 3-component vector tuple */
export type Vec3 = [number, number, number]

// =============================================================================
// Vector3 helpers
// =============================================================================

export function vec3(x: number, y: number, z: number): Vec3 {
  return [x, y, z]
}

export function vec3FromMedical(v: Vector3): Vec3 {
  return [v.x, v.y, v.z]
}

export function vec3ToMedical(v: Vec3): Vector3 {
  return { x: v[0], y: v[1], z: v[2] }
}

export function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

export function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

export function vec3Scale(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s]
}

export function vec3Dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

export function vec3Length(v: Vec3): number {
  return Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
}

export function vec3Normalize(v: Vec3): Vec3 {
  const len = vec3Length(v)
  if (len < 1e-10) return [0, 0, 0]
  return vec3Scale(v, 1 / len)
}

export function vec3Lerp(a: Vec3, b: Vec3, t: number): Vec3 {
  const s = 1 - t
  return [s * a[0] + t * b[0], s * a[1] + t * b[1], s * a[2] + t * b[2]]
}

// =============================================================================
// Matrix4×4 utilities (column-major)
// =============================================================================

export function mat4Identity(): Mat4 {
  return [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]
}

/**
 * Multiply two 4×4 column-major matrices: result = a * b
 */
export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Array<number>(16) as unknown as Mat4
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0
      for (let k = 0; k < 4; k++) {
        sum += a[k * 4 + row] * b[col * 4 + k]
      }
      out[col * 4 + row] = sum
    }
  }
  return out
}

/**
 * Build a 4×4 translation matrix from a translation vector.
 */
export function mat4Translation(tx: number, ty: number, tz: number): Mat4 {
  return [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    tx, ty, tz, 1,
  ]
}

/**
 * Build a rotation matrix from Euler angles (degrees, XYZ intrinsic order).
 */
export function mat4FromEulerXYZ(xDeg: number, yDeg: number, zDeg: number): Mat4 {
  const toRad = Math.PI / 180
  const cx = Math.cos(xDeg * toRad), sx = Math.sin(xDeg * toRad)
  const cy = Math.cos(yDeg * toRad), sy = Math.sin(yDeg * toRad)
  const cz = Math.cos(zDeg * toRad), sz = Math.sin(zDeg * toRad)

  // Rz * Ry * Rx — column-major
  return [
    cy * cz,                          cy * sz,                          -sy,      0,
    sx * sy * cz - cx * sz,           sx * sy * sz + cx * cz,           sx * cy,  0,
    cx * sy * cz + sx * sz,           cx * sy * sz - sx * cz,           cx * cy,  0,
    0,                                0,                                0,        1,
  ]
}

/**
 * Build a scale matrix.
 */
export function mat4Scale(sx: number, sy: number, sz: number): Mat4 {
  return [
    sx, 0,  0,  0,
    0,  sy, 0,  0,
    0,  0,  sz, 0,
    0,  0,  0,  1,
  ]
}

/**
 * Compose a rigid-body transform (scale → rotate → translate) from a Transform3D.
 */
export function mat4FromTransform3D(t: Transform3D): Mat4 {
  const S = mat4Scale(t.scale.x, t.scale.y, t.scale.z)
  const R = mat4FromEulerXYZ(t.rotation.x, t.rotation.y, t.rotation.z)
  const T = mat4Translation(t.translation.x, t.translation.y, t.translation.z)
  return mat4Multiply(T, mat4Multiply(R, S))
}

/**
 * Transform a 3D point by a 4×4 column-major matrix.
 */
export function mat4TransformPoint(m: Mat4, p: Vec3): Vec3 {
  const x = p[0], y = p[1], z = p[2]
  const w = m[3] * x + m[7] * y + m[11] * z + m[15]
  const invW = w !== 0 ? 1 / w : 1
  return [
    (m[0] * x + m[4] * y + m[8]  * z + m[12]) * invW,
    (m[1] * x + m[5] * y + m[9]  * z + m[13]) * invW,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) * invW,
  ]
}

/**
 * Compute the inverse of a 4×4 column-major matrix.
 * Uses the general cofactor/adjugate method.
 * Returns identity if the matrix is singular.
 */
export function mat4Inverse(m: Mat4): Mat4 {
  const [
    m00, m01, m02, m03,
    m10, m11, m12, m13,
    m20, m21, m22, m23,
    m30, m31, m32, m33,
  ] = m

  const b00 = m00 * m11 - m01 * m10
  const b01 = m00 * m12 - m02 * m10
  const b02 = m00 * m13 - m03 * m10
  const b03 = m01 * m12 - m02 * m11
  const b04 = m01 * m13 - m03 * m11
  const b05 = m02 * m13 - m03 * m12
  const b06 = m20 * m31 - m21 * m30
  const b07 = m20 * m32 - m22 * m30
  const b08 = m20 * m33 - m23 * m30
  const b09 = m21 * m32 - m22 * m31
  const b10 = m21 * m33 - m23 * m31
  const b11 = m22 * m33 - m23 * m32

  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
  if (Math.abs(det) < 1e-14) return mat4Identity()
  const inv = 1 / det

  return [
    (m11 * b11 - m12 * b10 + m13 * b09) * inv,
    (m02 * b10 - m01 * b11 - m03 * b09) * inv,
    (m31 * b05 - m32 * b04 + m33 * b03) * inv,
    (m22 * b04 - m21 * b05 - m23 * b03) * inv,
    (m12 * b08 - m10 * b11 - m13 * b07) * inv,
    (m00 * b11 - m02 * b08 + m03 * b07) * inv,
    (m32 * b02 - m30 * b05 - m33 * b01) * inv,
    (m20 * b05 - m22 * b02 + m23 * b01) * inv,
    (m10 * b10 - m11 * b08 + m13 * b06) * inv,
    (m01 * b08 - m00 * b10 - m03 * b06) * inv,
    (m30 * b04 - m31 * b02 + m33 * b00) * inv,
    (m21 * b02 - m20 * b04 - m23 * b00) * inv,
    (m11 * b07 - m10 * b09 - m12 * b06) * inv,
    (m00 * b09 - m01 * b07 + m02 * b06) * inv,
    (m31 * b01 - m30 * b03 - m32 * b00) * inv,
    (m20 * b03 - m21 * b01 + m22 * b00) * inv,
  ]
}

// =============================================================================
// Euler Angles ↔ Rotation Matrix
// =============================================================================

/**
 * Extract XYZ Euler angles (degrees) from the upper-left 3×3 of a column-major Mat4.
 * Assumes the matrix encodes Rz * Ry * Rx order.
 */
export function mat4ToEulerXYZ(m: Mat4): Vector3 {
  // m[2]  = -sin(y)
  // m[6]  = sin(x)*cos(y)
  // m[10] = cos(x)*cos(y)
  const sy = -m[2]
  const cosY = Math.sqrt(1 - sy * sy)
  const toDeg = 180 / Math.PI

  if (cosY < 1e-6) {
    // Gimbal lock: y = ±90°
    return {
      x: Math.atan2(-m[9], m[5]) * toDeg,
      y: Math.atan2(-m[2], cosY) * toDeg,
      z: 0,
    }
  }

  return {
    x: Math.atan2(m[6], m[10]) * toDeg,
    y: Math.asin(Math.max(-1, Math.min(1, -m[2]))) * toDeg,
    z: Math.atan2(m[1], m[0]) * toDeg,
  }
}

// =============================================================================
// Transform Interpolation
// =============================================================================

/**
 * Interpolate between two Transform3D values (for smooth animation).
 * Uses linear interpolation for translation/scale and shortest-path
 * interpolation for Euler angles.
 */
export function lerpTransform3D(a: Transform3D, b: Transform3D, t: number): Transform3D {
  return {
    translation: {
      x: lerp(a.translation.x, b.translation.x, t),
      y: lerp(a.translation.y, b.translation.y, t),
      z: lerp(a.translation.z, b.translation.z, t),
    },
    rotation: {
      x: lerpAngleDeg(a.rotation.x, b.rotation.x, t),
      y: lerpAngleDeg(a.rotation.y, b.rotation.y, t),
      z: lerpAngleDeg(a.rotation.z, b.rotation.z, t),
    },
    scale: {
      x: lerp(a.scale.x, b.scale.x, t),
      y: lerp(a.scale.y, b.scale.y, t),
      z: lerp(a.scale.z, b.scale.z, t),
    },
  }
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/** Lerp angles taking the shortest path through ±180°. */
function lerpAngleDeg(a: number, b: number, t: number): number {
  const diff = ((b - a) % 360 + 540) % 360 - 180
  return a + diff * t
}

// =============================================================================
// Distance Calculations
// =============================================================================

/** Euclidean distance between two points. */
export function pointToPointDistance(a: Vector3, b: Vector3): number {
  return Math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)
}

/**
 * Signed distance from a point P to a plane defined by (normal, d),
 * where plane equation is: normal · x = d.
 */
export function pointToPlaneDistance(point: Vector3, planeNormal: Vector3, planeD: number): number {
  const len = Math.sqrt(planeNormal.x ** 2 + planeNormal.y ** 2 + planeNormal.z ** 2)
  if (len < 1e-10) return 0
  return (planeNormal.x * point.x + planeNormal.y * point.y + planeNormal.z * point.z - planeD) / len
}

/**
 * Angle (degrees) between vectors BA and BC at vertex B.
 */
export function angleBetweenPoints(a: Vector3, b: Vector3, c: Vector3): number {
  const ba: Vec3 = [a.x - b.x, a.y - b.y, a.z - b.z]
  const bc: Vec3 = [c.x - b.x, c.y - b.y, c.z - b.z]
  const dot = vec3Dot(ba, bc)
  const lenBA = vec3Length(ba)
  const lenBC = vec3Length(bc)
  if (lenBA < 1e-10 || lenBC < 1e-10) return 0
  const cosA = Math.max(-1, Math.min(1, dot / (lenBA * lenBC)))
  return Math.acos(cosA) * (180 / Math.PI)
}

// =============================================================================
// Centroid & Bounding Box
// =============================================================================

/** Compute the centroid of an array of points. */
export function computeCentroid(points: Vector3[]): Vector3 {
  if (points.length === 0) return { x: 0, y: 0, z: 0 }
  let sx = 0, sy = 0, sz = 0
  for (const p of points) { sx += p.x; sy += p.y; sz += p.z }
  const n = points.length
  return { x: sx / n, y: sy / n, z: sz / n }
}

/** Compute an axis-aligned bounding box from an array of points. */
export function computeBoundingBox(points: Vector3[]): BoundingBox {
  if (points.length === 0) {
    const zero: Vector3 = { x: 0, y: 0, z: 0 }
    return { min: zero, max: zero, center: zero }
  }

  let minX = Infinity, minY = Infinity, minZ = Infinity
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity

  for (const p of points) {
    if (p.x < minX) minX = p.x
    if (p.y < minY) minY = p.y
    if (p.z < minZ) minZ = p.z
    if (p.x > maxX) maxX = p.x
    if (p.y > maxY) maxY = p.y
    if (p.z > maxZ) maxZ = p.z
  }

  return {
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
    center: { x: (minX + maxX) / 2, y: (minY + maxY) / 2, z: (minZ + maxZ) / 2 },
  }
}

/** Check whether two bounding boxes intersect (inclusive). */
export function boundingBoxIntersects(a: BoundingBox, b: BoundingBox): boolean {
  return (
    a.min.x <= b.max.x && a.max.x >= b.min.x &&
    a.min.y <= b.max.y && a.max.y >= b.min.y &&
    a.min.z <= b.max.z && a.max.z >= b.min.z
  )
}

/** Expand a bounding box by a uniform margin. */
export function expandBoundingBox(bb: BoundingBox, margin: number): BoundingBox {
  return {
    min: { x: bb.min.x - margin, y: bb.min.y - margin, z: bb.min.z - margin },
    max: { x: bb.max.x + margin, y: bb.max.y + margin, z: bb.max.z + margin },
    center: { ...bb.center },
  }
}

/** Diagonal length of a bounding box. */
export function boundingBoxDiagonal(bb: BoundingBox): number {
  return Math.sqrt(
    (bb.max.x - bb.min.x) ** 2 +
    (bb.max.y - bb.min.y) ** 2 +
    (bb.max.z - bb.min.z) ** 2
  )
}

// =============================================================================
// Coordinate System Conversions
// =============================================================================

/**
 * DICOM LPS → Three.js coordinate system.
 *
 * DICOM uses Left-Posterior-Superior (LPS):
 *   +X = patient's left
 *   +Y = posterior (towards back)
 *   +Z = superior (towards head)
 *
 * Three.js uses Right-Hand Y-up:
 *   +X = right
 *   +Y = up
 *   +Z = towards viewer (out of screen)
 *
 * Conversion: x_threejs = -x_lps, y_threejs = z_lps, z_threejs = -y_lps
 */
export function lpsToThreeJs(lps: Vector3): Vector3 {
  return {
    x: -lps.x,
    y: lps.z,
    z: -lps.y,
  }
}

/** Inverse: Three.js → DICOM LPS */
export function threeJsToLps(v: Vector3): Vector3 {
  return {
    x: -v.x,
    y: -v.z,
    z: v.y,
  }
}

/**
 * Build the 4×4 LPS-to-Three.js conversion matrix (column-major).
 *
 * x' = -x,  y' = z,  z' = -y
 */
export function lpsToThreeJsMatrix(): Mat4 {
  return [
    -1, 0,  0, 0,
     0, 0, -1, 0,
     0, 1,  0, 0,
     0, 0,  0, 1,
  ]
}

// =============================================================================
// Cross-section plane helpers
// =============================================================================

export type SlicePlane = 'axial' | 'coronal' | 'sagittal'

/**
 * Get the world-space normal vector for each orthogonal CT slice plane
 * in Three.js coordinate space.
 */
export function getSlicePlaneNormal(plane: SlicePlane): Vector3 {
  switch (plane) {
    case 'axial':    return { x: 0, y: 1, z: 0 }  // horizontal (superior/inferior)
    case 'coronal':  return { x: 0, y: 0, z: 1 }  // front/back (anterior/posterior)
    case 'sagittal': return { x: 1, y: 0, z: 0 }  // left/right
  }
}

/**
 * Map a normalised slice position [0..1] to a world-space plane offset,
 * given the bounding box of the volume.
 */
export function slicePositionToWorld(
  t: number,
  plane: SlicePlane,
  bb: BoundingBox
): number {
  switch (plane) {
    case 'axial':    return bb.min.y + t * (bb.max.y - bb.min.y)
    case 'coronal':  return bb.min.z + t * (bb.max.z - bb.min.z)
    case 'sagittal': return bb.min.x + t * (bb.max.x - bb.min.x)
  }
}

// =============================================================================
// Quaternion helpers (for smooth rotation interpolation)
// =============================================================================

export type Quat = [number, number, number, number] // [x, y, z, w]

export function quatFromEulerXYZ(xDeg: number, yDeg: number, zDeg: number): Quat {
  const toRad = Math.PI / 180 / 2
  const cx = Math.cos(xDeg * toRad), sx = Math.sin(xDeg * toRad)
  const cy = Math.cos(yDeg * toRad), sy = Math.sin(yDeg * toRad)
  const cz = Math.cos(zDeg * toRad), sz = Math.sin(zDeg * toRad)

  return [
    sx * cy * cz - cx * sy * sz,
    cx * sy * cz + sx * cy * sz,
    cx * cy * sz - sx * sy * cz,
    cx * cy * cz + sx * sy * sz,
  ]
}

/** Spherical linear interpolation between two quaternions. */
export function quatSlerp(a: Quat, b: Quat, t: number): Quat {
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

  // Ensure shortest path
  let b2: Quat = [...b] as Quat
  if (dot < 0) { b2 = [-b[0], -b[1], -b[2], -b[3]]; dot = -dot }

  if (dot > 0.9995) {
    // Numerically stable linear interpolation for nearly identical quaternions
    const r: Quat = [
      a[0] + t * (b2[0] - a[0]),
      a[1] + t * (b2[1] - a[1]),
      a[2] + t * (b2[2] - a[2]),
      a[3] + t * (b2[3] - a[3]),
    ]
    const len = Math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2)
    return [r[0] / len, r[1] / len, r[2] / len, r[3] / len]
  }

  const theta0 = Math.acos(dot)
  const theta = theta0 * t
  const sinTheta = Math.sin(theta)
  const sinTheta0 = Math.sin(theta0)
  const s0 = Math.cos(theta) - dot * sinTheta / sinTheta0
  const s1 = sinTheta / sinTheta0

  return [
    s0 * a[0] + s1 * b2[0],
    s0 * a[1] + s1 * b2[1],
    s0 * a[2] + s1 * b2[2],
    s0 * a[3] + s1 * b2[3],
  ]
}

/** Convert quaternion to Euler XYZ angles (degrees). */
export function quatToEulerXYZ(q: Quat): Vector3 {
  const [x, y, z, w] = q
  const toDeg = 180 / Math.PI

  const sinRCosP = 2 * (w * x + y * z)
  const cosRCosP = 1 - 2 * (x * x + y * y)
  const rx = Math.atan2(sinRCosP, cosRCosP)

  const sinP = 2 * (w * y - z * x)
  const ry = Math.abs(sinP) >= 1
    ? Math.sign(sinP) * Math.PI / 2
    : Math.asin(sinP)

  const sinYCosP = 2 * (w * z + x * y)
  const cosYCosP = 1 - 2 * (y * y + z * z)
  const rz = Math.atan2(sinYCosP, cosYCosP)

  return { x: rx * toDeg, y: ry * toDeg, z: rz * toDeg }
}

/**
 * Smooth transform interpolation using quaternion slerp for rotation
 * and linear interpolation for translation/scale.
 */
export function slerpTransform3D(a: Transform3D, b: Transform3D, t: number): Transform3D {
  const qa = quatFromEulerXYZ(a.rotation.x, a.rotation.y, a.rotation.z)
  const qb = quatFromEulerXYZ(b.rotation.x, b.rotation.y, b.rotation.z)
  const qInterp = quatSlerp(qa, qb, t)
  const rotation = quatToEulerXYZ(qInterp)

  return {
    translation: {
      x: lerp(a.translation.x, b.translation.x, t),
      y: lerp(a.translation.y, b.translation.y, t),
      z: lerp(a.translation.z, b.translation.z, t),
    },
    rotation,
    scale: {
      x: lerp(a.scale.x, b.scale.x, t),
      y: lerp(a.scale.y, b.scale.y, t),
      z: lerp(a.scale.z, b.scale.z, t),
    },
  }
}
