//! Tiny math helpers that the URDF loader (and future code) rely on.
//
//  All matrices are **row-major** `[f32; 16]` (4×4) or `[f32; 9]` (3×3).

use rerun::components::{Position3D, Vector3D};

/// R = Rz * Ry * Rx  (extrinsic XYZ)
pub fn rot_xyz(rx: f64, ry: f64, rz: f64) -> [f32; 9] {
    let (cx, sx) = (rx.cos() as f32, rx.sin() as f32);
    let (cy, sy) = (ry.cos() as f32, ry.sin() as f32);
    let (cz, sz) = (rz.cos() as f32, rz.sin() as f32);

    [
        cz * cy,
        cz * sy * sx - sz * cx,
        cz * sy * cx + sz * sx,
        sz * cy,
        sz * sy * sx + cz * cx,
        sz * sy * cx - cz * sx,
        -sy,
        cy * sx,
        cy * cx,
    ]
}

/// Make a 4 × 4 row-major transform from xyz+RPY.
pub fn make_tf(xyz: [f64; 3], rpy: [f64; 3]) -> [f32; 16] {
    let r = rot_xyz(rpy[0], rpy[1], rpy[2]);
    [
        r[0], r[1], r[2], xyz[0] as f32, //
        r[3], r[4], r[5], xyz[1] as f32, //
        r[6], r[7], r[8], xyz[2] as f32, //
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// C = A × B  (row-major)
pub fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut o = [0.0; 16];
    for r in 0..4 {
        for c in 0..4 {
            o[r * 4 + c] = (0..4).map(|k| a[r * 4 + k] * b[k * 4 + c]).sum();
        }
    }
    o
}

/// Apply transform to every vertex *(positions only)*.
pub fn apply_tf(verts: &mut [[f32; 3]], tf: [f32; 16]) {
    for v in verts {
        let (x, y, z) = (v[0], v[1], v[2]);
        v[0] = tf[0] * x + tf[1] * y + tf[2] * z + tf[3];
        v[1] = tf[4] * x + tf[5] * y + tf[6] * z + tf[7];
        v[2] = tf[8] * x + tf[9] * y + tf[10] * z + tf[11];
    }
}

/// Very small area-weighted vertex-normal generator.
pub fn vertex_normals(verts: &[[f32; 3]], tris: &[[u32; 3]]) -> Vec<Vector3D> {
    let mut acc = vec![[0.0f32; 3]; verts.len()];
    for t in tris {
        let (i0, i1, i2) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let p0 = verts[i0];
        let p1 = verts[i1];
        let p2 = verts[i2];
        let v10 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v20 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n = [
            v10[1] * v20[2] - v10[2] * v20[1],
            v10[2] * v20[0] - v10[0] * v20[2],
            v10[0] * v20[1] - v10[1] * v20[0],
        ];
        for i in [i0, i1, i2] {
            acc[i][0] += n[0];
            acc[i][1] += n[1];
            acc[i][2] += n[2];
        }
    }
    acc.into_iter()
        .map(|n| {
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
            Vector3D::from([n[0] / l, n[1] / l, n[2] / l])
        })
        .collect()
}

/// Convenience for converting Vec<[f32;3]> to Vec<Position3D>.
pub fn to_positions(v: &[[f32; 3]]) -> Vec<Position3D> {
    v.iter().map(|p| Position3D::from(*p)).collect()
}
