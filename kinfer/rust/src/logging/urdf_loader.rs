//! High-level helper: *load a URDF, stream it to Rerun, and print
//! an easy-to-read joint table*.
//
//  This deliberately avoids animation – it just gives you the
//  same "sanity-check" view you used in `main.rs` so you can
//  integrate step-by-step.

use anyhow::Result;
use rerun::{
    archetypes::{Mesh3D, ViewCoordinates},
    components::TriangleIndices,
    RecordingStream,
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};
use stl_io;
use urdf_rs::{self as urdf, Geometry};

use crate::logging::math_utils::*;

const ID4: [f32; 16] = [
    1., 0., 0., 0., //
    0., 1., 0., 0., //
    0., 0., 1., 0., //
    0., 0., 0., 1.,
];

/// The one public entry-point you'll call from Kinfer.
///
/// ```rust
/// use kinfer::logging::log_urdf;
///
/// let rec = rerun::RecordingStreamBuilder::new("my_run").spawn()?;
/// log_urdf(&rec, "/path/to/robot.urdf", true)?;
/// ```
pub fn log_urdf(rec: &RecordingStream, urdf_path: impl AsRef<Path>, verbose: bool) -> Result<()> {
    let urdf_path = urdf_path.as_ref();
    let robot = urdf::read_file(urdf_path)?;
    let urdf_dir = urdf_path
        .parent()
        .expect("URDF has no parent directory")
        .to_path_buf();

    // 1) Set Rerun coordinate convention once.
    rec.log("", &ViewCoordinates::RIGHT_HAND_Z_UP())?;

    // 2) Build look-ups
    let mut link_map = HashMap::<String, &urdf::Link>::new();
    for l in &robot.links {
        link_map.insert(l.name.clone(), l);
    }
    let mut children = HashMap::<String, Vec<&urdf::Joint>>::new();
    for j in &robot.joints {
        children
            .entry(j.parent.link.clone())
            .or_default()
            .push(j);
    }

    // 3) Root = any link that is **never** a child
    let mut all_links: HashSet<_> = link_map.keys().cloned().collect();
    for j in &robot.joints {
        all_links.remove(&j.child.link);
    }
    let root = all_links
        .into_iter()
        .next()
        .expect("Could not find root link");

    if verbose {
        println!("\n========== URDF JOINT TABLE ==========");
        for j in &robot.joints {
            println!(
                "• {:<22} {:<10} {:<10}→{:<25}  xyz={:?}  rpy(rad)={:?}",
                j.name,
                format!("{:?}", j.joint_type),
                j.parent.link,
                j.child.link,
                j.origin.xyz,
                j.origin.rpy
            );
        }
        println!("======================================\n");
    }

    // 4) BFS traversal
    let mut q = VecDeque::new();
    q.push_back((root.clone(), ID4));

    while let Some((link_name, link_tf)) = q.pop_front() {
        // Pretty print info
        if verbose {
            println!("▶ VISITING link '{}'", link_name);
        }

        let link = link_map[&link_name];

        // (a) Stream every <visual>
        for (i, vis) in link.visual.iter().enumerate() {
            let local_tf = make_tf(
                [vis.origin.xyz[0], vis.origin.xyz[1], vis.origin.xyz[2]],
                [vis.origin.rpy[0], vis.origin.rpy[1], vis.origin.rpy[2]],
            );
            let world_tf = mat4_mul(link_tf, local_tf);

            if let Geometry::Mesh { filename, .. } = &vis.geometry {
                // We only support STL for now.
                let mesh_path = canonical(&urdf_dir, filename);
                if mesh_path.extension().and_then(|e| e.to_str()) == Some("stl") {
                    let stl = {
                        let mut reader = BufReader::new(File::open(&mesh_path)?);
                        stl_io::read_stl(&mut reader)?
                    };

                    let mut verts: Vec<[f32; 3]> =
                        stl.vertices.iter().map(|v| [v[0], v[1], v[2]]).collect();
                    apply_tf(&mut verts, world_tf);

                    let tris: Vec<[u32; 3]> = stl
                        .faces
                        .iter()
                        .map(|f| [f.vertices[0] as u32, f.vertices[1] as u32, f.vertices[2] as u32])
                        .collect();

                    let mesh = Mesh3D::new(to_positions(&verts))
                        .with_triangle_indices(
                            tris.iter()
                                .map(|t| TriangleIndices::from(*t))
                                .collect::<Vec<_>>(),
                        )
                        .with_vertex_normals(vertex_normals(&verts, &tris));

                    let ent_path = format!("{link_name}/visual_{i}");
                    rec.log(ent_path.as_str(), &mesh)?;
                }
            }
        }

        // (b) Queue children
        if let Some(kids) = children.get(&link_name) {
            for j in kids {
                let child_tf = make_tf(
                    [j.origin.xyz[0], j.origin.xyz[1], j.origin.xyz[2]],
                    [j.origin.rpy[0], j.origin.rpy[1], j.origin.rpy[2]],
                );
                let next_tf = mat4_mul(link_tf, child_tf);

                if verbose {
                    println!(
                        "  └─ joint '{:<22}' {} → {}   xyz={:?} rpy={:?}",
                        j.name, j.parent.link, j.child.link, j.origin.xyz, j.origin.rpy
                    );
                }

                q.push_back((j.child.link.clone(), next_tf));
            }
        }

        if verbose {
            println!();
        }
    }

    Ok(())
}

// Helper to keep paths readable
fn canonical(base: &Path, rel: &str) -> PathBuf {
    let joined = base.join(rel);
    joined
        .canonicalize()
        .unwrap_or_else(|_| joined) // fall back to non-canonical if it fails
}
