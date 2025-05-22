//! Public surface for the `logging` utilities used by Kinfer.
//
//  ── layout ──
//  logging/
//     ├─ mod.rs          (this file)
//     ├─ logger.rs       (your existing logger, unchanged)
//     ├─ urdf_loader.rs  (new; logs URDFs to Rerun)
//     └─ math.rs         (new; small linear-algebra helpers)

pub mod logger;
pub mod math_utils;
pub mod urdf_loader;

// Re-export the most useful items so callers can just do
//   use crate::logging::log_urdf;
pub use logger::*;
pub use math_utils::*;
pub use urdf_loader::*;
