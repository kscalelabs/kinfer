//! Rerun logger that records every model step.
//
//!  • Each observation scalar is stored under
//!    `inference/obs/<tensor-name>/<index>`
//!  • Each action scalar is stored under
//!    `inference/action/<index>`
//
//! Every step also advances the "frame" time-sequence so you can scrub
//! through the run inside the Rerun viewer.

use rerun::{
    archetypes::{Scalars, SeriesLines},
    RecordingStream,
};
use std::{
    collections::{HashMap, HashSet},
    ffi::OsStr,
    path::{Path, PathBuf},
    process::Command,    // spawn the CLI
};

use ndarray::{Array, IxDyn};

/// Wrapper around `rerun::RecordingStream`
pub struct RerunLogger {
    rec: RecordingStream,
    path: PathBuf,          // ← keep the filename so Drop can compact
    frame: u64,
    series_declared: HashSet<String>,
}

impl RerunLogger {
    /// Create a new logger that writes an `.rrd` file at `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let rec = rerun::RecordingStreamBuilder::new("kinfer_inference_logger")
            .save(&path)          // must pass &Path
            .unwrap();
        Self {
            rec,
            path,                 // store it
            frame: 0,
            series_declared: HashSet::new(),
        }
    }

    /// Log one inference step.
    ///
    /// * `inputs`  – map from tensor-name -> ndarray observation  
    /// * `actions` – joint-action vector produced by the model
    pub fn log_step(
        &mut self,
        inputs: &HashMap<String, Array<f32, IxDyn>>,
        actions: &Array<f32, IxDyn>,
    ) {
        self.frame += 1;
        self.rec.set_time_sequence("frame", self.frame as i64);

        // ── observations ──────────────────────────────────────────────
        // Skip the `carry` tensor – it contains the full model state and
        // clutters the viewer.
        for (name, arr) in inputs {
            if name == "carry" {
                continue;
            }
            for (i, v) in arr.iter().enumerate() {
                let path = format!("inference/obs/{name}/{i}");
                // ignore failures: logging must never crash the sim
                let _ = self.rec.log(path.as_str(), &Scalars::new([*v as f64]));
            }
        }

        // ── action vector ────────────────────────────────────────────
        for (i, v) in actions.iter().enumerate() {
            let path = format!("inference/action/{i}");
            let _ = self.rec.log(path.as_str(), &Scalars::new([*v as f64]));
        }
    }

}

/* ──────────────────────────────────────────────────────────────── */
/* NEW: automatically compact when the logger is dropped           */
impl Drop for RerunLogger {
    fn drop(&mut self) {
        // 1. best-effort flush
        let _ = self.rec.flush_blocking();

        // 2. build a temp output name: <file>.compact.rrd
        let tmp: PathBuf = self
            .path
            .with_extension("compact.rrd");

        // 3. invoke the CLI (ignore failure but print it for debug)
        let status = Command::new("rerun")
            .args([
                "rrd", "compact",
                "--max-rows", "8192",
                "--max-bytes", "8388608",
                self.path.to_str().unwrap(),
                "-o", tmp.to_str().unwrap(),
            ])
            .status();

        if matches!(status, Ok(s) if s.success()) {
            // 4. atomically replace original with compacted version
            let _ = std::fs::rename(tmp, &self.path);
        } else {
            eprintln!(
                "[kinfer] Warning: rrd compact failed for {:?} (status={:?})",
                &self.path, status
            );
            // cleanup temp if it exists
            let _ = std::fs::remove_file(tmp);
        }
    }
}
