//! Rerun logger that records every model step.
//
//!  • Each observation scalar is stored under
//!    `inference/obs/<tensor-name>/<index>`
//!  • Each action scalar is stored under
//!    `inference/action/<index>`
//
//! Every step also advances the "frame" time-sequence so you can scrub
//! through the run inside the Rerun viewer.

use rerun::{archetypes::Scalars, RecordingStream};
use std::collections::HashMap;
use std::path::PathBuf;

use ndarray::{Array, IxDyn};

/// Wrapper around `rerun::RecordingStream`
pub struct RerunLogger {
    rec: RecordingStream,
    frame: u64,
}

impl RerunLogger {
    /// Create a new logger that writes an `.rrd` file at `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let rec = rerun::RecordingStreamBuilder::new("kinfer_inference_logger")
            .save(path)
            .unwrap();
        Self { rec, frame: 0 }
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
