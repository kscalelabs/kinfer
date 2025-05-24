use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    thread,
};

use crossbeam_channel::{bounded, Sender};
use log::info;
use serde::Serialize;

#[derive(Serialize)]
struct NdjsonStep<'a> {
    step_id:      u64,
    t_us:         u128,
    joint_angles: &'a [f32],
    joint_vels:   &'a [f32],
    projected_g:  &'a [f32],
    accel:        &'a [f32],
    gyro:         &'a [f32],
    command:      Option<&'a [f32]>,
    output:       &'a [f32],
}

const CHANNEL_CAP: usize = 1024;
const FLUSH_EVERY: u64 = 100;

pub struct StepLogger {
    tx:        Sender<Vec<u8>>,
    worker:    Option<thread::JoinHandle<()>>,
    next_id:   std::sync::atomic::AtomicU64,
}

impl StepLogger {
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path: PathBuf = path.as_ref().into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        info!("kinfer: logging to NDJSON: {}", path.display());

        // I/O objects created here, but moved into the worker thread.
        let file  = OpenOptions::new().create(true).append(true).open(&path)?;
        let mut bw = BufWriter::new(file);

        // Bounded channel -> back-pressure capped at CHANNEL_CAP lines
        let (tx, rx) = bounded::<Vec<u8>>(CHANNEL_CAP);

        let worker = thread::spawn(move || {
            let mut line_ctr: u64 = 0;
            for msg in rx { // drains until all senders dropped
                let _ = bw.write_all(&msg);
                line_ctr += 1;
                if line_ctr % FLUSH_EVERY == 0 {
                    let _ = bw.flush();
                }
            }
            // Final flush on graceful shutdown
            let _ = bw.flush();
        });

        Ok(Self {
            tx,
            worker: Some(worker),
            next_id: std::sync::atomic::AtomicU64::new(0),
        })
    }

    #[inline]
    fn now_us() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros()
    }

    /// Non-blocking; drops a line if the channel is full.
    pub fn log_step(
        &self,
        joint_angles: &[f32],
        joint_vels:   &[f32],
        projected_g:  &[f32],
        accel:        &[f32],
        gyro:         &[f32],
        command:      Option<&[f32]>,
        output:       &[f32],
    ) {
        let record = NdjsonStep {
            step_id: self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            t_us:    Self::now_us(),
            joint_angles,
            joint_vels,
            projected_g,
            accel,
            gyro,
            command,
            output,
        };

        // Serialise directly into a Vec<u8>; then push newline and send.
        if let Ok(mut line) = serde_json::to_vec(&record) {
            line.push(b'\n');
            let _ = self.tx.try_send(line);   // drop if the queue is full
        }
    }
}

/// Ensure the worker drains and flushes before program exit.
impl Drop for StepLogger {
    fn drop(&mut self) {
        // drop Sender -> channel closes
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
