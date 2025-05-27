use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;

use crate::model::{ModelError, ModelRunner};
use crate::types::InputType;
use std::time::Duration;
use tokio::time::interval;

pub struct ModelRuntime {
    model_runner: Arc<ModelRunner>,
    dt: Duration,
    slowdown_factor: i32,
    magnitude_factor: f32,
    running: Arc<AtomicBool>,
    runtime: Option<Runtime>,
}

impl ModelRuntime {
    pub fn new(model_runner: Arc<ModelRunner>, dt: u64) -> Self {
        Self {
            model_runner,
            dt: Duration::from_millis(dt),
            slowdown_factor: 1,
            magnitude_factor: 1.0,
            running: Arc::new(AtomicBool::new(false)),
            runtime: None,
        }
    }

    pub fn set_slowdown_factor(&mut self, slowdown_factor: i32) {
        assert!(slowdown_factor >= 1);
        self.slowdown_factor = slowdown_factor;
    }

    pub fn set_magnitude_factor(&mut self, magnitude_factor: f32) {
        assert!(magnitude_factor >= 0.0);
        assert!(magnitude_factor <= 1.0);
        self.magnitude_factor = magnitude_factor;
    }

    pub fn start(&mut self) -> Result<(), ModelError> {
        if self.running.load(Ordering::Relaxed) {
            return Ok(());
        }

        let running = self.running.clone();
        let model_runner = self.model_runner.clone();
        let dt = self.dt;
        let slowdown_factor = self.slowdown_factor;
        let magnitude_factor = self.magnitude_factor;

        let runtime = Runtime::new()?;
        running.store(true, Ordering::Relaxed);

        runtime.spawn(async move {
            let mut carry = model_runner
                .init()
                .await
                .map_err(|e| ModelError::Provider(e.to_string()))?;

            let model_inputs = model_runner
                .get_inputs(&[InputType::JointAngles])
                .await
                .map_err(|e| ModelError::Provider(e.to_string()))?;
            let mut joint_positions = model_inputs[&InputType::JointAngles].clone();

            // Wait for the first tick, since it happens immediately.
            let mut interval = interval(dt);
            interval.tick().await;

            while running.load(Ordering::Relaxed) {
                let (output, next_carry) = model_runner
                    .step(carry)
                    .await
                    .map_err(|e| ModelError::Provider(e.to_string()))?;
                carry = next_carry;

                for i in 1..(slowdown_factor + 1) {
                    if !running.load(Ordering::Relaxed) {
                        break;
                    }
                    let t = i as f32 / slowdown_factor as f32;
                    let interp_joint_positions = &joint_positions * (1.0 - t) + &output * t;
                    model_runner
                        .take_action(interp_joint_positions * magnitude_factor)
                        .await
                        .map_err(|e| ModelError::Provider(e.to_string()))?;
                    interval.tick().await;
                }

                joint_positions = output;
            }
            Ok::<(), ModelError>(())
        });

        self.runtime = Some(runtime);
        Ok(())
    }

    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}
