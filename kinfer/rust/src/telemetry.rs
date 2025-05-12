use eyre::Result;
use lazy_static::lazy_static;
use rumqttc::{AsyncClient, MqttOptions, QoS};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct Telemetry {
    client: Arc<AsyncClient>,
    pub robot_id: String,
}

lazy_static! {
    static ref TELEMETRY: Arc<Mutex<Option<Telemetry>>> = Arc::new(Mutex::new(None));
    static ref TELEMETRY_ENABLED: bool = std::env::var("ENABLE_TELEMETRY")
        .map(|v| v.to_lowercase() != "false")
        .unwrap_or(true);
}

#[derive(Serialize)]
struct TelemetryPayload<T> {
    timestamp: u64,
    data: T,
}

impl Telemetry {
    pub async fn initialize(robot_id: &str, mqtt_host: &str, mqtt_port: u16) -> Result<()> {
        let mut mqtt_options = MqttOptions::new(format!("kos-{}", robot_id), mqtt_host, mqtt_port);
        mqtt_options.set_keep_alive(std::time::Duration::from_secs(5));

        let (client, mut eventloop) = AsyncClient::new(mqtt_options, 10);

        // Spawn a task to handle MQTT connection events
        tokio::spawn(async move {
            while let Ok(notification) = eventloop.poll().await {
                tracing::trace!("MQTT Event: {:?}", notification);
            }
        });

        let telemetry = Telemetry {
            client: Arc::new(client),
            robot_id: robot_id.to_string(),
        };

        tracing::debug!("Initializing telemetry for robot {}", robot_id);
        let mut global = TELEMETRY.lock().await;
        *global = Some(telemetry);

        Ok(())
    }

    pub async fn get() -> Option<Telemetry> {
        if !*TELEMETRY_ENABLED {
            return None;
        }
        TELEMETRY.lock().await.clone()
    }

    pub async fn publish<T: Serialize>(&self, topic: &str, payload: &T) -> Result<()> {
        let telemetry_payload = TelemetryPayload {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| eyre::eyre!("Failed to get system time: {}", e))?
                .as_millis() as u64,
            data: payload,
        };

        let payload = serde_json::to_string(&telemetry_payload)?;
        let full_topic = format!("robots/{}/{}", self.robot_id, topic);

        self.client
            .publish(full_topic, QoS::AtLeastOnce, false, payload)
            .await?;

        Ok(())
    }

    pub fn try_get() -> Option<Self> {
        // Try to get the global telemetry instance
        if let Ok(guard) = TELEMETRY.try_lock() {
            guard.as_ref().cloned()
        } else {
            None
        }
    }
}
