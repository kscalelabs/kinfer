mod model;
use model::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        // Add this line
        return Err("Expected one argument".into());
    }

    let model_path = &args[1];
    let model = load_onnx_model(model_path).unwrap();
    println!("Model loaded");

    let x_vel = ndarray::Array1::<f32>::zeros(1);
    let y_vel = ndarray::Array1::<f32>::zeros(1);
    let rot = ndarray::Array1::<f32>::zeros(1);
    let mut t = ndarray::Array1::<f32>::zeros(1);
    let dof_pos = ndarray::Array1::<f32>::zeros(10);
    let dof_vel = ndarray::Array1::<f32>::zeros(10);
    let mut prev_actions = ndarray::Array1::<f32>::zeros(10);
    let imu_ang_vel = ndarray::Array1::<f32>::zeros(3);
    let imu_euler_xyz = ndarray::Array1::<f32>::zeros(3);
    let mut buffer = ndarray::Array1::<f32>::zeros(574);

    for t_i in 0..50 {
        t[0] = t_i as f32 / 50.0;

        // Runs the model.
        let outputs = model.run(ort::inputs![
            "x_vel.1" => x_vel.clone(),
            "y_vel.1" => y_vel.clone(),
            "rot.1" => rot.clone(),
            "t.1" => t.clone(),
            "dof_pos.1" => dof_pos.clone(),
            "dof_vel.1" => dof_vel.clone(),
            "prev_actions.1" => prev_actions.clone(),
            "imu_ang_vel.1" => imu_ang_vel.clone(),
            "imu_euler_xyz.1" => imu_euler_xyz.clone(),
            "buffer.1" => buffer.clone(),
        ]?)?;

        // Updates the prev_actions and buffer.
        let actions = outputs[1].try_extract_tensor::<f32>().unwrap();
        let next_buffer = outputs[2].try_extract_tensor::<f32>().unwrap();
        prev_actions.assign(&actions);
        buffer.assign(&next_buffer);

        // Print the resulting torques.
        let torques = outputs[0].try_extract_tensor::<f32>().unwrap();
        println!("{:?}", torques);
    }

    Ok(())
}
