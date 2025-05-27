"""Plot NDJSON logs saved by kinfer."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def read_ndjson(filepath: str) -> list[dict]:
    """Read NDJSON file and return list of parsed objects."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def skip_initial_data(data: list[dict], skip_seconds: float) -> list[dict]:
    """Skip the first n seconds of data based on timestamps."""
    if not data or skip_seconds <= 0.0:
        return data

    # Extract timestamps and convert to seconds relative to first timestamp
    timestamps = [d["t_us"] for d in data]
    t_start = timestamps[0]
    times = [(t - t_start) / 1e6 for t in timestamps]  # Convert to seconds

    # Find indices where time >= skip_seconds
    skip_indices = [i for i, t in enumerate(times) if t >= skip_seconds]
    if not skip_indices:
        logger.info("All data points are within the skip period (%.2f seconds). No data to plot.", skip_seconds)
        return []

    # Filter data
    start_idx = skip_indices[0]
    filtered_data = data[start_idx:]
    logger.info("Skipped first %.2f seconds (%d data points)", skip_seconds, start_idx)

    return filtered_data


def plot_data(data: list[dict], save_path: Optional[Union[str, Path]] = None) -> None:
    """Plot all data fields from the NDJSON."""
    if not data:
        logger.info("No data to plot")
        return

    # Extract timestamps and convert to seconds relative to first timestamp
    timestamps = [d["t_us"] for d in data]
    t_start = timestamps[0]
    times = [(t - t_start) / 1e6 for t in timestamps]  # Convert to seconds

    # Extract data arrays
    joint_angles = np.array([d["joint_angles"] for d in data if d["joint_angles"] is not None])
    joint_vels = np.array([d["joint_vels"] for d in data if d["joint_vels"] is not None])
    projected_g = np.array([d["projected_g"] for d in data if d["projected_g"] is not None])
    accel = np.array([d["accel"] for d in data if d["accel"] is not None])
    command = np.array([d["command"] for d in data if d["command"] is not None])
    output = np.array([d["output"] for d in data if d["output"] is not None])

    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Robot Data Over Time", fontsize=16)

    if len(joint_angles) > 0:
        ax = axes[0, 0]
        for i in range(joint_angles.shape[1]):
            ax.plot(times[: len(joint_angles)], joint_angles[:, i], alpha=0.7, linewidth=0.8)
        ax.set_title("Joint Angles")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (rad)")
        ax.grid(True, alpha=0.3)

    if len(joint_vels) > 0:
        ax = axes[0, 1]
        for i in range(joint_vels.shape[1]):
            ax.plot(times[: len(joint_vels)], joint_vels[:, i], alpha=0.7, linewidth=0.8)
        ax.set_title("Joint Velocities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (rad/s)")
        ax.grid(True, alpha=0.3)

    if len(projected_g) > 0:
        ax = axes[1, 0]
        labels = ["X", "Y", "Z"]
        for i in range(projected_g.shape[1]):
            ax.plot(times[: len(projected_g)], projected_g[:, i], label=labels[i], linewidth=1.5)
        ax.set_title("Projected Gravity")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if len(accel) > 0:
        ax = axes[1, 1]
        labels = ["X", "Y", "Z"]
        for i in range(accel.shape[1]):
            ax.plot(times[: len(accel)], accel[:, i], label=labels[i], linewidth=1.5)
        ax.set_title("Acceleration")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if len(command) > 0:
        ax = axes[2, 0]
        for i in range(command.shape[1]):
            ax.plot(times[: len(command)], command[:, i], label=f"Cmd {i}", linewidth=1.2)
        ax.set_title("Command")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Command Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if len(output) > 0:
        ax = axes[2, 1]
        for i in range(output.shape[1]):
            ax.plot(times[: len(output)], output[:, i], alpha=0.7, linewidth=0.8)
        ax.set_title("Output")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Output Value")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Plot saved to: %s", save_path)
        plt.close()
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NDJSON logs saved by kinfer")
    parser.add_argument("filepath", help="Path to the NDJSON file to plot")
    parser.add_argument("--skip", type=float, default=0.0, help="Skip the first n seconds of data")
    parser.add_argument("--save", action="store_true", help="Save the plot to a PNG file in a plots folder")
    args = parser.parse_args()

    filepath = args.filepath
    if not Path(filepath).exists():
        logger.info("File not found: %s", filepath)
        return

    logger.info("Reading data from %s...", filepath)
    data = read_ndjson(filepath)
    logger.info("Loaded %d data points", len(data))

    filtered_data = skip_initial_data(data, args.skip)

    save_path = None
    if args.save:
        # Create save path in plots folder with same name but .png extension
        input_path = Path(filepath)
        plots_dir = input_path.parent / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Change extension from .ndjson to .png
        filename = input_path.stem + ".png"
        save_path = str(plots_dir / filename)

    plot_data(filtered_data, save_path)


if __name__ == "__main__":
    main()
