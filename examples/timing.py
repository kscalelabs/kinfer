"""Demonstrates timing for the K-Infer runtime loop."""

import argparse
import logging
import tarfile
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch
from torch import Tensor

from kinfer.export.pytorch import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import (
    ModelProviderABC,
    PyModelMetadata,
    PyModelRunner,
    metadata_from_json,
)

logger = logging.getLogger(__name__)

JOINT_NAMES = ["left_arm", "right_arm", "left_leg", "right_leg"]
NUM_JOINTS = len(JOINT_NAMES)
CARRY_SIZE = 10
COMMAND_NAMES = ["xvel", "yvel", "yawrate", "baseheight"]


class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.isoformat()


@torch.jit.script
def init_fn() -> Tensor:
    return torch.zeros((10,))  # NOTE: Can't use the CARRY_SIZE constant here.


@torch.jit.script
def step_fn(
    joint_angles: Tensor,
    joint_angular_velocities: Tensor,
    projected_gravity: Tensor,
    accelerometer: Tensor,
    gyroscope: Tensor,
    command: Tensor,
    time: Tensor,
    carry: Tensor,
) -> tuple[Tensor, Tensor]:
    output = (
        joint_angles.mean()
        + joint_angular_velocities.mean()
        + projected_gravity.mean()
        + accelerometer.mean()
        + gyroscope.mean()
        + command.mean()
        + torch.cos(time).mean()
        + torch.sin(time).mean()
        + carry.mean()
    ) * joint_angles
    next_carry = carry + 1
    return output, next_carry


class DummyModelProvider(ModelProviderABC):
    def __init__(self) -> None:
        self.event_times: defaultdict[str, list[float]] = defaultdict(list)

    def pre_fetch_inputs(self, input_types: Sequence[str], metadata: PyModelMetadata) -> None:
        self.record_step_event("pre_fetch_inputs")

    def get_inputs(self, input_types: Sequence[str], metadata: PyModelMetadata) -> dict[str, np.ndarray]:
        self.record_step_event("get_inputs")
        return_values: dict[str, np.ndarray] = {}
        for input_type in input_types:
            match input_type:
                case "joint_angles":
                    return_values["joint_angles"] = np.random.randn(NUM_JOINTS)
                case "joint_angular_velocities":
                    return_values["joint_angular_velocities"] = np.random.randn(NUM_JOINTS)
                case "projected_gravity":
                    return_values["projected_gravity"] = np.random.randn(3)
                case "accelerometer":
                    return_values["accelerometer"] = np.random.randn(3)
                case "gyroscope":
                    return_values["gyroscope"] = np.random.randn(3)
                case "command":
                    return_values["command"] = np.random.randn(len(COMMAND_NAMES))
                case "time":
                    return_values["time"] = np.random.randn(1)
                case _:
                    raise ValueError(f"Unknown input type: {input_type}")
        return return_values

    def take_action(self, action: np.ndarray, metadata: PyModelMetadata) -> None:
        self.record_step_event("take_action")
        assert metadata.joint_names == JOINT_NAMES  # type: ignore[attr-defined]
        assert action.shape == (NUM_JOINTS,)

    def record_step_event(self, event_name: str) -> None:
        """Record a step-related event."""
        current_time = time.monotonic()
        logger.info("Running %s at time %s", event_name, current_time)
        self.event_times[event_name].append(current_time)


def create_timing_plot(provider: DummyModelProvider, dt: timedelta, runtime: timedelta | None) -> plt.Figure:
    """Create a matplotlib plot showing event timing relative to expected tick times."""
    dt_ms = dt.total_seconds() * 1000

    # Calculate expected tick times
    start_time: float = min(min(times) for times in provider.event_times.values()) if provider.event_times else 0

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for different events
    event_colors = {
        "pre_fetch_inputs": "blue",
        "get_inputs": "green",
        "step_start": "red",
        "step_end": "darkred",
        "take_action_start": "orange",
        "take_action_end": "darkorange",
        "init_start": "purple",
        "init_end": "darkviolet",
    }

    # Plot each event type
    for event_name, times in provider.event_times.items():
        if not times:
            continue

        # Calculate relative times (ms from start)
        relative_times = [(t - start_time) * 1000 for t in times]

        # For each event, find which tick it belongs to
        tick_numbers = []
        relative_to_tick = []
        for event_time in relative_times:
            tick_number = int(event_time / dt_ms)
            relative_time = event_time - tick_number * dt_ms
            tick_numbers.append(tick_number)
            relative_to_tick.append(relative_time)

        # Plot the events
        color = event_colors.get(event_name, "black")
        ax.scatter(tick_numbers, relative_to_tick, label=event_name, color=color, alpha=0.7, s=30)

    # Add horizontal line at y=0 (expected tick time)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="Expected tick time")

    # Customize the plot
    ax.set_xlabel("Tick Number")
    ax.set_ylabel("Time Relative to Expected Tick Start (ms)")
    ax.set_title(f"K-Infer Event Timing (Runtime={runtime.total_seconds() if runtime else 'None'}s)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Set reasonable y-axis limits
    if provider.event_times:
        all_relative_times = []
        for times in provider.event_times.values():
            if times:
                relative_times = [(t - start_time) * 1000 for t in times]
                all_relative_times.extend(relative_times)

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=float, help="Runtime in seconds")
    parser.add_argument("--plot", action="store_true", help="Generate and display timing plot")
    parser.add_argument("--save-plot", type=str, help="Save plot to file (e.g., timing_plot.png)")
    parser.add_argument("--dt", type=int, default=20, help="Time step in milliseconds (default: 10)")
    parser.add_argument("--pre-fetch-time", type=int, help="Pre-fetch time in milliseconds (default: 2)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dt = timedelta(milliseconds=args.dt)
    runtime = None if args.runtime is None else timedelta(seconds=args.runtime)
    pre_fetch_time: int | None = args.pre_fetch_time

    for handler in logging.getLogger().handlers:
        handler.setFormatter(MicrosecondFormatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S.%f"))

    metadata = PyModelMetadata(
        joint_names=JOINT_NAMES,
        command_names=COMMAND_NAMES,
        carry_size=[CARRY_SIZE],
    )

    init_fn_onnx = export_fn(init_fn, metadata)
    step_fn_onnx = export_fn(step_fn, metadata)
    kinfer_model = pack(init_fn_onnx, step_fn_onnx, metadata)

    # Saves the model to disk.
    with TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        (kinfer_path := root_dir / "model.kinfer").write_bytes(kinfer_model)

        # Ensures that we can open the file like a regular tar file.
        with tarfile.open(kinfer_path, "r:gz") as tar:
            assert tar.getnames() == ["init_fn.onnx", "step_fn.onnx", "metadata.json"]

            # Checks that joint_names.json is valid JSON.
            if (fpath := tar.extractfile("metadata.json")) is None:
                raise ValueError("metadata.json not found")
            metadata = metadata_from_json(fpath.read().decode("utf-8"))
            assert metadata.joint_names == JOINT_NAMES  # type: ignore[attr-defined]

            # Validates that we can construct a session in Python.
            if (fpath := tar.extractfile("init_fn.onnx")) is None:
                raise ValueError("init_fn.onnx not found")
            init_session = onnxruntime.InferenceSession(fpath.read())
            assert init_session.get_modelmeta().graph_name == "main_graph"
            if (fpath := tar.extractfile("step_fn.onnx")) is None:
                raise ValueError("step_fn.onnx not found")
            step_session = onnxruntime.InferenceSession(fpath.read())
            assert step_session.get_modelmeta().graph_name == "main_graph"

        # Creates a model runner from the kinfer model.
        model_provider = DummyModelProvider()
        model_runner = PyModelRunner(str(kinfer_path), model_provider, pre_fetch_time)

    model_runner.run(dt, total_runtime=runtime)

    # Generate timing plot if requested
    if args.plot or args.save_plot:
        logger.info("Generating timing plot...")
        fig = create_timing_plot(model_provider, dt, runtime)

        if args.save_plot:
            fig.savefig(args.save_plot, dpi=300, bbox_inches="tight")
            logger.info("Plot saved to %s", args.save_plot)

        if args.plot:
            plt.show()

        # Print timing statistics
        print("\n=== Timing Statistics ===")
        for event_name, times in model_provider.event_times.items():
            if times:
                print(f"{event_name}: {len(times)} events")
                if len(times) > 1:
                    intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
                    avg_interval = sum(intervals) / len(intervals)
                    print(f"  Average interval: {avg_interval * 1000:.2f} ms")

    else:
        logger.info("Use --plot to display timing visualization or --save-plot <filename> to save it")


if __name__ == "__main__":
    main()
