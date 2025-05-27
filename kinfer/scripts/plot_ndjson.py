#!/usr/bin/env python3
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def read_ndjson(filepath):
    """Read NDJSON file and return list of parsed objects"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def filter_data_by_time(data, skip_seconds=0.1):
    """Filter out the first skip_seconds of data"""
    if not data or skip_seconds <= 0:
        return data
    
    # Get the first timestamp
    t_start = data[0]['t_us']
    skip_us = skip_seconds * 1e6  # Convert to microseconds
    
    # Filter data to skip the first skip_seconds
    filtered_data = [d for d in data if (d['t_us'] - t_start) >= skip_us]
    
    print(f"Skipped first {skip_seconds}s of data ({len(data) - len(filtered_data)} points)")
    return filtered_data

def plot_data(data, output_path=None, skip_seconds=0.1):
    """Plot all data fields from the NDJSON"""
    if not data:
        print("No data to plot")
        return
    
    # Filter out the first skip_seconds of data
    data = filter_data_by_time(data, skip_seconds)
    
    if not data:
        print("No data remaining after filtering")
        return
    
    # Extract timestamps and convert to seconds relative to first timestamp
    timestamps = [d['t_us'] for d in data]
    t_start = timestamps[0]
    times = [(t - t_start) / 1e6 for t in timestamps]  # Convert to seconds
    
    # Extract data arrays
    joint_angles = np.array([d['joint_angles'] for d in data if d['joint_angles'] is not None])
    joint_vels = np.array([d['joint_vels'] for d in data if d['joint_vels'] is not None])
    projected_g = np.array([d['projected_g'] for d in data if d['projected_g'] is not None])
    accel = np.array([d['accel'] for d in data if d['accel'] is not None])
    command = np.array([d['command'] for d in data if d['command'] is not None])
    output = np.array([d['output'] for d in data if d['output'] is not None])
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Robot Data Over Time (skipped first {skip_seconds}s)', fontsize=16)
    
    # Plot joint angles
    if len(joint_angles) > 0:
        ax = axes[0, 0]
        for i in range(joint_angles.shape[1]):
            ax.plot(times[:len(joint_angles)], joint_angles[:, i], alpha=0.7, linewidth=0.8)
        ax.set_title('Joint Angles')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rad)')
        ax.grid(True, alpha=0.3)
    
    # Plot joint velocities
    if len(joint_vels) > 0:
        ax = axes[0, 1]
        for i in range(joint_vels.shape[1]):
            ax.plot(times[:len(joint_vels)], joint_vels[:, i], alpha=0.7, linewidth=0.8)
        ax.set_title('Joint Velocities')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (rad/s)')
        ax.grid(True, alpha=0.3)
    
    # Plot projected gravity
    if len(projected_g) > 0:
        ax = axes[1, 0]
        labels = ['X', 'Y', 'Z']
        for i in range(projected_g.shape[1]):
            ax.plot(times[:len(projected_g)], projected_g[:, i], label=labels[i], linewidth=1.5)
        ax.set_title('Projected Gravity')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot acceleration
    if len(accel) > 0:
        ax = axes[1, 1]
        labels = ['X', 'Y', 'Z']
        for i in range(accel.shape[1]):
            ax.plot(times[:len(accel)], accel[:, i], label=labels[i], linewidth=1.5)
        ax.set_title('Acceleration')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot command
    if len(command) > 0:
        ax = axes[2, 0]
        for i in range(command.shape[1]):
            ax.plot(times[:len(command)], command[:, i], label=f'Cmd {i}', linewidth=1.2)
        ax.set_title('Command')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Command Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot output
    if len(output) > 0:
        ax = axes[2, 1]
        for i in range(output.shape[1]):
            ax.plot(times[:len(output)], output[:, i], alpha=0.7, linewidth=0.8)
        ax.set_title('Output')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Output Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot robot data from NDJSON file')
    parser.add_argument('filepath', help='Path to the NDJSON file to plot')
    parser.add_argument('--save', action='store_true', help='Save plot to disk')
    parser.add_argument('--skip', type=float, default=0.1, 
                       help='Skip first N seconds of data (default: 0.1)')
    
    args = parser.parse_args()
    
    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return 1
    
    print(f"Reading data from {filepath}...")
    data = read_ndjson(filepath)
    print(f"Loaded {len(data)} data points")
    
    # Generate output path if saving is requested
    output_path = None
    if args.save:
        # Create output filename: original_name_plot.png
        output_path = filepath.parent / f"{filepath.stem}_plot.png"
    
    plot_data(data, output_path, skip_seconds=args.skip)
    return 0

if __name__ == "__main__":
    exit(main())