"""
visualize_metrics.py — Generate 4-panel chart from metrics.json.
Run after agent stops: python visualize_metrics.py [--input metrics.json] [--output chart.png]
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: str) -> list:
    """Load metrics from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def plot_metrics(metrics: list, output_path: str = "chart.png"):
    """Generate 4-panel chart."""
    steps = [m["step"] for m in metrics]
    
    # Extract data with defaults for missing keys
    latency = [m.get("latency", 0) for m in metrics]
    cold_starts = [m.get("cold_starts", 0) for m in metrics]
    warm_hits = [m.get("warm_hits", 0) for m in metrics]
    queue_depth = [m.get("queue", 0) for m in metrics]
    containers = [m.get("warm_containers", 1) for m in metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("FaaS RL Agent Metrics", fontsize=14, fontweight="bold")
    
    # Top-left: Latency
    axes[0, 0].plot(steps, latency, color="steelblue", linewidth=1.5)
    axes[0, 0].set_title("Latency (seconds)", fontweight="bold")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Seconds")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top-right: Cold vs Warm
    axes[0, 1].plot(steps, cold_starts, label="Cold Starts", color="coral", linewidth=1.5)
    axes[0, 1].plot(steps, warm_hits, label="Warm", color="seagreen", linewidth=1.5)
    axes[0, 1].set_title("Cold vs Warm (requests)", fontweight="bold")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Number of Requests")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom-left: Queue Size
    axes[1, 0].plot(steps, queue_depth, color="steelblue", linewidth=1.5)
    axes[1, 0].set_title("Queue Size (requests)", fontweight="bold")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Number of Requests")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom-right: Containers
    axes[1, 1].plot(steps, containers, color="steelblue", linewidth=1.5)
    axes[1, 1].set_title("Containers (count)", fontweight="bold")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Number of Containers")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize FaaS RL metrics")
    parser.add_argument("--input", default="checkpoints/metrics.json", help="Input metrics JSON")
    parser.add_argument("--output", default="chart.png", help="Output PNG path")
    args = parser.parse_args()
    
    metrics = load_metrics(args.input)
    print(f"Loaded {len(metrics)} steps from {args.input}")
    plot_metrics(metrics, args.output)
