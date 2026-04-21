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


def filter_metrics(metrics: list, start_step: int | None, end_step: int | None) -> list:
    out = metrics
    if start_step is not None:
        out = [m for m in out if m.get("step", -1) >= start_step]
    if end_step is not None:
        out = [m for m in out if m.get("step", -1) <= end_step]
    return out


def plot_metrics(metrics: list, output_path: str = "chart.png", title: str = "FaaS RL Agent Metrics"):
    """Generate 4-panel chart."""
    if not metrics:
        raise ValueError("No metrics found for requested range")

    steps = [m["step"] for m in metrics]
    
    # Extract data with defaults for missing keys
    req_rate = [m.get("req_rate", 0) for m in metrics]
    latency = [m.get("latency", 0) for m in metrics]
    cold_starts = [m.get("cold_starts", 0) for m in metrics]
    warm_hits = [m.get("warm_hits", 0) for m in metrics]
    queue_depth = [m.get("queue", 0) for m in metrics]
    containers = [m.get("warm_containers", 1) for m in metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # Top-left: Latency (fallback to RPS for older metrics files)
    if any(v > 0 for v in latency):
        axes[0, 0].plot(steps, latency, color="steelblue", linewidth=1.5)
        axes[0, 0].set_title("Latency (seconds)", fontweight="bold")
        axes[0, 0].set_ylabel("Seconds")
    else:
        axes[0, 0].plot(steps, req_rate, color="steelblue", linewidth=1.5)
        axes[0, 0].set_title("Request Rate (RPS)", fontweight="bold")
        axes[0, 0].set_ylabel("Requests/sec")
    axes[0, 0].set_xlabel("Time Step")
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
    parser.add_argument("--start-step", type=int, default=None, help="Inclusive start step filter")
    parser.add_argument("--end-step", type=int, default=None, help="Inclusive end step filter")
    parser.add_argument("--title", default="FaaS RL Agent Metrics", help="Chart title")
    args = parser.parse_args()
    
    metrics = load_metrics(args.input)
    filtered = filter_metrics(metrics, args.start_step, args.end_step)
    print(f"Loaded {len(metrics)} steps from {args.input}, plotting {len(filtered)}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_metrics(filtered, args.output, title=args.title)
