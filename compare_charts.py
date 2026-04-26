"""
compare_charts.py - Vẽ biểu đồ so sánh Active vs Inactive chồng lên nhau.
Chạy từ thư mục gốc: python compare_charts.py
"""
import json
import csv
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE = Path(__file__).parent
METRICS_FILE = BASE / "rl_agent/checkpoints/metrics.json"
CSV_FILE     = BASE / "locust/results/case_ranges.csv"
OUT_DIR      = BASE / "rl_agent/checkpoints/comparison"

CASES = ["stable_low", "stable_high", "gradual_ramp", "sudden_spike"]
CASE_LABELS = {
    "stable_low":    "Tải Ổn Định Thấp (Stable Low)",
    "stable_high":   "Tải Ổn Định Cao (Stable High)",
    "gradual_ramp":  "Tăng Tải Dần (Gradual Ramp)",
    "sudden_spike":  "Tải Đột Biến (Sudden Spike)",
}


def load_data():
    with open(METRICS_FILE, encoding="utf-8") as f:
        metrics = json.load(f)
    with open(CSV_FILE, encoding="utf-8-sig") as f:
        ranges = list(csv.DictReader(f))
    return metrics, ranges


def get_slice(metrics, start, end):
    return [m for m in metrics if start <= m.get("step", -1) <= end]


def smooth(data, window=2):
    if len(data) < window:
        return data
    result = []
    for i in range(len(data)):
        lo = max(0, i - window + 1)
        result.append(np.mean(data[lo:i+1]))
    return result


def plot_comparison(case, inactive_m, active_m, out_path):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"So sánh Baseline vs RL Agent - {CASE_LABELS.get(case, case)}",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    def vals(ms, key):
        return smooth([m.get(key, 0) for m in ms])

    panels = [
        ("Latency P95 (giây)",         "latency",         "Seconds"),
        ("Cold Starts (ước tính)",      "cold_starts",     "Requests"),
        ("Queue Depth (ước tính)",      "queue",           "Requests"),
        ("Số Container đang chạy",      "warm_containers", "Containers"),
    ]

    colors = {"inactive": ("#e05454", "#fbb4b4"), "active": ("#2196F3", "#90CAF9")}

    for ax, (title, key, ylabel) in zip(axes, panels):
        xi = list(range(len(inactive_m)))
        xa = list(range(len(active_m)))

        yi = vals(inactive_m, key)
        ya = vals(active_m,   key)

        ax.plot(xi, yi, color=colors["inactive"][0], linewidth=2,
                label="Baseline (Không có AI)", zorder=3)
        ax.fill_between(xi, 0, yi, color=colors["inactive"][1], alpha=0.3)

        ax.plot(xa, ya, color=colors["active"][0], linewidth=2,
                linestyle="--", label="RL Agent (Có AI)", zorder=3)
        ax.fill_between(xa, 0, ya, color=colors["active"][1], alpha=0.3)

        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("Bước thời gian (Time Step)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    print("Đang tải dữ liệu...\n")
    metrics, ranges = load_data()

    lookup = {}
    for row in ranges:
        key = (row["mode"], row["case"])
        lookup[key] = (int(row["start_step"]), int(row["end_step"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        if ("inactive", case) not in lookup or ("active", case) not in lookup:
            print(f"  Bỏ qua {case}: không tìm thấy đủ dữ liệu cả 2 chế độ")
            continue

        s_in, e_in = lookup[("inactive", case)]
        s_ac, e_ac = lookup[("active",   case)]

        inactive_m = get_slice(metrics, s_in, e_in)
        active_m   = get_slice(metrics, s_ac, e_ac)

        out = OUT_DIR / f"compare_{case}.png"
        print(f"Vẽ biểu đồ: {case} ...")
        plot_comparison(case, inactive_m, active_m, str(out))

    print(f"\nHoàn thành! Mở thư mục {OUT_DIR} để xem 4 biểu đồ so sánh.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
