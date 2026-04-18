"""
server.py — FastAPI online control loop
=========================================
Runs in a background thread:
  every STEP_SECONDS → observe → act → learn

Exposes HTTP endpoints for:
  GET  /health          liveness probe
  GET  /metrics         current agent stats (Prometheus text format)
  GET  /status          JSON snapshot
  POST /pause           pause the control loop
  POST /resume          resume the control loop
  POST /train           trigger N extra training gradient steps
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel

from ppo_agent import PPOAgent, ROLLOUT_STEPS
from environment import EnvConfig, FaaSEnv
from metrics_logger import MetricsLogger

# ──────────────────────────────────────────────────────────────
# Config from environment variables
# ──────────────────────────────────────────────────────────────
PROM_URL      = os.getenv("PROMETHEUS_URL",    "http://prometheus:9090")
FAAS_URL      = os.getenv("OPENFAAS_URL",      "http://gateway.openfaas:8080")
FAAS_USER     = os.getenv("OPENFAAS_USER",     "admin")
FAAS_PASS     = os.getenv("OPENFAAS_PASS",     "admin")
FUNCTION_NAME = os.getenv("FAAS_FUNCTION",     "echo-fn")
STEP_SECONDS  = int(os.getenv("STEP_SECONDS",  "15"))
CKPT_PATH     = os.getenv("CHECKPOINT_PATH",   "checkpoints/agent.pt")
PRETRAIN_CKPT = os.getenv("PRETRAIN_CKPT",     "checkpoints/pretrained.pt")
METRICS_PATH  = os.getenv("METRICS_PATH",      "checkpoints/metrics.json")
AGENT_START_MODE = os.getenv("AGENT_START_MODE", "active").strip().lower()
AGENT_LEARN_ENABLED = os.getenv("AGENT_LEARN_ENABLED", "true").strip().lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
log = logging.getLogger("rl-server")

# ──────────────────────────────────────────────────────────────
app    = FastAPI(title="FaaS RL Pre-warmer", version="1.0.0")
_agent: PPOAgent | None = None
_env:   FaaSEnv   | None = None
_metrics: MetricsLogger = MetricsLogger(METRICS_PATH)
_stats: dict = {
    "steps":         0,
    "total_reward":  0.0,
    "last_reward":   0.0,
    "last_loss":     0.0,
    "warm_containers": 0,
    "last_cold_starts": 0,
    "entropy":       0.0,
    "paused":        AGENT_START_MODE == "inactive",
    "agent_active":  AGENT_START_MODE != "inactive",
    "learning_enabled": AGENT_LEARN_ENABLED,
}
_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────
# Control loop (runs in background thread)
# ──────────────────────────────────────────────────────────────
def _control_loop():
    global _agent, _env, _stats

    cfg  = EnvConfig(
        prometheus_url = PROM_URL,
        openfaas_url   = FAAS_URL,
        openfaas_user  = FAAS_USER,
        openfaas_pass  = FAAS_PASS,
        function_name  = FUNCTION_NAME,
        step_seconds   = STEP_SECONDS,
    )
    _env   = FaaSEnv(cfg)
    _agent = PPOAgent(device="cpu")

    # Load pre-trained weights if available
    pt = Path(PRETRAIN_CKPT)
    if pt.exists():
        _agent.load(str(pt))
        log.info("Loaded pre-trained checkpoint from %s", pt)
    elif Path(CKPT_PATH).exists():
        _agent.load(CKPT_PATH)
        log.info("Resumed from existing checkpoint %s", CKPT_PATH)

    state = _env.reset()
    last_losses: dict = {}

    while True:
        with _lock:
            agent_active = _stats["agent_active"]
            learning_enabled = _stats["learning_enabled"]

        # ── act ───────────────────────────────────────────────
        if agent_active:
            action, log_prob, value = _agent.select_action(
                state, current_reps=_env._current_warm, rr=_env._prev_req_rate
            )
        else:
            action, log_prob, value = 2, 0.0, 0.0  # hold while inactive
        raw_delta = _agent.action_delta(action)
        mode_text = "active" if agent_active else "inactive"
        log.info(
            f"Step {_stats['steps']}: Agent mode={mode_text}, Action {action} "
            f"(Raw Delta: {raw_delta} containers)"
        )
        
        next_state, reward, done, info = _env.step(action)
        applied_delta = info.get("applied_delta", 0)
        
        log.info(
            f"Result -> Reps: {info.get('warm_containers')}, Traffic: {info.get('req_rate'):.1f} RPS, "
            f"Queue: {info.get('queue', 0):.1f}, Applied Delta: {applied_delta}, Reward: {reward:.2f}"
        )
        # ── store ─────────────────────────────────────────────
        if agent_active:
            _agent.store(state, action, log_prob, reward, value, done)

        state = next_state if not done else _env.reset()

        # ── learn — only when rollout buffer is full ───────────
        # PPO must accumulate ROLLOUT_STEPS transitions before updating.
        # Calling learn() with 1 sample causes std()=0 on the single-element
        # advantages tensor → degenerate gradients → NaN weights → crash.
        if agent_active and learning_enabled and len(_agent.buffer) >= ROLLOUT_STEPS:
            # Bootstrap V(s) for the final non-terminal state
            if not done:
                import torch as _torch
                with _torch.no_grad():
                    s_t = _torch.tensor(state, dtype=_torch.float32).unsqueeze(0)
                    _, last_val = _agent.policy(s_t)
                    bootstrap = last_val.item()
            else:
                bootstrap = 0.0
            last_losses = _agent.learn(last_value=bootstrap)

        # ── update stats & log metrics ────────────────────────
        with _lock:
            _stats["steps"]           += 1
            _stats["total_reward"]    += reward
            _stats["last_reward"]      = reward
            _stats["last_loss"]        = last_losses.get("actor_loss", 0.0)
            _stats["warm_containers"]  = info.get("warm_containers", 0)
            _stats["last_cold_starts"] = info.get("cold_starts", 0)
            _stats["entropy"]          = last_losses.get("entropy", 0.0)
            _stats["agent_active"]     = agent_active
            _stats["paused"]           = not agent_active
            _stats["learning_enabled"] = learning_enabled
            
            # Log to metrics file
            _metrics.log_step(
                step=_stats["steps"],
                latency=info.get("queue", 0),
                cold_starts=info.get("cold_starts", 0),
                warm_hits=info.get("warm_hits", 0),
                queue=info.get("queue", 0),
                warm_containers=info.get("warm_containers", 0),
                req_rate=info.get("req_rate", 0.0),
                reward=reward,
                agent_active=agent_active,
            )

        # ── periodic checkpoint (only after at least one update) ──
        if _stats["steps"] % 20 == 0 and last_losses:
            Path(CKPT_PATH).parent.mkdir(parents=True, exist_ok=True)
            _agent.save(CKPT_PATH)


# ──────────────────────────────────────────────────────────────
# HTTP endpoints
# ──────────────────────────────────────────────────────────────
@app.on_event("startup")
def _startup():
    t = threading.Thread(target=_control_loop, daemon=True, name="rl-loop")
    t.start()
    log.info("RL control loop started (start mode: %s)", AGENT_START_MODE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    with _lock:
        return dict(_stats)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Expose agent metrics in Prometheus text format."""
    with _lock:
        s = dict(_stats)
    lines = [
        "# HELP rl_agent_steps Total control-loop steps",
        "# TYPE rl_agent_steps counter",
        f'rl_agent_steps {s["steps"]}',
        "# HELP rl_agent_reward Last step reward",
        "# TYPE rl_agent_reward gauge",
        f'rl_agent_reward {s["last_reward"]:.4f}',
        "# HELP rl_agent_warm_containers Current pre-warmed containers",
        "# TYPE rl_agent_warm_containers gauge",
        f'rl_agent_warm_containers {s["warm_containers"]}',
        "# HELP rl_agent_cold_starts Cold starts in last step",
        "# TYPE rl_agent_cold_starts gauge",
        f'rl_agent_cold_starts {s["last_cold_starts"]}',
        "# HELP rl_agent_entropy Policy entropy (exploration)",
        "# TYPE rl_agent_entropy gauge",
        f'rl_agent_entropy {s["entropy"]:.4f}',
        "# HELP rl_agent_loss Last training loss",
        "# TYPE rl_agent_loss gauge",
        f'rl_agent_loss {s["last_loss"]:.6f}',
    ]
    return "\n".join(lines) + "\n"


@app.post("/pause")
def pause():
    with _lock:
        _stats["agent_active"] = False
        _stats["paused"] = True
    return {"paused": True, "agent_active": False}


@app.post("/resume")
def resume():
    with _lock:
        _stats["agent_active"] = True
        _stats["paused"] = False
    return {"paused": False, "agent_active": True}


@app.post("/set-mode")
def set_mode(mode: str):
    m = mode.strip().lower()
    if m not in {"active", "inactive"}:
        return {"error": "mode must be 'active' or 'inactive'"}
    active = m == "active"
    with _lock:
        _stats["agent_active"] = active
        _stats["paused"] = not active
    return {"mode": m, "agent_active": active}


@app.post("/set-learning")
def set_learning(enabled: bool):
    with _lock:
        _stats["learning_enabled"] = bool(enabled)
    return {"learning_enabled": bool(enabled)}


class TrainRequest(BaseModel):
    steps: int = 100


@app.post("/train")
def manual_train(req: TrainRequest):
    """Trigger extra gradient steps (useful after pre-training upload)."""
    if _agent is None:
        return {"error": "agent not ready"}
    losses = []
    for _ in range(req.steps):
        l = _agent.learn()
        if l is not None:
            losses.append(l)
    return {
        "gradient_steps": len(losses),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
    }


@app.post("/save-metrics")
def save_metrics(
    start_step: int | None = None,
    end_step: int | None = None,
    output: str = "checkpoints/chart.png",
    title: str = "FaaS RL Agent Metrics",
):
    """Save collected metrics to JSON and generate chart."""
    _metrics.save()
    import subprocess
    try:
        cmd = ["python", "visualize_metrics.py", "--input", METRICS_PATH, "--output", output, "--title", title]
        if start_step is not None:
            cmd += ["--start-step", str(start_step)]
        if end_step is not None:
            cmd += ["--end-step", str(end_step)]
        subprocess.run(
            cmd,
            check=True,
            cwd=str(Path(__file__).parent),
        )
        return {"saved": METRICS_PATH, "chart": output, "start_step": start_step, "end_step": end_step}
    except subprocess.CalledProcessError as e:
        return {"error": str(e)}


@app.post("/reset-metrics")
def reset_metrics():
    _metrics.reset()
    return {"reset": True}


@app.get("/download-chart")
def download_chart(name: str = "chart.png"):
    """Download a generated chart PNG from checkpoints/."""
    safe_name = Path(name).name
    chart_path = Path("checkpoints") / safe_name
    if chart_path.exists():
        return FileResponse(chart_path, media_type="image/png", filename=safe_name)
    return {"error": f"Chart not found: {safe_name}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
