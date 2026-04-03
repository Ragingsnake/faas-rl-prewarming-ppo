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
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ppo_agent import PPOAgent, ROLLOUT_STEPS
from environment import EnvConfig, FaaSEnv

# ──────────────────────────────────────────────────────────────
# Config from environment variables
# ──────────────────────────────────────────────────────────────
PROM_URL      = os.getenv("PROMETHEUS_URL",    "http://prometheus:9090")
FAAS_URL      = os.getenv("OPENFAAS_URL",      "http://gateway.openfaas:8080")
FAAS_USER     = os.getenv("OPENFAAS_USER",     "admin")
FAAS_PASS     = os.getenv("OPENFAAS_PASS",     "admin")
FUNCTION_NAME = os.getenv("FAAS_FUNCTION",     "echo-fn")
STEP_SECONDS  = int(os.getenv("STEP_SECONDS",  "15"))
CKPT_PATH     = os.getenv("CHECKPOINT_PATH",   "/checkpoints/agent.pt")
PRETRAIN_CKPT = os.getenv("PRETRAIN_CKPT",     "/checkpoints/pretrained.pt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
log = logging.getLogger("rl-server")

# ──────────────────────────────────────────────────────────────
app    = FastAPI(title="FaaS RL Pre-warmer", version="1.0.0")
_agent: PPOAgent | None = None
_env:   FaaSEnv   | None = None
_stats: dict = {
    "steps":         0,
    "total_reward":  0.0,
    "last_reward":   0.0,
    "last_loss":     0.0,
    "warm_containers": 0,
    "last_cold_starts": 0,
    "entropy":       0.0,
    "paused":        False,
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
            paused = _stats["paused"]

        if paused:
            time.sleep(1)
            continue

        # ── act ───────────────────────────────────────────────
        action, log_prob, value = _agent.select_action(state)
        next_state, reward, done, info = _env.step(action)

        # ── store ─────────────────────────────────────────────
        _agent.store(state, action, log_prob, reward, value, done)

        state = next_state if not done else _env.reset()

        # ── learn — only when rollout buffer is full ───────────
        # PPO must accumulate ROLLOUT_STEPS transitions before updating.
        # Calling learn() with 1 sample causes std()=0 on the single-element
        # advantages tensor → degenerate gradients → NaN weights → crash.
        if len(_agent.buffer) >= ROLLOUT_STEPS:
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

        # ── update stats ──────────────────────────────────────
        with _lock:
            _stats["steps"]           += 1
            _stats["total_reward"]    += reward
            _stats["last_reward"]      = reward
            _stats["last_loss"]        = last_losses.get("actor_loss", 0.0)
            _stats["warm_containers"]  = info.get("warm_containers", 0)
            _stats["last_cold_starts"] = info.get("cold_starts", 0)
            _stats["entropy"]          = last_losses.get("entropy", 0.0)

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
    log.info("RL control loop started")


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
        _stats["paused"] = True
    return {"paused": True}


@app.post("/resume")
def resume():
    with _lock:
        _stats["paused"] = False
    return {"paused": False}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)