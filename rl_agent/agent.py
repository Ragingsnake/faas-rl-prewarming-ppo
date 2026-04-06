"""
PPO Agent for FaaS Pre-warming.

State  (9-dim):
  [req_rate, req_rate_delta, hour_sin, hour_cos,
   dow_sin,  dow_cos,        warm_containers,
   cold_start_ratio,         queue_depth]

Actions (5 discrete):
  0 → -2 warm containers
  1 → -1 warm containers
  2 →  0 (hold)
  3 → +1 warm containers
  4 → +2 warm containers

Reward:
  R = -5.0 * cold_starts_this_step
      - 0.2 * max(0, idle_warm - 1)
      + 2.0 * warm_hits_this_step
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────────
STATE_DIM   = 9
ACTION_DIM  = 5          # {-2, -1, 0, +1, +2}
ACTION_MAP  = [-2, -1, 0, 1, 2]

GAMMA        = 0.99
LR           = 1e-3
BATCH_SIZE   = 64
BUFFER_CAP   = 20_000
TARGET_UPDATE = 200       # steps between hard target-net copies
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 5_000      # steps to decay over


# ──────────────────────────────────────────────────────────────
# Neural network
# ──────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# Replay buffer
# ──────────────────────────────────────────────────────────────
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_CAP):
        self._buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buf.append((
            np.asarray(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int = BATCH_SIZE) -> List[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)

    # ── serialisation (persist buffer across restarts) ──────
    def save(self, path: str):
        import pickle
        Path(path).write_bytes(pickle.dumps(list(self._buf)))
        log.info("Replay buffer saved (%d transitions) → %s", len(self), path)

    def load(self, path: str):
        import pickle
        data: list = pickle.loads(Path(path).read_bytes())
        self._buf.extend(data)
        log.info("Replay buffer loaded (%d transitions) from %s", len(self), path)


# ──────────────────────────────────────────────────────────────
# DQN agent
# ──────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.q_net      = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self._sync_target()
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer     = ReplayBuffer()
        self.steps      = 0
        self._eps       = EPS_START

    # ── epsilon (ε-greedy) ───────────────────────────────────
    @property
    def epsilon(self) -> float:
        self._eps = EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.steps / EPS_DECAY
        )
        return self._eps

    # ── action selection ─────────────────────────────────────
    def select_action(self, state: np.ndarray, exploit: bool = False) -> int:
        """Return action index (0-4). Set exploit=True for inference."""
        if not exploit and random.random() < self.epsilon:
            return random.randrange(ACTION_DIM)

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_net(s).argmax(dim=1).item())

    def action_delta(self, action_idx: int) -> int:
        """Convert action index → container delta."""
        return ACTION_MAP[action_idx]

    # ── learning ─────────────────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        """One gradient step. Returns loss or None if buffer too small."""
        if len(self.buffer) < BATCH_SIZE:
            return None

        batch = self.buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        S  = torch.tensor(np.stack(states),      dtype=torch.float32, device=self.device)
        A  = torch.tensor(actions,               dtype=torch.long,    device=self.device).unsqueeze(1)
        R  = torch.tensor(rewards,               dtype=torch.float32, device=self.device).unsqueeze(1)
        S2 = torch.tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        D  = torch.tensor(dones,                 dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q values
        q_vals = self.q_net(S).gather(1, A)

        # Target Q values (Double DQN: online net selects, target net evaluates)
        with torch.no_grad():
            best_actions = self.q_net(S2).argmax(dim=1, keepdim=True)
            q_next = self.target_net(S2).gather(1, best_actions)
            q_target = R + GAMMA * q_next * (1 - D)

        loss = nn.functional.smooth_l1_loss(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self._sync_target()
            log.debug("Target network synced at step %d", self.steps)

        return loss.item()

    def _sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── persistence ──────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "steps":      self.steps,
        }, path)
        log.info("Agent checkpoint saved → %s", path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ck["q_net"])
        self.target_net.load_state_dict(ck["target_net"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.steps = ck["steps"]
        log.info("Agent checkpoint loaded from %s (step %d)", path, self.steps)
