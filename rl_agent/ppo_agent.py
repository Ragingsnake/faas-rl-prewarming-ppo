"""
ppo_agent.py — Proximal Policy Optimization for FaaS Pre-warming
=================================================================

Replaces the Double DQN agent. Key differences:

  DQN                          PPO
  ──────────────────────────   ────────────────────────────────────
  Off-policy (replay buffer)   On-policy (rollout buffer, cleared)
  ε-greedy exploration         Stochastic policy + entropy bonus
  Learns from old experience   Learns only from fresh rollouts
  One network (Q)              Two heads: actor π + critic V
  TD(0) targets                GAE(γ,λ) advantage estimates
  Hard target-net copies       No target net needed

Architecture
────────────
  Shared backbone → actor head (softmax over 5 actions)
                  → critic head (scalar state value V(s))

  Using a shared backbone lets the two heads share representations
  of arrival rate, queue depth, and time — features that matter for
  both "what to do" and "how good is this state".

Rollout buffer
──────────────
  Collect ROLLOUT_STEPS transitions online, then run K_EPOCHS
  gradient steps over the whole buffer in mini-batches, then clear.
  No experience replay — PPO is on-policy by design.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger(__name__)

# ── Hyper-parameters ──────────────────────────────────────────
STATE_DIM     = 9
ACTION_DIM    = 5
ACTION_MAP    = [-2, -1, 0, 1, 2]

GAMMA         = 0.99      # discount
GAE_LAMBDA    = 0.95      # GAE smoothing (λ)
CLIP_EPS      = 0.2       # PPO clip ratio ε
K_EPOCHS      = 10        # gradient epochs per rollout
BATCH_SIZE    = 64        # mini-batch size within an epoch
ROLLOUT_STEPS = 64        # steps to collect before each update
LR            = 3e-4
ENTROPY_COEF  = 0.02      # slightly higher than notebook (was 0.01) for more exploration
VALUE_COEF    = 0.5       # critic loss weight
MAX_GRAD_NORM = 0.5       # gradient clipping


# ── Shared Actor-Critic network ───────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),      # stabilises mixed-scale state features
            nn.Tanh(),              # Tanh (not ReLU) is standard for PPO —
            nn.Linear(256, 256),    # bounded activations → stabler gradients
            nn.Tanh(),
        )

        # Actor head: outputs action probabilities
        self.actor_head = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic head: outputs scalar V(s)
        self.critic_head = nn.Linear(256, 1)

        # Orthogonal init (recommended for PPO)
        self._init_weights()

    def _init_weights(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head[0].weight, gain=0.01)
        nn.init.zeros_(self.actor_head[0].bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        probs    = self.actor_head(features)
        value    = self.critic_head(features)
        return probs, value


# ── Rollout buffer (replaces replay buffer) ───────────────────
class RolloutBuffer:
    """
    Stores one rollout's worth of on-policy experience.
    Cleared after every PPO update — no stale data.
    """
    def __init__(self):
        self.states:    List[np.ndarray] = []
        self.actions:   List[int]        = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []

    def push(self, state, action, log_prob, reward, value, done):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def __len__(self) -> int:
        return len(self.rewards)

    def clear(self):
        self.__init__()


# ── PPO Agent ─────────────────────────────────────────────────
class PPOAgent:
    """
    On-policy PPO with GAE advantage estimation.

    Control loop:
      1. Collect ROLLOUT_STEPS transitions using current policy π_old
      2. Compute GAE advantages and returns
      3. Run K_EPOCHS mini-batch gradient steps with clipped objective
      4. Clear rollout buffer
      5. Repeat
    """

    def __init__(self, device: str = "cpu"):
        self.device  = torch.device(device)
        self.policy  = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR, eps=1e-5)
        self.buffer  = RolloutBuffer()
        self.steps   = 0

    # ── action selection (inference mode) ────────────────────
    @torch.no_grad()                  # FIX: was missing → wasted memory on grads
    def select_action(self, state: np.ndarray, current_reps: int = 1, rr: float = 0.0) -> Tuple[int, float, float]:
        """Returns (action_idx, log_prob, value). Masks invalid actions."""
        s     = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs, value = self.policy(s)
        
        # Mask invalid actions
        valid_mask = self._get_valid_action_mask(current_reps, rr)
        masked_probs = probs.clone()
        masked_probs[0, ~valid_mask] = 0.0
        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
        
        dist  = torch.distributions.Categorical(masked_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()
    
    def _get_valid_action_mask(self, current_reps: int, rr: float) -> torch.Tensor:
        """Return boolean mask [True=valid, False=blocked]."""
        import math
        from environment import MIN_WARM, MAX_WARM, PER_REPLICA_RPS
        
        valid = torch.ones(ACTION_DIM, dtype=torch.bool, device=self.device)
        min_safe_reps = max(MIN_WARM, math.ceil(rr / PER_REPLICA_RPS))
        
        for i, delta in enumerate(ACTION_MAP):
            new_reps = current_reps + delta
            if new_reps > MAX_WARM or new_reps < min_safe_reps:
                valid[i] = False
        
        return valid

    def action_delta(self, action_idx: int) -> int:
        return ACTION_MAP[action_idx]

    # ── store one transition ──────────────────────────────────
    def store(self, state, action, log_prob, reward, value, done):
        self.buffer.push(state, action, log_prob, reward, value, done)

    # ── GAE advantage computation ─────────────────────────────
    def _compute_gae(self, last_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalised Advantage Estimation:

          δ_t = r_t + γ·V(s_{t+1})·(1-d_t) - V(s_t)
          A_t = δ_t + (γλ)·(1-d_t)·A_{t+1}

        last_value: V(s_T) — bootstrap for the final non-terminal state.
        This is the FIX over the notebook which hard-coded next_value=0
        even for non-terminal final states (continuous env, done=always False).
        """
        rewards  = self.buffer.rewards
        values   = self.buffer.values
        dones    = self.buffer.dones
        n        = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        gae        = 0.0
        next_val   = last_value   # FIX: bootstrap from critic, not 0

        for i in reversed(range(n)):
            mask    = 1.0 - float(dones[i])
            delta   = rewards[i] + GAMMA * next_val * mask - values[i]
            gae     = delta + GAMMA * GAE_LAMBDA * mask * gae
            advantages[i] = gae
            next_val      = values[i]

        returns = advantages + np.array(values, dtype=np.float32)
        return (
            torch.tensor(advantages, dtype=torch.float32, device=self.device),
            torch.tensor(returns,    dtype=torch.float32, device=self.device),
        )

    # ── PPO update ────────────────────────────────────────────
    def learn(self, last_value: float = 0.0) -> dict:
        """
        Run K_EPOCHS of mini-batch PPO updates over the current rollout.
        Returns dict of mean losses for logging.
        """
        if len(self.buffer) == 0:
            return {}

        advantages, returns = self._compute_gae(last_value)

        # Normalise advantages (zero mean, unit std) — standard PPO trick
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert buffer to tensors
        states    = torch.tensor(np.stack(self.buffer.states), dtype=torch.float32, device=self.device)
        actions   = torch.tensor(self.buffer.actions,   dtype=torch.long,    device=self.device)
        old_lps   = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)

        n = len(states)
        actor_losses, critic_losses, entropies, clip_fracs = [], [], [], []

        for _ in range(K_EPOCHS):
            idx = np.random.permutation(n)

            for start in range(0, n, BATCH_SIZE):
                b = idx[start: start + BATCH_SIZE]

                s   = states[b]
                a   = actions[b]
                adv = advantages[b]
                ret = returns[b]
                olp = old_lps[b]

                probs, values_new = self.policy(s)
                dist = torch.distributions.Categorical(probs)

                new_lps = dist.log_prob(a)
                entropy = dist.entropy().mean()

                # Probability ratio r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(new_lps - olp)

                # Clipped surrogate objective
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                actor_loss  = -torch.min(surr1, surr2).mean()

                # FIX: squeeze(-1) not squeeze() — safe for batch_size=1
                critic_loss = nn.functional.mse_loss(values_new.squeeze(-1), ret)

                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                # FIX: gradient clipping (was missing)
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                clip_frac = ((ratio - 1).abs() > CLIP_EPS).float().mean().item()
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())
                clip_fracs.append(clip_frac)

        self.steps += n
        self.buffer.clear()

        return {
            "actor_loss":  float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy":     float(np.mean(entropies)),
            "clip_frac":   float(np.mean(clip_fracs)),
        }

    # ── persistence ───────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "policy":    self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps":     self.steps,
        }, path)
        log.info("PPO checkpoint saved → %s", path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.steps = ck["steps"]
        log.info("PPO checkpoint loaded from %s (step %d)", path, self.steps)
