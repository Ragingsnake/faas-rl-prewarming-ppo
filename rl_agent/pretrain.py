"""
pretrain.py (PPO edition) — offline pre-training on SyntheticFaaSEnv.
PPO collects ROLLOUT_STEPS then updates; no replay buffer fill needed.
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from ppo_agent import PPOAgent, ROLLOUT_STEPS
from environment import SyntheticFaaSEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger(__name__)
PATTERNS = ["daily", "spike", "jitter", "flat"]


def run_episode(agent, env):
    obs, _ = env.reset()
    total_r, cold_total, all_losses = 0.0, 0, []
    while True:
        action, lp, value = agent.select_action(obs)
        next_obs, reward, done, _, info = env.step(action)
        agent.store(obs, action, lp, reward, value, done)
        obs = next_obs; total_r += reward; cold_total += info.get("cold_starts", 0)
        if len(agent.buffer) >= ROLLOUT_STEPS or done:
            if not done:
                with torch.no_grad():
                    _, lv = agent.policy(torch.tensor(obs,dtype=torch.float32).unsqueeze(0))
                    last_val = lv.item()
            else:
                last_val = 0.0
            all_losses.append(agent.learn(last_value=last_val))
        if done:
            break
    mean = lambda k: float(np.mean([l[k] for l in all_losses if k in l])) if all_losses else 0
    return dict(total_reward=total_r, cold_starts=cold_total,
                actor_loss=mean("actor_loss"), critic_loss=mean("critic_loss"),
                entropy=mean("entropy"), clip_frac=mean("clip_frac"))


def pretrain(epochs, out_path, seed):
    agent = PPOAgent(device="cpu")
    log.info("=== PPO Pre-training: %d epochs ===", epochs)
    for ep in range(1, epochs+1):
        pattern = PATTERNS[(ep-1) % len(PATTERNS)]
        stats   = run_episode(agent, SyntheticFaaSEnv(pattern=pattern, seed=seed+ep))
        log.info("Ep %3d/%d  %-6s  R=%8.1f  cold=%4d  actor=%.4f  critic=%.4f  H=%.3f  clip=%.3f",
                 ep, epochs, pattern, stats["total_reward"], stats["cold_starts"],
                 stats["actor_loss"], stats["critic_loss"], stats["entropy"], stats["clip_frac"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(out_path)
    log.info("=== Saved → %s ===", out_path)
    return agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out",    default="checkpoints/pretrained.pt")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()
    pretrain(args.epochs, args.out, args.seed)
