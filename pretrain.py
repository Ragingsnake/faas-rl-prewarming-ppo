"""
pretrain.py - Pre-train PPO agent offline on SyntheticFaaSEnv.
Chạy từ thư mục rl_agent/: python pretrain.py

Sau khi xong, file checkpoints/pretrained.pt sẽ được tạo ra.
Server.py tự động load file này khi khởi động nếu tồn tại.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np

# Thêm thư mục rl_agent vào sys.path
sys.path.insert(0, str(Path(__file__).parent / "rl_agent"))

from ppo_agent import PPOAgent, ROLLOUT_STEPS
from environment import SyntheticFaaSEnv, MIN_WARM, MAX_WARM

# ── Cấu hình ──────────────────────────────────────────────────
TOTAL_STEPS    = 8000   # Tổng số bước mô phỏng
SAVE_PATH      = "rl_agent/checkpoints/pretrained.pt"
PATTERNS       = ["daily", "spike", "jitter", "zero"]  # Các pattern tải
PRINT_INTERVAL = 500

def pretrain():
    os.makedirs(Path(SAVE_PATH).parent, exist_ok=True)
    agent = PPOAgent(device="cpu")

    total_steps = 0
    episode     = 0
    last_losses: dict = {}

    print(f"Pre-training bat dau: {TOTAL_STEPS} buoc tren {len(PATTERNS)} pattern...")
    print(f"Checkpoint se luu tai: {SAVE_PATH}\n")

    while total_steps < TOTAL_STEPS:
        pattern = PATTERNS[episode % len(PATTERNS)]
        env     = SyntheticFaaSEnv(pattern=pattern, seed=episode)
        obs, _  = env.reset()
        episode_reward = 0.0
        done = False
        step_in_ep = 0

        while not done and total_steps < TOTAL_STEPS:
            # Chọn action
            action, log_prob, value = agent.select_action(
                obs, current_reps=env._warm, rr=env._prev_rr
            )
            next_obs, reward, done, _, info = env.step(action)
            agent.store(obs, action, log_prob, reward, value, done)

            episode_reward += reward
            obs = next_obs
            total_steps += 1
            step_in_ep  += 1

            # Học sau mỗi ROLLOUT_STEPS bước
            if len(agent.buffer) >= ROLLOUT_STEPS:
                import torch as _torch
                with _torch.no_grad():
                    s_t = _torch.tensor(obs, dtype=_torch.float32).unsqueeze(0)
                    _, last_val = agent.policy(s_t)
                    bootstrap = last_val.item() if not done else 0.0
                last_losses = agent.learn(last_value=bootstrap)

            # In tiến độ
            if total_steps % PRINT_INTERVAL == 0:
                loss_str = f"loss={last_losses.get('total_loss', 0):.4f}" if last_losses else "warming up..."
                print(f"  Step {total_steps:>5}/{TOTAL_STEPS} | "
                      f"Pattern={pattern:<8} | "
                      f"Ep_reward={episode_reward:>7.2f} | "
                      f"Reps={env._warm} | "
                      f"{loss_str}")

        episode += 1

    # Lưu checkpoint
    agent.save(SAVE_PATH)
    print(f"\nPre-training hoan thanh!")
    print(f"Da luu: {SAVE_PATH}")
    print(f"Tong episodes: {episode} | Tong steps: {total_steps}")


if __name__ == "__main__":
    pretrain()
