"""
Unit tests for PPO agent and SyntheticFaaSEnv.
Run: pytest tests/ -v  (from repo root, PYTHONPATH=rl_agent)
"""
import sys
from pathlib import Path
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "rl_agent"))

from ppo_agent import PPOAgent, RolloutBuffer, ACTION_MAP, STATE_DIM, ACTION_DIM, ROLLOUT_STEPS
from environment import SyntheticFaaSEnv, MAX_WARM, MIN_WARM


class TestRolloutBuffer:
    def test_push_and_clear(self):
        buf = RolloutBuffer()
        for _ in range(50):
            buf.push(np.zeros(STATE_DIM), 0, -0.5, 1.0, 0.5, False)
        assert len(buf) == 50
        buf.clear()
        assert len(buf) == 0

    def test_stores_correct_types(self):
        buf = RolloutBuffer()
        buf.push(np.ones(STATE_DIM), 3, -0.3, 2.5, 1.2, True)
        assert buf.actions[0] == 3
        assert buf.dones[0] is True
        assert isinstance(buf.log_probs[0], float)


class TestPPOAgent:
    def test_select_action_returns_valid(self):
        agent = PPOAgent(device="cpu")
        s = np.random.rand(STATE_DIM).astype(np.float32)
        action, lp, val = agent.select_action(s)
        assert 0 <= action < ACTION_DIM
        assert isinstance(lp, float)
        assert isinstance(val, float)

    def test_select_action_no_grad(self):
        """select_action must not track gradients."""
        agent = PPOAgent(device="cpu")
        s = np.random.rand(STATE_DIM).astype(np.float32)
        agent.select_action(s)
        for p in agent.policy.parameters():
            assert p.grad is None, "select_action should not compute grads"

    def test_learn_returns_losses(self):
        agent = PPOAgent(device="cpu")
        for _ in range(ROLLOUT_STEPS):
            s = np.random.rand(STATE_DIM).astype(np.float32)
            agent.store(s, np.random.randint(ACTION_DIM), -0.5, 1.0, 0.5, False)
        losses = agent.learn(last_value=0.0)
        assert "actor_loss"  in losses
        assert "critic_loss" in losses
        assert "entropy"     in losses
        assert "clip_frac"   in losses

    def test_buffer_cleared_after_learn(self):
        agent = PPOAgent(device="cpu")
        for _ in range(ROLLOUT_STEPS):
            agent.store(np.zeros(STATE_DIM), 0, 0.0, 0.0, 0.0, False)
        agent.learn(last_value=0.0)
        assert len(agent.buffer) == 0, "Buffer must be cleared after PPO update"

    def test_clip_frac_bounded(self):
        agent = PPOAgent(device="cpu")
        for _ in range(ROLLOUT_STEPS):
            s = np.random.rand(STATE_DIM).astype(np.float32)
            agent.store(s, np.random.randint(ACTION_DIM), -0.3, 0.5, 0.0, False)
        losses = agent.learn(0.0)
        assert 0.0 <= losses["clip_frac"] <= 1.0

    def test_action_map_coverage(self):
        assert set(ACTION_MAP) == {-2, -1, 0, 1, 2}

    def test_save_load(self, tmp_path):
        agent = PPOAgent(device="cpu")
        for _ in range(ROLLOUT_STEPS):
            agent.store(np.zeros(STATE_DIM), 0, 0.0, 0.0, 0.0, False)
        agent.learn(0.0)
        path = str(tmp_path / "ckpt.pt")
        agent.save(path)
        agent2 = PPOAgent(device="cpu")
        agent2.load(path)
        s = np.random.rand(STATE_DIM).astype(np.float32)
        t = torch.tensor(s).unsqueeze(0)
        with torch.no_grad():
            p1, v1 = agent.policy(t)
            p2, v2 = agent2.policy(t)
        assert torch.allclose(p1, p2)


class TestSyntheticEnv:
    @pytest.mark.parametrize("pattern", ["daily", "spike", "jitter", "flat"])
    def test_step_valid(self, pattern):
        env = SyntheticFaaSEnv(pattern=pattern, seed=0)
        obs, _ = env.reset()
        assert obs.shape == (STATE_DIM,)
        # FIX validation: obs should be in approx [-1,1], not near-zero
        assert obs.max() > 0.01, "State must not be near-zero (double-norm bug)"
        for _ in range(10):
            ns, r, done, _, info = env.step(np.random.randint(ACTION_DIM))
            assert ns.shape == (STATE_DIM,)
            assert isinstance(r, float)

    def test_no_double_normalisation(self):
        """State values must not be squashed to ~0 (the notebook bug)."""
        env = SyntheticFaaSEnv(pattern="flat", seed=1)
        env.reset()
        obs, _, _, _, _ = env.step(2)
        # At flat 30 RPS: req_rate = 30/500 = 0.06, not 0.0006
        assert obs[0] > 0.01, f"req_rate={obs[0]} looks double-normalised"

    def test_warm_clamped(self):
        env = SyntheticFaaSEnv(seed=0); env.reset()
        env._warm = MAX_WARM; env.step(4)
        assert env._warm <= MAX_WARM
        env._warm = MIN_WARM; env.step(0)
        assert env._warm >= MIN_WARM

    def test_episode_terminates(self):
        env = SyntheticFaaSEnv(pattern="flat", seed=0); env.reset()
        done, steps = False, 0
        while not done:
            _, _, done, _, _ = env.step(2); steps += 1
        assert steps == 1440

    def test_mem_updates(self):
        """mem must change across steps (was static 0.2 in notebook).
        Requires idle containers: use action=4 (+2) so warm grows above
        what flat-30-RPS needs (needed=2), creating idle containers."""
        env = SyntheticFaaSEnv(pattern="flat", seed=0); env.reset()
        mems = set()
        for _ in range(20):
            _, _, _, _, info = env.step(4)   # +2 each step → creates idle containers
            mems.add(round(info["mem"], 4))
        assert len(mems) > 1, f"mem must not be static (got {mems})"

    def test_reward_sign(self):
        # Use flat 30 RPS at t=0. 5 containers (MAX_WARM=5) handles it fine,
        # 1 container (~15 RPS capacity) cold-starts most requests.
        # Action=2 is hold (Δ=0), so _warm stays where we set it.
        env_many = SyntheticFaaSEnv(pattern="flat", seed=7); env_many.reset(); env_many._warm=5
        env_few  = SyntheticFaaSEnv(pattern="flat", seed=7); env_few.reset();  env_few._warm=1
        r_many = [env_many.step(2)[1] for _ in range(30)]
        r_few  = [env_few.step(2)[1]  for _ in range(30)]
        assert np.mean(r_many) > np.mean(r_few), (
            f"5 containers should outperform 1 at 30 RPS flat load "
            f"(got many={np.mean(r_many):.3f}, few={np.mean(r_few):.3f})"
        )