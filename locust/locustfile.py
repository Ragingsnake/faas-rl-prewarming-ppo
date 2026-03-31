"""
Locust Load Test — FaaS RL Pre-warmer
=======================================

5 Scenarios (run sequentially via LoadTestShape):

  1. STABLE_LOW    — steady 10 RPS for 3 min   → agent should hold 1-2 warm
  2. STABLE_HIGH   — steady 60 RPS for 3 min   → agent should ramp up warm pool
  3. GRADUAL_RAMP  — 5 → 120 RPS over 5 min    → agent should track the ramp
  4. SUDDEN_SPIKE  — 5 RPS → 200 RPS → 5 RPS   → agent must react fast
  5. JITTERY       — random between 5-180 RPS   → tests agent under uncertainty

Run:
    locust -f locustfile.py \
           --host http://<OPENFAAS_GATEWAY>:8080 \
           --headless --run-time 19m \
           --html report.html

Set OPENFAAS_USER / OPENFAAS_PASS env vars for basic auth.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Generator, List, Tuple

from locust import HttpUser, TaskSet, between, events, task
from locust.env import Environment
from locust.shape import LoadTestShape

# ──────────────────────────────────────────────────────────────
FAAS_USER = os.getenv("OPENFAAS_USER", "admin")
FAAS_PASS = os.getenv("OPENFAAS_PASS", "admin")
FUNCTION  = os.getenv("FAAS_FUNCTION", "echo-fn")

# ──────────────────────────────────────────────────────────────
# Scenario definitions (duration_s, target_rps, description)
# ──────────────────────────────────────────────────────────────
ScenarioPoint = Tuple[int, int, str]   # (elapsed_s, users, label)

# 1 Locust "user" ≈ 1 RPS with wait_time=between(0.8, 1.2)
# So target_users ≈ target_RPS

SCENARIO_TIMELINE: List[ScenarioPoint] = [
    # elapsed, users, label
    (0,    10,  "1-STABLE_LOW-start"),
    (180,  10,  "1-STABLE_LOW-end"),
    (181,  60,  "2-STABLE_HIGH-start"),
    (360,  60,  "2-STABLE_HIGH-end"),
    (361,   5,  "3-GRADUAL_RAMP-start"),
    (661,  120, "3-GRADUAL_RAMP-end"),
    (662,   5,  "4-SUDDEN_SPIKE-idle"),
    (680,  200, "4-SUDDEN_SPIKE-burst"),
    (710,   5,  "4-SUDDEN_SPIKE-recovery"),
    (740,   5,  "5-JITTERY-start"),
    (1140,  5,  "5-JITTERY-end"),
]

# ──────────────────────────────────────────────────────────────
# Custom LoadTestShape
# ──────────────────────────────────────────────────────────────

class FaaSScenarioShape(LoadTestShape):
    """
    Returns (user_count, spawn_rate) at each tick.
    Between timeline checkpoints we interpolate linearly.
    During the JITTERY phase we inject random noise.
    """

    _jitter_seed = random.Random(42)

    def tick(self):
        elapsed = self.get_current_run_time()

        # Past final point → stop
        if elapsed > SCENARIO_TIMELINE[-1][0]:
            return None

        # Find surrounding checkpoints
        users = self._interpolate(elapsed)

        # Jittery phase: add ±40% random noise
        if 740 <= elapsed <= 1140:
            noise = self._jitter_seed.uniform(-0.4, 0.4)
            users = max(1, int(users * (1 + noise)))
            # Occasional spikes
            if self._jitter_seed.random() < 0.05:
                users = min(180, users + self._jitter_seed.randint(50, 120))

        return users, max(20, users // 3)   # (count, spawn_rate)

    def _interpolate(self, elapsed: float) -> int:
        """Linear interpolation between timeline points."""
        prev_t, prev_u = SCENARIO_TIMELINE[0][0], SCENARIO_TIMELINE[0][1]
        for t, u, _ in SCENARIO_TIMELINE[1:]:
            if elapsed <= t:
                if t == prev_t:
                    return u
                frac = (elapsed - prev_t) / (t - prev_t)
                return max(1, int(prev_u + frac * (u - prev_u)))
            prev_t, prev_u = t, u
        return SCENARIO_TIMELINE[-1][1]


# ──────────────────────────────────────────────────────────────
# User behaviour
# ──────────────────────────────────────────────────────────────

class FaaSUser(HttpUser):
    wait_time  = between(0.8, 1.2)
    abstract   = False

    def on_start(self):
        self.client.auth = (FAAS_USER, FAAS_PASS)

    @task(8)
    def invoke_echo(self):
        payload = {"ts": time.time(), "data": "hello-rl"}
        with self.client.post(
            f"/function/{FUNCTION}",
            json=payload,
            catch_response=True,
            name=f"/function/{FUNCTION}",
        ) as resp:
            if resp.status_code == 200:
                try:
                    body = resp.json()
                    warm = body.get("warm", False)
                    # Tag the sample so Grafana can split warm vs cold
                    resp.success()
                    events.request.fire(
                        request_type = "WARM" if warm else "COLD",
                        name         = FUNCTION,
                        response_time= resp.elapsed.total_seconds() * 1000,
                        response_length= len(resp.content),
                        exception    = None,
                        context      = {},
                    )
                except Exception:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(2)
    def health_check(self):
        """Keep a baseline of gateway health calls."""
        self.client.get(
            "/healthz",
            name="/healthz",
            catch_response=True,
        )


# ──────────────────────────────────────────────────────────────
# Event hooks — print scenario transitions
# ──────────────────────────────────────────────────────────────

_last_label = ""

@events.test_start.add_listener
def on_test_start(environment: Environment, **_kw):
    print("\n" + "=" * 60)
    print("FaaS RL Pre-warmer — Locust test starting")
    print(f"Function : {FUNCTION}")
    print("Scenarios: STABLE_LOW → STABLE_HIGH → GRADUAL_RAMP → SPIKE → JITTERY")
    print("=" * 60 + "\n")


@events.spawning_complete.add_listener
def on_spawn(user_count: int, **_kw):
    global _last_label
    elapsed = time.time()
    for t, u, label in SCENARIO_TIMELINE:
        if abs(u - user_count) < 5 and label != _last_label:
            print(f"\n>>> [{elapsed:.0f}s] Scenario → {label}  ({user_count} users) <<<\n")
            _last_label = label
            break


@events.test_stop.add_listener
def on_test_stop(**_kw):
    print("\n" + "=" * 60)
    print("Test complete. Check report.html for full results.")
    print("=" * 60 + "\n")
