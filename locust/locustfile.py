"""
locustfile.py - Load Test for RL Pre-warmer
Targets the figlet function across 5 load scenarios.
Algorithm-agnostic: the agent reads Prometheus, Locust doesn't talk to it.
"""

import os
import time
import random
from typing import List, Tuple
from locust import HttpUser, task, between, events
from locust.shape import LoadTestShape

FAAS_USER = os.getenv("OPENFAAS_USER", "admin")
FAAS_PASS = os.getenv("OPENFAAS_PASS", "admin")
FUNCTION  = os.getenv("FAAS_FUNCTION", "echo-fn")

# (elapsed_seconds, target_users)  ~1 user ≈ 1 RPS with wait_time between(0.8, 1.2)
SCENARIO_TIMELINE: List[Tuple[int, int]] = [
    (0,    10),   # 1. STABLE_LOW
    (180,  10),
    (181,  60),   # 2. STABLE_HIGH
    (360,  60),
    (361,  5),    # 3. GRADUAL_RAMP
    (661,  120),
    (662,  5),    # 4. SUDDEN_SPIKE
    (680,  200),
    (710,  5),
    (740,  5),    # 5. JITTERY
    (1140, 5),
]


class FaaSScenarioShape(LoadTestShape):
    _jitter_seed = random.Random(42)

    # FIX: get_current_run_time() was removed in Locust 2.x
    # Track start time manually on first tick instead
    _start_time: float = 0.0

    def tick(self):
        if not self._start_time:
            self._start_time = time.time()
        elapsed = time.time() - self._start_time

        if elapsed > SCENARIO_TIMELINE[-1][0]:
            return None

        users = self._interpolate(elapsed)

        # Jittery phase: random noise + occasional spike
        if elapsed >= 740:
            noise = self._jitter_seed.uniform(-0.4, 0.4)
            users = max(1, int(users * (1 + noise)))
            if self._jitter_seed.random() < 0.05:
                users = min(180, users + self._jitter_seed.randint(50, 100))

        return (users, 20)

    def _interpolate(self, elapsed):
        prev_t, prev_u = SCENARIO_TIMELINE[0]
        for t, u in SCENARIO_TIMELINE[1:]:
            if elapsed <= t:
                if t == prev_t:
                    return u
                return int(prev_u + (elapsed - prev_t) / (t - prev_t) * (u - prev_u))
            prev_t, prev_u = t, u
        return SCENARIO_TIMELINE[-1][1]


class FaaSUser(HttpUser):
    wait_time = between(0.8, 1.2)

    def on_start(self):
        self.client.auth = (FAAS_USER, FAAS_PASS)

    @task(8)
    def invoke_function(self):
        payload = "RL-Test-Tick-" + str(time.time())
        self.client.post(
            f"/function/{FUNCTION}",
            data=payload,
            name=f"/function/{FUNCTION}",
        )

    @task(2)
    def health_check(self):
        self.client.get("/healthz", name="/healthz")


@events.test_start.add_listener
def on_test_start(environment, **_kw):
    print(f"Starting test → function: {FUNCTION}")


@events.test_stop.add_listener
def on_test_stop(environment, **_kw):
    print("Test finished. Check report.html")