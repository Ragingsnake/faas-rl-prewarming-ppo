"""
locustfile.py - Load Test for RL Pre-warmer
Targets the 'figlet' function to test scaling and cold starts.
"""

import os
import time
import random
from typing import List, Tuple
from locust import HttpUser, task, between, events
from locust.shape import LoadTestShape

# Grab config from env or use defaults from our setup script
FAAS_USER = os.getenv("OPENFAAS_USER", "admin")
FAAS_PASS = os.getenv("OPENFAAS_PASS", "admin") # You'll pass this in the run script
FUNCTION  = os.getenv("FAAS_FUNCTION", "figlet")

# (elapsed_seconds, target_users/rps)
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

    def tick(self):
        elapsed = self.get_current_run_time()
        if elapsed > SCENARIO_TIMELINE[-1][0]:
            return None

        # Linear interpolation between points
        users = self._interpolate(elapsed)

        # Add noise if we are in the JITTERY phase (last 400s)
        if elapsed >= 740:
            noise = self._jitter_seed.uniform(-0.4, 0.4)
            users = max(1, int(users * (1 + noise)))
            if self._jitter_seed.random() < 0.05: # Occasional random spikes
                users = min(180, users + self._jitter_seed.randint(50, 100))

        return (users, 20) # (user count, spawn rate)

    def _interpolate(self, elapsed):
        prev_t, prev_u = SCENARIO_TIMELINE[0]
        for t, u in SCENARIO_TIMELINE[1:]:
            if elapsed <= t:
                if t == prev_t: return u
                return int(prev_u + (elapsed - prev_t) / (t - prev_t) * (u - prev_u))
            prev_t, prev_u = t, u
        return SCENARIO_TIMELINE[-1][1]

class FaaSUser(HttpUser):
    # 1 user approx 1 RPS
    wait_time = between(0.8, 1.2)

    def on_start(self):
        self.client.auth = (FAAS_USER, FAAS_PASS)

    @task(8)
    def invoke_function(self):
        # figlet just takes a raw string, not JSON
        payload = "RL-Test-Tick-" + str(time.time())
        self.client.post(f"/function/{FUNCTION}", data=payload, name=f"/function/{FUNCTION}")

    @task(2)
    def health_check(self):
        self.client.get("/healthz", name="/healthz")

@events.test_start.add_listener
def on_test_start(environment, **_kw):
    print(f"Starting test on function: {FUNCTION}")

@events.test_stop.add_listener
def on_test_stop(environment, **_kw):
    print("Test finished. Check report.html")