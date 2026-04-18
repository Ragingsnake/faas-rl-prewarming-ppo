"""
locustfile.py - Case-based Load Tests for RL Pre-warmer.
Set LOAD_CASE env to one of: stable_low, stable_high, gradual_ramp, sudden_spike
"""

import os
import time
from typing import List, Tuple
from locust import HttpUser, task, between, events
from locust.shape import LoadTestShape

FAAS_USER = os.getenv("OPENFAAS_USER", "admin")
FAAS_PASS = os.getenv("OPENFAAS_PASS", "admin")
FUNCTION  = os.getenv("FAAS_FUNCTION", "echo-fn")

LOAD_CASE = os.getenv("LOAD_CASE", "stable_low").strip().lower()
CASE_TIMELINES: dict[str, List[Tuple[int, int]]] = {
    # (elapsed_seconds, target_users)
    "stable_low": [(0, 10), (240, 10)],
    "stable_high": [(0, 60), (240, 60)],
    "gradual_ramp": [(0, 5), (240, 120)],
    "sudden_spike": [(0, 5), (90, 5), (110, 180), (150, 180), (170, 5), (240, 5)],
}

if LOAD_CASE not in CASE_TIMELINES:
    raise ValueError(f"Unknown LOAD_CASE={LOAD_CASE}. Choose one of: {', '.join(CASE_TIMELINES)}")
SCENARIO_TIMELINE = CASE_TIMELINES[LOAD_CASE]


class FaaSScenarioShape(LoadTestShape):
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
    print(f"Starting case '{LOAD_CASE}' → function: {FUNCTION}")


@events.test_stop.add_listener
def on_test_stop(environment, **_kw):
    print("Test finished. Check report.html")
