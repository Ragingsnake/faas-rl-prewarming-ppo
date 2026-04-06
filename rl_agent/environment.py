"""
environment.py (PPO edition)
=============================
Real FaaSEnv (Prometheus + OpenFaaS) and SyntheticFaaSEnv for pre-training.

Bugs fixed vs the uploaded notebook's ServerlessEnv:
  1. Observation space bounds were [0,100] but state values are in [0,1] -> fixed
  2. alpha1-4 and cmem defined but never used -> removed
  3. mem was always 0.2 (never updated) -> now tracks idle container memory
  4. latency = total_requests (bad proxy) -> queue-weighted latency

Bugs fixed in this revision:
  5. _current_warm was set to requested replicas, not actual replicas from
     OpenFaaS CE. CE caps at 5; if agent asked for 7, _current_warm=7 but
     reality was 5. State vector was wrong, /status lied, agent kept pushing
     +2 thinking commands weren't landing. Fixed: read actual replicas back
     after every sleep and update _current_warm from that truth.
  6. SyntheticFaaSEnv had no zero-traffic pattern. Agent never learned
     "scale down when idle". Added "zero" pattern (rr ~ 0-2 RPS).
"""
from __future__ import annotations
import logging, math, time
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import requests

log = logging.getLogger(__name__)

MAX_WARM   = 5
MIN_WARM   = 0
STEP_SLEEP = 15
ACTION_MAP = [-2, -1, 0, 1, 2]


@dataclass
class EnvConfig:
    prometheus_url: str = "http://prometheus:9090"
    openfaas_url:   str = "http://gateway.openfaas:8080"
    openfaas_user:  str = "admin"
    openfaas_pass:  str = "admin"
    function_name:  str = "echo-fn"
    step_seconds:   int = STEP_SLEEP


class FaaSEnv:
    def __init__(self, cfg: EnvConfig = EnvConfig()):
        self.cfg = cfg
        self._prev_req_rate = self._prev_cold = self._prev_warm = 0
        self._current_warm  = 1

    def reset(self):
        self._prev_req_rate = self._prev_cold = self._prev_warm = 0
        self._current_warm  = self._get_replicas()
        return self._state(0, 0, 0, 0)

    def step(self, action_idx: int):
        delta = ACTION_MAP[action_idx]
        target = int(np.clip(self._current_warm + delta, MIN_WARM, MAX_WARM))
        self._set_replicas(target)
        # FIX 5: do NOT set _current_warm = target here.
        # CE may cap the replica count below what we requested.
        # We read the actual value back from the API after sleeping.
        time.sleep(self.cfg.step_seconds)
        m = self._scrape()
        # _current_warm is now the ground-truth count from OpenFaaS,
        # so state and /status always match `faas-cli list`.
        self._current_warm = m["reps"]
        r = self._reward(m)
        return self._state(m["rr"], m["rdelta"], m["csr"], m["q"]), r, False, {
            "warm_containers": self._current_warm,
            "cold_starts": m["cold"], "warm_hits": m["warm"],
            "req_rate": m["rr"], "reward": r,
        }

    def _state(self, rr, rdelta, csr, q):
        now = datetime.utcnow()
        h, d = now.hour + now.minute/60, now.weekday()
        return np.array([
            np.clip(rr/500,0,1), np.clip(rdelta/100,-1,1),
            math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24),
            math.sin(2*math.pi*d/7),  math.cos(2*math.pi*d/7),
            self._current_warm/MAX_WARM, csr, np.clip(q/50,0,1),
        ], dtype=np.float32)

    def _reward(self, m: dict) -> float:
        cold_penalty = -5.0 * m["cold"]
        # FIX 5b: idle penalty scales with actual idle count, not capped at 1.
        # Also a hard zero-traffic penalty: if rr ~ 0 and we have >1 container,
        # charge proportionally so the agent learns to scale down when idle.
        idle = m["idle"]
        if m["rr"] < 1.0:
            # zero-traffic: full idle penalty per container above minimum
            idle_penalty = -0.5 * max(0, m["reps"] - 1)
        else:
            idle_penalty = -0.2 * max(0, idle - 1)
        warm_bonus = 2.0 * m["warm"] * 0.05
        return cold_penalty + idle_penalty + warm_bonus

    def _scrape(self):
        fn, dur = self.cfg.function_name, self.cfg.step_seconds
        rr  = self._prom(f'rate(gateway_function_invocation_total{{function_name="{fn}"}}[{dur}s])')
        ct  = int(self._prom(f'gateway_function_invocation_total{{function_name="{fn}",code="cold"}}'))
        wt  = int(self._prom(f'gateway_function_invocation_total{{function_name="{fn}",code="warm"}}'))
        q   = self._prom(f'gateway_service_queue_depth{{function_name="{fn}"}}')
        # FIX 5: reps is returned so step() can sync _current_warm to ground truth
        reps = self._get_replicas()
        cold = max(0,ct-self._prev_cold);  self._prev_cold=ct
        warm = max(0,wt-self._prev_warm);  self._prev_warm=wt
        rd   = rr - self._prev_req_rate;   self._prev_req_rate=rr
        idle = max(0, reps - max(1,int(rr/15)))
        return dict(rr=rr,rdelta=rd,cold=cold,warm=warm,
                    csr=cold/max(1,cold+warm),q=q,idle=idle,reps=reps)

    def _prom(self, query, default=0.0):
        try:
            r = requests.get(f"{self.cfg.prometheus_url}/api/v1/query",
                             params={"query":query}, timeout=5)
            d = r.json()["data"]["result"]
            return float(d[0]["value"][1]) if d else default
        except: return default

    def _get_replicas(self):
        try:
            r = requests.get(f"{self.cfg.openfaas_url}/system/function/{self.cfg.function_name}",
                             auth=(self.cfg.openfaas_user, self.cfg.openfaas_pass), timeout=5)
            return int(r.json().get("availableReplicas", 1))
        except: return self._current_warm

    def _set_replicas(self, n):
        try:
            requests.post(f"{self.cfg.openfaas_url}/system/scale-function/{self.cfg.function_name}",
                          json={"service":self.cfg.function_name,"replicas":n},
                          auth=(self.cfg.openfaas_user, self.cfg.openfaas_pass), timeout=5)
        except Exception as e: log.warning("Scale failed: %s", e)


class SyntheticFaaSEnv:
    """Simulates FaaS workload for offline pre-training. Gymnasium-compatible."""

    BASE_MEM = 0.02

    def __init__(self, pattern="daily", seed=42):
        self.rng, self.pattern = np.random.default_rng(seed), pattern
        self._t = self._warm = 0
        self._prev_rr = 0.0

    def reset(self, seed=None, options=None):
        self._t, self._warm, self._prev_rr = 0, 2, 0.0
        return self._obs(), {}

    def step(self, action_idx):
        self._warm = int(np.clip(self._warm + ACTION_MAP[action_idx], MIN_WARM, MAX_WARM))
        self._t   += 1
        rr = self._rr()
        m  = self._sim(rr)
        # FIX 6: match the real env reward — stronger idle penalty at zero traffic
        if rr < 1.0:
            idle_penalty = -0.5 * max(0, self._warm - 1)
        else:
            idle_penalty = -0.2 * max(0, m["idle_warm"] - 1)
        r = -5.0*m["cold_starts"] + idle_penalty + 2.0*m["warm_hits"]*0.05
        done = self._t >= 1440
        return self._obs(rr,m), r, done, False, m

    def _rr(self):
        t, h = self._t%1440, (self._t%1440)/60
        if   self.pattern=="daily":  base = 50*math.exp(-0.5*((h-12)/3)**2)+5
        elif self.pattern=="spike":  base = 10+(200 if 360<=t<380 else 0)
        elif self.pattern=="jitter": base = 20+30*abs(math.sin(t/15))
        # FIX 6: "zero" pattern — near-zero traffic, teaches agent to scale down
        elif self.pattern=="zero":   base = float(self.rng.uniform(0, 2))
        else:                        base = 30.0
        noise_scale = base*0.05 if base > 1 else 0.1
        return max(0.0, base + float(self.rng.normal(0, noise_scale)))

    def _sim(self, rr):
        surp = self._warm*15 - rr
        cold_p = 1/(1+math.exp(surp/5))
        n    = max(0, int(rr))
        cold = int(self.rng.binomial(n, cold_p))
        warm = n - cold
        idle = max(0, self._warm - max(1, int(rr/15)))
        return dict(cold_starts=cold, warm_hits=warm, idle_warm=idle,
                    cold_start_ratio=cold/max(1,n), containers=self._warm,
                    # FIX: latency = queue backlog, not raw total_requests
                    latency=max(0,n-self._warm), queue=max(0,n-self._warm),
                    # FIX: mem now reflects idle containers (was static 0.2)
                    mem=min(1.0, idle*self.BASE_MEM))

    def _obs(self, rr=None, m=None):
        rr = rr if rr is not None else self._rr()
        m  = m  if m  is not None else self._sim(rr)
        h  = (self._t%1440)/60;  d = (self._t//1440)%7
        # FIX: state already normalized — no /100 needed here
        delta, self._prev_rr = np.clip((rr-self._prev_rr)/100,-1,1), rr
        return np.array([
            np.clip(rr/500,0,1), delta,
            math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24),
            math.sin(2*math.pi*d/7),  math.cos(2*math.pi*d/7),
            self._warm/MAX_WARM, m["cold_start_ratio"], np.clip(m["queue"]/50,0,1),
        ], dtype=np.float32)
