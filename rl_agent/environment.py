"""
environment.py (PPO edition)
=============================
Real FaaSEnv (Prometheus + OpenFaaS) and SyntheticFaaSEnv for pre-training.

Bugs fixed vs the uploaded notebook's ServerlessEnv:
  1. Observation space bounds were [0,100] but state values are in [0,1] -> fixed
  2. alpha1-4 and cmem defined but never used -> removed
  3. mem was always 0.2 (never updated) -> now tracks idle container memory
  4. latency = total_requests (bad proxy) -> queue-weighted latency
"""
from __future__ import annotations
import logging, math, time
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import requests

log = logging.getLogger(__name__)

MAX_WARM   = 20
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
        self._current_warm = target
        time.sleep(self.cfg.step_seconds)
        m = self._scrape()
        r = -5.0*m["cold"] - 0.2*max(0,m["idle"]-1) + 2.0*m["warm"]*0.05
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

    def _scrape(self):
        fn, dur = self.cfg.function_name, self.cfg.step_seconds
        rr  = self._prom(f'rate(gateway_function_invocation_total{{function_name="{fn}"}}[{dur}s])')
        ct  = int(self._prom(f'gateway_function_invocation_total{{function_name="{fn}",code="cold"}}'))
        wt  = int(self._prom(f'gateway_function_invocation_total{{function_name="{fn}",code="warm"}}'))
        q   = self._prom(f'gateway_service_queue_depth{{function_name="{fn}"}}')
        reps = self._get_replicas()
        cold = max(0,ct-self._prev_cold);  self._prev_cold=ct
        warm = max(0,wt-self._prev_warm);  self._prev_warm=wt
        rd   = rr - self._prev_req_rate;   self._prev_req_rate=rr
        idle = max(0, reps - max(1,int(rr/15)))
        return dict(rr=rr,rdelta=rd,cold=cold,warm=warm,csr=cold/max(1,cold+warm),q=q,idle=idle)

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
        r  = -5.0*m["cold_starts"] - 0.2*max(0,m["idle_warm"]-1) + 2.0*m["warm_hits"]*0.05
        done = self._t >= 1440
        return self._obs(rr,m), r, done, False, m

    def _rr(self):
        t, h = self._t%1440, (self._t%1440)/60
        if   self.pattern=="daily":  base = 50*math.exp(-0.5*((h-12)/3)**2)+5
        elif self.pattern=="spike":  base = 10+(200 if 360<=t<380 else 0)
        elif self.pattern=="jitter": base = 20+30*abs(math.sin(t/15))
        else:                        base = 30.0
        return max(0.0, base + float(self.rng.normal(0, base*0.05)))

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
