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

MAX_WARM   = 5    # CE hard cap — keeps action space and state range realistic
MIN_WARM   = 1    # CE minimum is 1 — never evict last container
STEP_SLEEP = 15
ACTION_MAP = [-2, -1, 0, 1, 2]
PER_REPLICA_RPS = 15.0
SCALE_UP_UTIL = 0.90
SCALE_DOWN_UTIL = 0.80
MAX_STEP_CHANGE = 1
RR_NORM = 150.0
QUEUE_NORM = 150.0


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
        raw_delta = ACTION_MAP[action_idx]
        prev_reps = self._current_warm
        delta = self._stabilize_delta(raw_delta, self._prev_req_rate, prev_reps)
        requested = prev_reps + delta
        target = int(np.clip(requested, MIN_WARM, MAX_WARM))
        applied_delta = target - prev_reps
        self._set_replicas(target)
        # FIX 5: do NOT set _current_warm = target here.
        # CE may cap the replica count below what we requested.
        # We read the actual value back from the API after sleeping.
        time.sleep(self.cfg.step_seconds)
        m = self._scrape()
        # _current_warm is now the ground-truth count from OpenFaaS,
        # so state and /status always match `faas-cli list`.
        self._current_warm = m["reps"]

        # No action penalty when scaling down at zero traffic — encourages aggressive scale-down
        action_penalty = -0.1 if (delta != 0 and m["rr"] > 1.0) else 0.0
        
        # FIX 7: Penalize impossible actions (asking to scale beyond bounds)
        # Agent should learn that +1/+2 at MAX_WARM or -1/-2 at MIN_WARM is wasteful
        if requested > MAX_WARM or requested < MIN_WARM:
            action_penalty -= 0.3  # extra penalty for invalid action
        
        r = self._reward(m) + action_penalty

        return self._state(m["rr"], m["rdelta"], m["csr"], m["q"]), r, False, {
            "warm_containers": self._current_warm,
            "cold_starts": m["cold"], "warm_hits": m["warm"],
            "req_rate": m["rr"], "reward": r,
            "latency": m["latency"],
            "queue": m["q"],
            "raw_delta": raw_delta,
            "applied_delta": applied_delta,
        }

    def _stabilize_delta(self, delta: int, rr: float, reps: int) -> int:
        # Deterministic safety mode: when idle, always move toward 1 replica.
        if rr <= 5.0 and reps > MIN_WARM:
            return -1
        # Emergency safety mode: if heavily overloaded, force scale-up by 1.
        if rr > reps * PER_REPLICA_RPS * 1.2 and reps < MAX_WARM:
            return 1
        if delta == 0:
            return 0
        delta = int(np.clip(delta, -MAX_STEP_CHANGE, MAX_STEP_CHANGE))
        min_safe_reps = max(MIN_WARM, math.ceil(rr / PER_REPLICA_RPS))
        if delta < 0 and reps + delta < min_safe_reps:
            return 0
        if delta > 0:
            up_threshold = reps * PER_REPLICA_RPS * SCALE_UP_UTIL
            return delta if rr > up_threshold else 0
        down_threshold = max(1, reps - 1) * PER_REPLICA_RPS * SCALE_DOWN_UTIL
        return delta if rr < down_threshold else 0

    def _state(self, rr, rdelta, csr, q):
        now = datetime.utcnow()
        h, d = now.hour + now.minute/60, now.weekday()
        return np.array([
            np.clip(rr / RR_NORM, 0, 1), np.clip(rdelta / RR_NORM, -1, 1),
            math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24),
            math.sin(2*math.pi*d/7),  math.cos(2*math.pi*d/7),
            self._current_warm/MAX_WARM, csr, np.clip(q / QUEUE_NORM, 0, 1),
        ], dtype=np.float32)

    def _reward(self, m: dict) -> float:
        # REWRITE: reward is now rate-based (not count-based) so it's bounded
        # regardless of traffic volume. Before: range was [-276k, +3k] which
        # caused critic MSE loss in the millions → divergence → actor frozen.
        # Now: range is [-6, +1] per step, critic can actually learn.
        total_reqs = m["cold"] + m["warm"]
        if total_reqs > 0:
            cold_rate = m["cold"] / total_reqs        # [0, 1]
            # quality: -5 if all cold, +1 if all warm
            quality = -5.0 * cold_rate + 1.0 * (1.0 - cold_rate)
        else:
            quality = 0.0

        # Idle cost: stronger penalties at low traffic for faster scale-down.
        if m["rr"] < 1.0:
            idle_penalty = -1.2 * max(0, m["reps"] - 1)
        elif m["rr"] < 5.0:
            idle_penalty = -0.7 * max(0, m["reps"] - 1)
        else:
            needed = max(1, math.ceil(m["rr"] / PER_REPLICA_RPS))
            idle_penalty = -0.4 * max(0, m["reps"] - needed)

        queue_penalty = -0.02 * min(100.0, m["q"])
        return quality + idle_penalty + queue_penalty

    def _scrape(self):
        # 1. Prometheus requires the namespace suffix in OpenFaaS CE!
        prom_fn = f"{self.cfg.function_name}.openfaas-fn"
        dur = self.cfg.step_seconds
        
        # Query Prometheus using the fully qualified name (No fake queue metric!)
        rr = self._prom(f'rate(gateway_function_invocation_total{{function_name="{prom_fn}"}}[{dur}s])')
        latency = self._prom_first([
            # OpenFaaS gateway latency histogram (preferred)
            f'histogram_quantile(0.95, sum(rate(gateway_functions_seconds_bucket{{function_name="{prom_fn}"}}[{dur}s])) by (le))',
            f'histogram_quantile(0.95, sum(rate(gateway_functions_seconds_bucket{{function_name="{self.cfg.function_name}"}}[{dur}s])) by (le))',
            f'histogram_quantile(0.95, sum(rate(gateway_function_invocation_seconds_bucket{{function_name="{prom_fn}"}}[{dur}s])) by (le))',
            f'histogram_quantile(0.95, sum(rate(gateway_function_invocation_seconds_bucket{{function_name="{self.cfg.function_name}"}}[{dur}s])) by (le))',
            # Fallback to average latency when histogram buckets are unavailable
            f'sum(rate(gateway_functions_seconds_sum{{function_name="{prom_fn}"}}[{dur}s])) / clamp_min(sum(rate(gateway_functions_seconds_count{{function_name="{prom_fn}"}}[{dur}s])), 1e-9)',
            f'sum(rate(gateway_functions_seconds_sum{{function_name="{self.cfg.function_name}"}}[{dur}s])) / clamp_min(sum(rate(gateway_functions_seconds_count{{function_name="{self.cfg.function_name}"}}[{dur}s])), 1e-9)',
        ], default=0.0)
        
        # The OpenFaaS API still expects just the base name
        reps = self._get_replicas()

        # 2. Mathematical Cold/Warm/Queue Estimation
        # 1 replica handles ~PER_REPLICA_RPS smoothly. Overflow is penalized.
        capacity = max(1, reps) * PER_REPLICA_RPS
        
        if rr > capacity:
            cold_estimate = rr - capacity
            warm_estimate = capacity
            q_estimate = rr - capacity  # The overflow IS the backlog queue
        else:
            cold_estimate = 0.0
            warm_estimate = rr
            q_estimate = 0.0            # No overflow, no queue

        # 3. Calculate deltas
        rd = rr - self._prev_req_rate
        self._prev_req_rate = rr
        idle = max(0, reps - max(1, int(rr / PER_REPLICA_RPS)))

        return dict(rr=rr, rdelta=rd, cold=cold_estimate, warm=warm_estimate,
                    csr=cold_estimate/max(1, rr), latency=latency, q=q_estimate, idle=idle, reps=reps)

    # def _scrape(self):
    #     fn, dur = self.cfg.function_name, self.cfg.step_seconds
    #     rr  = self._prom(f'rate(gateway_function_invocation_total{{function_name="{fn}"}}[{dur}s])')
    #     ct  = int(self._prom(f'gateway_function_invocation_total{{function_name="{fn}",code="cold"}}'))
    #     wt  = int(self._prom(f'gateway_function_invocation_total{{function_name="{fn}",code="warm"}}'))
    #     q   = self._prom(f'gateway_service_queue_depth{{function_name="{fn}"}}')
    #     # FIX 5: reps is returned so step() can sync _current_warm to ground truth
    #     reps = self._get_replicas()
    #     cold = max(0,ct-self._prev_cold);  self._prev_cold=ct
    #     warm = max(0,wt-self._prev_warm);  self._prev_warm=wt
    #     rd   = rr - self._prev_req_rate;   self._prev_req_rate=rr
    #     idle = max(0, reps - max(1,int(rr/15)))
    #     return dict(rr=rr,rdelta=rd,cold=cold,warm=warm,
    #                 csr=cold/max(1,cold+warm),q=q,idle=idle,reps=reps)

    def _prom(self, query, default=0.0):
        try:
            r = requests.get(f"{self.cfg.prometheus_url}/api/v1/query",
                             params={"query":query}, timeout=5)
            d = r.json()["data"]["result"]
            return float(d[0]["value"][1]) if d else default
        except: return default

    def _prom_first(self, queries: list[str], default=0.0) -> float:
        for q in queries:
            v = self._prom(q, default=float("nan"))
            if not math.isnan(v):
                return v
        return default

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
        self._t, self._warm, self._prev_rr = 0, MIN_WARM, 0.0
        return self._obs(), {}

    def step(self, action_idx):
        raw_delta = ACTION_MAP[action_idx]
        prev_warm = self._warm
        delta = self._stabilize_delta(raw_delta, self._prev_rr, prev_warm)
        requested = prev_warm + delta
        self._warm = int(np.clip(requested, MIN_WARM, MAX_WARM))
        applied_delta = self._warm - prev_warm
        self._t   += 1
        rr = self._rr()
        m  = self._sim(rr)
        # Rate-based reward: identical formula to FaaSEnv._reward()
        # so the synthetic agent learns the same objective as the online agent.
        total_reqs = m["cold_starts"] + m["warm_hits"]
        if total_reqs > 0:
            cold_rate = m["cold_starts"] / total_reqs
            quality   = -5.0 * cold_rate + 1.0 * (1.0 - cold_rate)
        else:
            quality = 0.0
        if rr < 1.0:
            idle_penalty = -1.2 * max(0, self._warm - 1)
        elif rr < 5.0:
            idle_penalty = -0.7 * max(0, self._warm - 1)
        else:
            needed = max(1, math.ceil(rr / PER_REPLICA_RPS))
            idle_penalty = -0.4 * max(0, self._warm - needed)

        # No action penalty when scaling down at zero traffic — encourages aggressive scale-down
        action_penalty = -0.1 if (delta != 0 and rr > 1.0) else 0.0
        
        # FIX 7: Penalize impossible actions (asking to scale beyond bounds)
        if requested > MAX_WARM or requested < MIN_WARM:
            action_penalty -= 0.3
        
        queue_penalty = -0.02 * min(100.0, float(m["queue"]))
        r = quality + idle_penalty + queue_penalty + action_penalty
        
        done = self._t >= 1440
        m["raw_delta"] = raw_delta
        m["applied_delta"] = applied_delta
        return self._obs(rr,m), r, done, False, m

    def _stabilize_delta(self, delta: int, rr: float, reps: int) -> int:
        if rr <= 5.0 and reps > MIN_WARM:
            return -1
        if rr > reps * PER_REPLICA_RPS * 1.2 and reps < MAX_WARM:
            return 1
        if delta == 0:
            return 0
        delta = int(np.clip(delta, -MAX_STEP_CHANGE, MAX_STEP_CHANGE))
        min_safe_reps = max(MIN_WARM, math.ceil(rr / PER_REPLICA_RPS))
        if delta < 0 and reps + delta < min_safe_reps:
            return 0
        if delta > 0:
            up_threshold = reps * PER_REPLICA_RPS * SCALE_UP_UTIL
            return delta if rr > up_threshold else 0
        down_threshold = max(1, reps - 1) * PER_REPLICA_RPS * SCALE_DOWN_UTIL
        return delta if rr < down_threshold else 0

    def _rr(self):
        t, h = self._t%1440, (self._t%1440)/60
        if   self.pattern=="daily":  base = 50*math.exp(-0.5*((h-12)/3)**2)+5
        elif self.pattern=="spike":  base = 10+(200 if 360<=t<380 else 0)
        elif self.pattern=="jitter": base = 20+30*abs(math.sin(t/15))
        # FIX 6: "zero" pattern — truly zero traffic so agent learns exact
        # online state (rr=0). uniform(0,2) was giving quality=+1 at 1 RPS
        # with any containers → no incentive to scale down.
        elif self.pattern=="zero":   base = 0.0
        else:                        base = 30.0
        noise_scale = base * 0.05 if base > 0 else 0.0
        return max(0.0, base + float(self.rng.normal(0, noise_scale)))

    def _sim(self, rr):
        surp = self._warm * PER_REPLICA_RPS - rr
        cold_p = 1/(1+math.exp(surp/5))
        n    = max(0, int(rr))
        cold = int(self.rng.binomial(n, cold_p))
        warm = n - cold
        idle = max(0, self._warm - max(1, int(rr / PER_REPLICA_RPS)))
        capacity = max(1.0, self._warm * PER_REPLICA_RPS)
        queue = max(0.0, rr - capacity)
        # Approximate queueing delay in seconds from overload ratio.
        latency = queue / capacity if queue > 0.0 else 0.0
        return dict(cold_starts=cold, warm_hits=warm, idle_warm=idle,
                    cold_start_ratio=cold/max(1,n), containers=self._warm,
                    latency=latency, queue=queue,
                    # FIX: mem now reflects idle containers (was static 0.2)
                    mem=min(1.0, idle*self.BASE_MEM))

    def _obs(self, rr=None, m=None):
        rr = rr if rr is not None else self._rr()
        m  = m  if m  is not None else self._sim(rr)
        h  = (self._t%1440)/60;  d = (self._t//1440)%7
        delta, self._prev_rr = np.clip((rr - self._prev_rr) / RR_NORM, -1, 1), rr
        return np.array([
            np.clip(rr / RR_NORM, 0, 1), delta,
            math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24),
            math.sin(2*math.pi*d/7),  math.cos(2*math.pi*d/7),
            self._warm/MAX_WARM, m["cold_start_ratio"], np.clip(m["queue"] / QUEUE_NORM, 0, 1),
        ], dtype=np.float32)
