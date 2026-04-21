# FaaS RL Prewarmer (OpenFaaS CE)

This project trains a PPO agent to pre-warm OpenFaaS function replicas before cold starts hurt latency.

It is built for:
- **OpenFaaS CE limits** (replicas clamped to 1..5).
- **Prometheus-first signals** (request rate, estimated queue, and latency when available).
- **Repeatable load tests** with Locust case matrix runs.

## What this does.

The agent runs a control loop every `STEP_SECONDS`:
1. Read metrics from Prometheus and OpenFaaS.
2. Choose an action from `[-2, -1, 0, +1, +2]`.
3. Apply safe scaling logic (clamp/gating for CE reality).
4. Log metrics.
5. Learn online (if learning is enabled).

## Real behavior vs estimated behavior.

Some metrics are real, some are model-based:

- **Containers**: real (`availableReplicas` from OpenFaaS API).
- **Request rate (RPS)**: real (Prometheus query).
- **Latency**: real when latency series exist in Prometheus; otherwise 0 and chart fallback is used.
- **Warm/Cold chart**: currently a **capacity-based estimate**, not native OpenFaaS cold/warm labels.
  - `warm_hits = min(rr, capacity)`.
  - `cold_starts = max(rr - capacity, 0)`.
  - where `capacity = replicas * PER_REPLICA_RPS`.

So the warm/cold panel is best read as "within estimated warm capacity vs overflow pressure."

## Modes.

The RL server supports:
- **active**: policy acts every step.
- **inactive**: agent holds action (useful for baseline).

Learning can also be toggled at runtime:
- `POST /set-learning?enabled=true|false`.

This is used in matrix runs so case comparisons are fair and not polluted by online drift.

## Chart panels (current meaning).

The generated chart has 4 panels:
1. **Top-left**: latency in seconds (or RPS fallback if latency metric is missing).
2. **Top-right**: estimated warm vs cold requests (capacity model).
3. **Bottom-left**: estimated queue size (overflow beyond capacity).
4. **Bottom-right**: container count (actual replicas).

## Quick start.

### 1. Start the RL server.

From `rl_agent/`:

```bash
bash run_agent.sh active
```

Use `inactive` if you want baseline mode at startup:

```bash
bash run_agent.sh inactive
```

### 2. Run one load case.

From `locust/`:

```bash
LOAD_CASE=stable_low bash run_loadtest.sh
```

Available `LOAD_CASE` values:
- `stable_low`
- `stable_high`
- `gradual_ramp`
- `sudden_spike`

### 3. Run full case matrix (active/inactive).

From `locust/`:

```bash
bash run_case_matrix.sh
```

This script:
- resets in-memory metrics.
- runs all four cases in inactive and active modes.
- disables online learning during comparison runs.
- saves per-case ranged charts and metadata CSV.

## API summary.

- `GET /health`: liveness.
- `GET /status`: JSON snapshot of server state.
- `GET /metrics`: Prometheus text metrics.
- `POST /set-mode?mode=active|inactive`: switch policy mode.
- `POST /set-learning?enabled=true|false`: enable or disable online learning.
- `POST /reset-metrics`: clear in-memory chart data.
- `POST /save-metrics`: save metrics JSON and render chart.
- `GET /download-chart?name=...`: download PNG from checkpoints.

## Latency metric troubleshooting.

If latency is missing, usually Prometheus does not expose the exact series name/label your query expects.

Check:
1. metric family name differs by OpenFaaS version.
2. `function_name` label format differs (`figlet-fn` vs `figlet-fn.openfaas-fn`).
3. gateway scrape target is missing.
4. histogram buckets are not exported or were dropped.
5. lookback window is too short for sparse traffic.

## Current practical notes.

- This repo targets OpenFaaS CE scaling reality, not unlimited autoscaling.
- Spike profiles above CE capacity will still show degraded quality.
- For fair policy comparisons, keep case inputs identical and learning disabled during benchmark sweeps.
