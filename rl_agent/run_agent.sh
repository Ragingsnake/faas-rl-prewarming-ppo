#!/bin/bash
# run_agent.sh - sets up the python env, pretrains, and runs the ppo agent
# Run from inside rl_agent/: bash run_agent.sh [active|inactive]

set -e  # stop on first error so you see exactly what failed

AGENT_MODE=${1:-active}
if [ "$AGENT_MODE" != "active" ] && [ "$AGENT_MODE" != "inactive" ]; then
    echo "Usage: bash run_agent.sh [active|inactive]"
    exit 1
fi

# ── locate the gateway password from minikube ──────────────────
OPENFAAS_PASS=$(kubectl get secret -n openfaas basic-auth \
    -o jsonpath="{.data.basic-auth-password}" | base64 --decode)

# ── point everything at local port-forwards ────────────────────
export PROMETHEUS_URL=http://127.0.0.1:9090
export OPENFAAS_URL=http://127.0.0.1:8080
export OPENFAAS_USER=admin
export OPENFAAS_PASS=$OPENFAAS_PASS
export FAAS_FUNCTION=figlet-fn        # must match the name in stack.yml
export STEP_SECONDS=15
export AGENT_START_MODE=$AGENT_MODE

# ── checkpoint paths — both relative to rl_agent/ ─────────────
# FIX: pretrain.py saves here, server.py must look here (not /checkpoints/)
export PRETRAIN_CKPT=checkpoints/pretrained.pt
export CHECKPOINT_PATH=checkpoints/online_ppo.pt

echo "setting up python venv..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "installing agent requirements..."
pip install -r requirements.txt
# Needed by /save-metrics chart generation endpoint
pip install matplotlib==3.8.4

echo "running pretraining..."
mkdir -p checkpoints
if [ -f "$PRETRAIN_CKPT" ]; then
    echo "pretrained checkpoint '$PRETRAIN_CKPT' already exists — skipping pretraining."
else
    echo "pretrained checkpoint not found — starting pretraining..."
    python3 pretrain.py --epochs 100 --out "$PRETRAIN_CKPT"
fi

# FIX: was python3 main.py — main.py doesn't exist, entrypoint is server.py
echo "launching the ppo agent server on :9000 (mode=$AGENT_START_MODE)..."
python3 server.py
