#!/bin/bash
# run_agent.sh - sets up the python env, pretrains, and runs the ppo agent

echo "setting up python venv..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "installing agent requirements..."
pip install -r requirements.txt

echo "running pretraining..."
# swap pretrain.py with whatever your actual pretrain script is called
python3 pretrain.py 

echo "launching the main agent..."
# swap main.py with your actual run script
python3 main.py