#!/bin/bash
# run_loadtest.sh - fires up locust with the right credentials
# Run from inside locust/: bash run_loadtest.sh

set -e

echo "setting up venv..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "making sure locust is here..."
pip install locust --quiet

# grab the password from k8s so you don't have to hunt for it
export OPENFAAS_USER=admin
export OPENFAAS_PASS=$(kubectl get secret -n openfaas basic-auth \
    -o jsonpath="{.data.basic-auth-password}" | base64 --decode)
export FAAS_FUNCTION=echo-fn    # must match stack.yml function name

echo "OPENFAAS_USER: $OPENFAAS_USER"
echo "FAAS_FUNCTION: $FAAS_FUNCTION"
echo "Gateway: http://127.0.0.1:8080"
echo ""

echo "starting headless load test for 19 minutes..."
# FIX: --headless requires --users and --spawn-rate
# These are the max caps; the LoadTestShape in locustfile-local.py
# controls the actual ramp profile within these limits
locust -f locustfile-local.py \
    --host http://127.0.0.1:8080 \
    --headless \
    --users 200 \
    --spawn-rate 20 \
    --run-time 19m \
    --html report.html \
    --exit-code-on-error 0

echo ""
echo "done. report saved to locust/report.html"
