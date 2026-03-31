#!/bin/bash
# run_loadtest.sh - fires up locust with the right credentials

echo "setting up venv..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo "making sure locust is here..."
pip install locust

# grab the password from k8s so you don't have to hunt for it
export OPENFAAS_PASS=$(kubectl get secret -n openfaas basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode)

echo "starting headless load test for 19 minutes..."
locust -f locustfile.py \
    --host http://127.0.0.1:8080 \
    --headless \
    --run-time 19m \
    --html report.html