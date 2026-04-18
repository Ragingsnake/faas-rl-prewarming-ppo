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
export FAAS_FUNCTION=figlet-fn    # must match stack.yml function name
export LOAD_CASE=${LOAD_CASE:-stable_low}
export OPENFAAS_URL=${OPENFAAS_URL:-http://127.0.0.1:8080}

echo "OPENFAAS_USER: $OPENFAAS_USER"
echo "FAAS_FUNCTION: $FAAS_FUNCTION"
echo "Gateway: $OPENFAAS_URL"
echo ""

reset_replicas_to_one() {
    echo "resetting replicas to 1 for '$FAAS_FUNCTION'..."
    curl -sS -u "$OPENFAAS_USER:$OPENFAAS_PASS" \
      -H "Content-Type: application/json" \
      -X POST "$OPENFAAS_URL/system/scale-function/$FAAS_FUNCTION" \
      -d "{\"service\":\"$FAAS_FUNCTION\",\"replicas\":1}" > /dev/null

    for _ in $(seq 1 30); do
        reps=$(curl -sS -u "$OPENFAAS_USER:$OPENFAAS_PASS" \
          "$OPENFAAS_URL/system/function/$FAAS_FUNCTION" | \
          python3 -c "import json,sys; print(int(json.load(sys.stdin).get('availableReplicas', -1)))" 2>/dev/null || echo -1)
        if [ "$reps" = "1" ]; then
            echo "replicas confirmed at 1"
            return 0
        fi
        sleep 1
    done

    echo "failed to confirm replicas=1 before test start"
    return 1
}

echo "LOAD_CASE: $LOAD_CASE"
reset_replicas_to_one
echo "starting headless load test..."
# FIX: --headless requires --users and --spawn-rate
# These are the max caps; the LoadTestShape in locustfile-local.py
# controls the actual ramp profile within these limits
locust -f locustfile.py \
    --host "$OPENFAAS_URL" \
    --headless \
    --users 200 \
    --spawn-rate 20 \
    --run-time 250s \
    --html "report_${LOAD_CASE}.html" \
    --exit-code-on-error 0

echo ""
echo "done. report saved to locust/report_${LOAD_CASE}.html"
