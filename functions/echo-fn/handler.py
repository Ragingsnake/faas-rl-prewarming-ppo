"""
echo-fn — minimal OpenFaaS function used as the pre-warming target.
Simulates realistic latency so the RL agent's cold/warm distinction
is meaningful in Locust tests.
"""

import json
import time
import os

_BOOT_TIME = time.time()      # approximate "warm" marker

def handle(event, context):
    # Simulate variable processing time
    work_ms = int(os.getenv("WORK_MS", "10"))
    time.sleep(work_ms / 1000)

    body = event.body if event.body else b"{}"
    try:
        payload = json.loads(body)
    except Exception:
        payload = {"raw": body.decode("utf-8", errors="replace")}

    uptime = time.time() - _BOOT_TIME
    return {
        "statusCode": 200,
        "body": json.dumps({
            "echo":      payload,
            "uptime_s":  round(uptime, 2),
            "warm":      uptime > 2.0,   # heuristic
        }),
    }
