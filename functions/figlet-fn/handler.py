import json
import time
import pyfiglet

# This only runs once when the container starts
START_TIME = time.time()

def handle(req):
    uptime = time.time() - START_TIME
    
    # Try to parse input, fallback to raw string
    try:
        data = json.loads(req)
        payload = data.get("data", str(req))
    except:
        payload = str(req)

    # Generate the ascii art for the "echo" field
    ascii_echo = pyfiglet.figlet_format(payload)

    # Returning the exact format you requested
    return {
        "statusCode": 200,
        "body": json.dumps({
            "echo":     ascii_echo,
            "uptime_s":  round(uptime, 2),
            "warm":      uptime > 2.0,
        }),
    }