import json
import platform

print(json.dumps({
    "status": "ok",
    "message": "Harness subprocess execution works",
    "python": platform.python_version()
}, indent=2))
