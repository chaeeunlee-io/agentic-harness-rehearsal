#!/usr/bin/env python3
"""Initialize the AppWorld environment server for the current task.

Runs inside the sandbox, where `APPWORLD_TASK_ID` is available from the task
environment. Waits for the server to become healthy, then calls `/initialize`.
"""
import os
import sys
import time

import requests

URL = os.environ.get("APPWORLD_ENV_URL", "http://localhost:9000")

for _ in range(90):
    try:
        if requests.get(URL + "/", timeout=2).ok:
            break
    except Exception:
        pass
    time.sleep(1)
else:
    sys.stderr.write("env-server did not come up\n")
    try:
        sys.stderr.write(open("/tmp/aw_env_server.log").read())
    except Exception:
        pass
    sys.exit(1)

task_id = os.environ.get("APPWORLD_TASK_ID")
if not task_id:
    sys.stderr.write("APPWORLD_TASK_ID not set in sandbox\n")
    sys.exit(2)
experiment = os.environ.get("APPWORLD_EXPERIMENT_NAME", "harbor_trial")
resp = requests.post(
    URL + "/initialize",
    json={"task_id": task_id, "experiment_name": experiment},
    timeout=180,
)
if not resp.ok:
    sys.stderr.write("initialize failed: " + resp.text + "\n")
    sys.exit(1)
print("env-server ready + initialized:", task_id)
