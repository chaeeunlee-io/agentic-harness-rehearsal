#!/usr/bin/env python3
"""Save the current AppWorld state for the verifier.

Runs inside the sandbox with `APPWORLD_TASK_ID` available. Failures here do not
override the agent's result.
"""
import os
import sys

import requests

URL = os.environ.get("APPWORLD_ENV_URL", "http://localhost:9000")
task_id = os.environ.get("APPWORLD_TASK_ID")
if not task_id:
    sys.stderr.write("APPWORLD_TASK_ID not set in sandbox\n")
    sys.exit(0)
try:
    requests.post(URL + "/save_logs", json={"task_id": task_id}, timeout=180)
except Exception as exc:
    sys.stderr.write("save_logs failed: %s\n" % exc)
