"""Nightly restart of the local ComfyUI (systemd timer `comfyui-restart.timer`): a long-running ComfyUI grows until the OOM killer
takes it mid-job, so it gets a fresh process every night -- but only when it is idle. Busy = a studio job queued/running or
anything in ComfyUI's own queue (CLI runs); then it checks again every 5 minutes for up to 3 hours and otherwise skips the night.

    webui/.venv/bin/python restart_comfyui.py [--now]   (--now: one idle check, no waiting)
"""
import json
import subprocess
import sys
import time
import urllib.request

import config

WAIT_S, TRIES = 300, 36


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def busy() -> str:
    """Why a restart would interrupt work now, or "" when idle. An unreachable studio counts as idle (only ComfyUI's queue then)."""
    try:
        active = [j["id"] for j in _get(f"{config.STUDIO_URL}/api/jobs") if j["status"] in ("queued", "running")]
        if active:
            return f"studio jobs {active}"
    except OSError:
        pass
    try:
        q = _get(f"{config.LOCAL_COMFY_URL}/queue")
    except OSError:
        return ""  # ComfyUI down: a restart brings it back
    n = len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
    return f"{n} prompt(s) in the ComfyUI queue" if n else ""


def main():
    tries = 1 if "--now" in sys.argv else TRIES
    for i in range(tries):
        why = busy()
        if not why:
            subprocess.run(["systemctl", "--user", "restart", config.COMFYUI_SERVICE], check=True)
            print(f"restarted {config.COMFYUI_SERVICE}", flush=True)
            return
        print(f"busy ({why}), check {i + 1}/{tries}", flush=True)
        if i + 1 < tries:
            time.sleep(WAIT_S)
    print("still busy: skipped tonight's restart", flush=True)


if __name__ == "__main__":
    main()
