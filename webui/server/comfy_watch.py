"""Watches the local ComfyUI systemd user service and pushes (Pushover) when it goes down, comes back or restarts on its own
(crash, systemd-oomd kill). Restarts from the web UI's restart button are expected and stay quiet.
"""
import subprocess
import threading
import time

import config
import jobs

POLL_SECONDS = 10
QUIET_SECONDS = 120  # after a restart from the web UI
_quiet_until = 0.0


def expect_restart():
    """The web UI is about to restart ComfyUI itself: don't push about it."""
    global _quiet_until
    _quiet_until = time.time() + QUIET_SECONDS


def state():
    """(ActiveState, MainPID, Result) of the service, or None when systemctl can't tell."""
    try:
        out = subprocess.run(["systemctl", "--user", "show", config.COMFYUI_SERVICE, "-p", "ActiveState,MainPID,Result"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pairs = [line.split("=", 1) for line in out.splitlines() if "=" in line]
    p = dict(pairs)
    return (p["ActiveState"], p.get("MainPID", "0"), p.get("Result", "")) if "ActiveState" in p else None


def change(prev, cur):
    """The message for a transition between two state() readings, or None."""
    if not prev or not cur or prev == cur:
        return None
    (was, old_pid, _), (now, pid, result) = prev, cur
    why = f" ({result})" if result and result != "success" else ""
    if was == "active" and now != "active":
        return f"Local ComfyUI is down: {now}{why}"
    if was != "active" and now == "active":
        return "Local ComfyUI is back up"
    if was == now == "active" and pid != old_pid:
        return f"Local ComfyUI restarted{why}"
    return None


def _loop():
    prev = state()
    while True:
        time.sleep(POLL_SECONDS)
        cur = state()
        msg = change(prev, cur)
        prev = cur or prev
        if not msg or time.time() < _quiet_until or not all(config.pushover()):
            continue
        local = [f"{j.id} {j.action}" for j in jobs.manager.jobs.values() if j.lane == "local" and j.status == "running"]
        jobs.send_pushover("Comfy Studio", msg + "".join(f"\nrunning: {x}" for x in local))


def start():
    threading.Thread(target=_loop, daemon=True).start()
