"""The RunPod pod `animate` renders on when the ⚙ animate target is "runpod": one pod, found by NAME, created on demand on the
network volume runpod/seed.py filled (ComfyUI + the H3 models under their Comfy Cloud names, so animate graphs run unmapped)
and terminated, never stopped, when done. REST v1 (https://rest.runpod.io/v1), key RUNPOD_API_KEY.

Lifecycle: ensure_up() reuses a RUNNING pod or creates one (the first of config.RUNPOD_GPUS with stock in RUNPOD_DATACENTER, SECURE cloud, the volume at
/workspace, 8188/http) and waits until ComfyUI answers through the RunPod proxy. The pod's start command is runpod/watchdog.sh,
which deletes the pod after RUNPOD_IDLE_MINUTES without work, so it shuts down even when this machine doesn't. status() and
terminate() back the ⚙ page's pod line and its "Stop pod now".
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py
import config  # noqa: E402

API = "https://rest.runpod.io/v1"
NAME = "comfy-studio-animate"
WATCHDOG = config.ROOT / "runpod" / "watchdog.sh"
STARTS = config.LOGS_DIR / "runpod_starts.json"  # pods the studio started, newest last: boot seconds, first (cold) and second clip,
# $/hr and when status() last saw it up -- after it is gone that gives what the run cost (the billing API lags by hours)
READY_TIMEOUT = 900  # a fresh pod: image pull + the template's boot on the volume (404 -> 502 -> 200 on the proxy meanwhile);
# cold starts usually take 1-4 min but some hosts pull slowly, so only one silent past 15 is taken as stuck and deleted
GPU_RETRY_S = 30  # how often _create_waiting asks RunPod again while no GPU is free


class PodError(RuntimeError):
    pass


class NoGpu(PodError):
    """create(): none of RUNPOD_GPUS is free in the data center."""


def _api(path, body=None, method="GET"):
    if not config.RUNPOD_API_KEY:
        raise PodError("RUNPOD_API_KEY is not set: add it to .env")
    req = urllib.request.Request(f"{API}{path}", data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers={"Authorization": f"Bearer {config.RUNPOD_API_KEY}", "Content-Type": "application/json",
                                          "User-Agent": "comfy-studio"})  # the API's CDN refuses urllib's default agent (403)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise PodError(f"RunPod {method} {path}: {e.code} {e.read().decode(errors='replace')[:300]}") from e
    except urllib.error.URLError as e:
        raise PodError(f"can't reach the RunPod API ({e.reason})") from e
    return json.loads(raw) if raw.strip() else None


def balance() -> dict:
    """For the header chip when the animate target is runpod: the account's balance ($) and what it spends per hour right now
    (pods + volumes). Only RunPod's GraphQL API has these."""
    req = urllib.request.Request("https://api.runpod.io/graphql", method="POST",
                                 data=json.dumps({"query": "{ myself { clientBalance currentSpendPerHr } }"}).encode(),
                                 headers={"Authorization": f"Bearer {config.RUNPOD_API_KEY}", "Content-Type": "application/json",
                                          "User-Agent": "comfy-studio"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            me = json.loads(r.read())["data"]["myself"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise PodError(f"RunPod balance unavailable ({e})") from e
    return {"balance": me["clientBalance"], "spend_per_hr": me["currentSpendPerHr"]}


def url(pod_id: str) -> str:
    return f"https://{pod_id}-8188.proxy.runpod.net"


def find():
    """Our pod (any state but terminated), or None; with its machine, which carries the GPU it got (gpu_name)."""
    return next((p for p in _api("/pods?includeMachine=true") if p.get("name") == NAME and p.get("desiredStatus") != "TERMINATED"), None)


def answers(server: str) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{server}/system_stats", headers={"User-Agent": "comfy-studio"}),
                                    timeout=10) as r:  # the proxy answers urllib's default agent with 403
            return r.status == 200
    except (OSError, ValueError):
        return False


def _fetch_files() -> str:
    """"path sha256 url" per manifest LoRA with a url: the watchdog downloads the ones the volume lacks before ComfyUI starts."""
    try:
        m = json.loads(config.RUNPOD_MANIFEST.read_text())
    except (OSError, ValueError):
        return ""
    return "\n".join(f"{m['comfyui_dir']}/{p} {f['sha256']} {f['url']}" for p, f in m.get("files", {}).items()
                     if p.startswith("models/loras/") and f.get("url") and f.get("sha256"))


def create() -> dict:
    """A new pod on the first of RUNPOD_GPUS with stock (gpuTypePriority custom = try them in order); RunPod answers 400 when none
    has any in the data center, which is said plainly (no fallback to the cloud)."""
    if not config.RUNPOD_VOLUME_ID:
        raise PodError("RUNPOD_VOLUME_ID is not set: add it to .env")
    if not config.RUNPOD_DATACENTER:
        raise PodError("RUNPOD_DATACENTER is not set: add the volume's data center (e.g. EU-RO-1) to .env")
    body = {"name": NAME, "imageName": config.RUNPOD_IMAGE, "cloudType": "SECURE", "computeType": "GPU", "gpuCount": 1,
            "gpuTypeIds": config.RUNPOD_GPUS, "gpuTypePriority": "custom", "dataCenterIds": [config.RUNPOD_DATACENTER], "networkVolumeId": config.RUNPOD_VOLUME_ID,
            "volumeMountPath": "/workspace", "ports": ["8188/http"],
            "dockerEntrypoint": ["/bin/bash", "-c"], "dockerStartCmd": [WATCHDOG.read_text()],
            "env": {"IDLE_SECONDS": str(round(config.RUNPOD_IDLE_MINUTES * 60)),
                    "BOOT_SECONDS": str(READY_TIMEOUT + 120),  # the watchdog's own boot limit: past ours, so we delete it first
                    "FETCH_FILES": _fetch_files()}}
    try:
        return _api("/pods", body, "POST")
    except PodError as e:
        if "no longer any instances available" in str(e) or "no instances" in str(e).lower():
            raise NoGpu(f"RunPod has none of the RUNPOD_GPUS free in {config.RUNPOD_DATACENTER} right now: try again later, "
                           f"or switch the ⚙ animate target to cloud") from e
        raise


def gpu_name(pod: dict) -> str:
    """The GPU the pod actually got (RUNPOD_GPUS is a list)."""
    g = pod.get("gpu") or {}
    return g.get("displayName") or g.get("id") or (pod.get("machine") or {}).get("gpuTypeId") or "?"


def _starts() -> list:
    try:
        return json.loads(STARTS.read_text())
    except (OSError, ValueError):
        return []


def _save_starts(rows: list) -> None:
    STARTS.parent.mkdir(parents=True, exist_ok=True)
    STARTS.write_text(json.dumps(rows[-50:], indent=1))


def record_clip(server: str, seconds) -> None:
    """A clip's ComfyUI processing seconds on `server`: the first one after a cold start (which includes loading the models from
    the volume) and the second one (warm) go into that start's record; later clips and reused pods are ignored."""
    rows, pod_id = _starts(), server.split("//", 1)[-1].split("-8188.", 1)[0]
    rec = next((r for r in reversed(rows) if r["pod"] == pod_id), None)
    if not rec or not seconds or "warm_clip_s" in rec:
        return
    rec["warm_clip_s" if "first_clip_s" in rec else "first_clip_s"] = round(seconds)
    _save_starts(rows)


def last_start():
    """The newest cold start, for the header's RunPod menu (None before the first one)."""
    return (_starts() or [None])[-1]


def _seen(pod_id: str, cost) -> None:
    """status() saw the pod up: note when (at most every 30 s) and its $/hr in its start record."""
    rows = _starts()
    rec = next((r for r in reversed(rows) if r["pod"] == pod_id), None)
    if rec and (time.time() - rec.get("seen", 0) > 30 or not rec.get("cost_per_hr")):
        rec["seen"], rec["cost_per_hr"] = round(time.time()), rec.get("cost_per_hr") or cost
        _save_starts(rows)


def last_run():
    """What the newest studio-started pod cost once it is gone: from when it was rented until status() last saw it, at its $/hr
    (off by at most one poll, a cent or so). None when unknown."""
    rec = last_start()
    if not rec or not rec.get("cost_per_hr") or not rec.get("seen"):
        return None
    return {"cost": (rec["seen"] - rec["t"]) * rec["cost_per_hr"] / 3600, "minutes": (rec["seen"] - rec["t"]) / 60, "gpu": rec["gpu"],
            "ended": rec["seen"]}


def _create_waiting(log) -> dict:
    """create(), asking again every GPU_RETRY_S while none of RUNPOD_GPUS is free, for up to RUNPOD_GPU_WAIT_MINUTES."""
    t, said = time.time(), 0.0
    while True:
        try:
            return create()
        except NoGpu as e:
            waited = time.time() - t
            if waited >= config.RUNPOD_GPU_WAIT_MINUTES * 60:
                raise PodError(f"{e} (kept trying for {waited / 60:.0f} min)") from e
            if time.time() - said > 60:
                log(f"  no free GPU in {config.RUNPOD_DATACENTER} yet, retrying every {GPU_RETRY_S}s "
                    f"({waited / 60:.0f}/{config.RUNPOD_GPU_WAIT_MINUTES:g} min)")
                said = time.time()
            time.sleep(GPU_RETRY_S)


def ensure_up(log=print) -> str:
    """The URL of a pod whose ComfyUI answers: the running one, else a new one (a stopped or dead leftover is terminated first)."""
    t, created = time.time(), False
    pod = find()
    if pod and pod.get("desiredStatus") != "RUNNING":
        log(f"terminating the leftover pod {pod['id']} ({pod.get('desiredStatus')})")
        terminate(pod["id"])
        pod = None
    if pod:
        log(f"RunPod pod {pod['id']} is running on {gpu_name(pod)} (${pod.get('costPerHr', '?')}/hr)")
    else:
        log(f"starting RunPod pod… (first free of {len(config.RUNPOD_GPUS)} Blackwell types, {config.RUNPOD_DATACENTER})")
        pod = _create_waiting(log)
        created = True
        log(f"pod {pod['id']} created on {gpu_name(pod)} (${pod.get('costPerHr', '?')}/hr while up; it deletes itself after "
            f"{config.RUNPOD_IDLE_MINUTES:g} min idle)")
    server, said = url(pod["id"]), time.time()
    boot = time.time()  # the boot clock starts once the pod exists, not while waiting for a free GPU
    while not answers(server):
        if time.time() - boot > READY_TIMEOUT:
            terminate(pod["id"])
            raise PodError(f"pod {pod['id']}'s ComfyUI did not answer at {server} within {READY_TIMEOUT // 60} min, so it was deleted "
                           f"-- run again for a fresh pod")
        if (_api(f"/pods/{pod['id']}") or {}).get("desiredStatus") != "RUNNING":
            raise PodError(f"pod {pod['id']} stopped while booting -- see the RunPod console")
        if time.time() - said > 60:
            log(f"  still booting ({time.time() - boot:.0f}s)")
            said = time.time()
        time.sleep(10)
    log(f"RunPod pod ready in {time.time() - boot:.0f}s: {server}")
    if created:
        _save_starts(_starts() + [{"t": round(t), "pod": pod["id"], "gpu": gpu_name(pod), "ready_s": round(time.time() - boot),
                                    "cost_per_hr": pod.get("costPerHr"), "seen": round(time.time())}])
    return server


def _started(s):
    """lastStartedAt, as the API writes it ("2026-10-30 14:32:34.386 +0000 UTC"; the spec says ISO), as an aware datetime."""
    try:
        return datetime.fromisoformat(s[:19].replace("T", " ")).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def status() -> dict:
    """For the ⚙ page and the header: {"up": False} or the pod's id, state, whether ComfyUI answers, uptime seconds, $/hr and what
    this pod has cost so far (session_cost: uptime since it was rented, boot included, x $/hr -- how RunPod bills it)."""
    pod = find()
    if not pod:
        return {"up": False, "gpu": "first free Blackwell: " + config.RUNPOD_GPUS[0], "last_start": last_start(), "last_run": last_run()}
    started = _started(pod.get("lastStartedAt"))
    running = pod.get("desiredStatus") == "RUNNING"
    uptime = (datetime.now(timezone.utc) - started).total_seconds() if started and running else None
    cost = pod.get("adjustedCostPerHr") or pod.get("costPerHr")
    if running:
        _seen(pod["id"], cost)
    return {"up": running, "id": pod["id"], "state": pod.get("desiredStatus"), "ready": running and answers(url(pod["id"])),
            "uptime": uptime, "cost_per_hr": cost, "session_cost": uptime * cost / 3600 if uptime and cost else None,
            "gpu": gpu_name(pod), "url": url(pod["id"]), "last_start": last_start()}


def terminate(pod_id: str = None):
    """Delete the pod (the volume stays); returns its id, or None when there was none."""
    pod_id = pod_id or (find() or {}).get("id")
    if pod_id:
        _api(f"/pods/{pod_id}", method="DELETE")
    return pod_id
