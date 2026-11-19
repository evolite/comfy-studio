"""The RunPod pod `animate` renders on when the ⚙ animate target is "runpod": one pod, found by NAME, created on demand on the
network volume runpod/seed.py filled (ComfyUI + the H3 models under their Comfy Cloud names, so animate graphs run unmapped)
and terminated, never stopped, when done. REST v2 (https://api.runpod.io/v2), key RUNPOD_API_KEY; balance() stays on GraphQL
(v2's /account has no balance or spend).

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

API = "https://api.runpod.io/v2"
ALIVE = {"PROVISIONING", "STARTING", "RUNNING"}  # v2 pod status while it is (or is becoming) up; v1 said desiredStatus RUNNING for all three
STOCK = {"HIGH", "MEDIUM", "LOW"}  # catalog availability worth a create; NONE (or absent) is not tried
NAME = "comfy-studio-animate"
CONTAINER_DISK_GB = 50  # the image's ComfyUI + pip cache; models are on the network volume
WATCHDOG = config.ROOT / "runpod" / "watchdog.sh"
STARTS = config.LOGS_DIR / "runpod_starts.json"  # pods the studio started, newest last: boot seconds, first (cold) and second clip,
# $/hr and when status() last saw it up -- after it is gone that gives what the run cost (the billing API lags by hours)
READY_TIMEOUT = 900  # a fresh pod: image pull + the template's boot on the volume (404 -> 502 -> 200 on the proxy meanwhile);
# cold starts usually take 1-4 min but some hosts pull slowly, so only one silent past 15 is taken as stuck and deleted
GPU_RETRY_S = 30  # how often _create_waiting asks RunPod again while no GPU is free
WAKE = config.LOGS_DIR / "runpod_wake"  # touched when animate asks for a pod, when it is ready, created or deleted: status() in other processes re-asks at once
STATUS_TTL = {"up": 10, "down": 300, "limited": 900, "minute": 60}  # status() caches the pod list: the ⚙ page and the KDE widget poll it all
# day, and RunPod's REST API has a daily rate limit (429) that a 10 s poll with no pod used up on its own


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


# rp-migrate: keep-v1 start (v2 has no account balance or spend)
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
# rp-migrate: keep-v1 end


def url(pod_id: str) -> str:
    return f"https://{pod_id}-8188.proxy.runpod.net"


def find():
    """Our pod (any state but terminated), or None."""
    return next((p for p in _api("/pods")["pods"] if p.get("name") == NAME and p.get("status") != "TERMINATED"), None)  # rp-migrate: ignore (API is v2)


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


def create(gpus=None) -> dict:
    """A new pod on the first of RUNPOD_GPUS with stock. The 96 GB cards (RUNPOD_BIG_GPUS) are asked for on their own first and
    the full list only when none is free. None free in the data center is said plainly (no fallback to the cloud)."""
    if not config.RUNPOD_VOLUME_ID:
        raise PodError("RUNPOD_VOLUME_ID is not set: add it to .env")
    if not config.RUNPOD_DATACENTER:
        raise PodError("RUNPOD_DATACENTER is not set: add the volume's data center (e.g. EU-RO-1) to .env")
    gpus = gpus or config.RUNPOD_GPUS
    first = [g for g in gpus if g in config.RUNPOD_BIG_GPUS]
    for tier in ([first, gpus] if first and first != gpus else [gpus]):
        try:
            return _post(tier)
        except NoGpu:
            if tier is gpus:
                raise


def _stock(gpus) -> list:
    """The GPU types in `gpus` with any stock in RUNPOD_DATACENTER, in list order (the list is the preference). When the catalog
    itself fails (RunPod answers 500s at times) every type is tried, as v1 did."""
    try:
        cat = _api("/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&count=1")["gpus"]
    except PodError as e:
        if not any(f": {c} " in str(e) for c in (429, 500, 502, 503, 504)) and "can't reach" not in str(e):
            raise
        return list(gpus)
    level = {g["id"]: next((d.get("availability") for d in g.get("dataCenters", []) if d.get("id") == config.RUNPOD_DATACENTER), None)
             for g in cat}
    return [g for g in gpus if level.get(g) in STOCK]


def _post(gpus) -> dict:
    """v2 rents one GPU type per pod: try the stocked ones in turn. A 4xx other than capacity (bad body, auth) is raised at once.
    `disk` is required in practice: v2 has no default (v1's was 50 GB), and without it every create fails with a capacity-looking
    400 ("no longer any instances available") or "provide ... pod configuration parameters"."""
    body = {"name": NAME, "image": config.RUNPOD_IMAGE, "cloud": "SECURE", "disk": CONTAINER_DISK_GB,
            "dataCenterIds": [config.RUNPOD_DATACENTER], "mounts": {"network": [{"volumeId": config.RUNPOD_VOLUME_ID, "path": "/workspace"}]},
            "ports": ["8188/http"], "entrypoint": ["/bin/bash", "-c"], "cmd": [WATCHDOG.read_text()],
            "env": {"IDLE_SECONDS": str(round(config.RUNPOD_IDLE_MINUTES * 60)),
                    "BOOT_SECONDS": str(READY_TIMEOUT + 120),  # the watchdog's own boot limit: past ours, so we delete it first
                    "FETCH_FILES": _fetch_files()}}
    last = "the catalog lists none in stock"
    for gpu in _stock(gpus):
        try:
            pod = _api("/pods", {**body, "gpu": {"id": gpu, "count": 1}}, "POST")  # rp-migrate: ignore (API is v2)
        except PodError as e:
            if any(f": {c} " in str(e) for c in (401, 402, 403, 413, 422)) or ": 400 " in str(e) and "instances available" not in str(e):
                raise  # the request itself (or the account) is wrong: the next GPU type fails the same way
            last = f"{gpu}: {e}"
            continue
        _wake()
        return pod
    raise NoGpu(f"RunPod has none of the RUNPOD_GPUS free in {config.RUNPOD_DATACENTER} right now ({last[:200]}): try again later, "
                f"or switch the ⚙ animate target to cloud")


def gpu_name(pod: dict) -> str:
    """The GPU the pod actually got (RUNPOD_GPUS is a list)."""
    return (pod.get("gpu") or {}).get("id") or "?"


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


def _create_waiting(log, gpus=None) -> dict:
    """create(gpus), asking again every GPU_RETRY_S while none of RUNPOD_GPUS is free, for up to RUNPOD_GPU_WAIT_MINUTES."""
    t, said = time.time(), 0.0
    while True:
        try:
            return create(gpus)
        except NoGpu as e:
            waited = time.time() - t
            if waited >= config.RUNPOD_GPU_WAIT_MINUTES * 60:
                raise PodError(f"{e} (kept trying for {waited / 60:.0f} min)") from e
            if time.time() - said > 60:
                log(f"  no free GPU in {config.RUNPOD_DATACENTER} yet, retrying every {GPU_RETRY_S}s "
                    f"({waited / 60:.0f}/{config.RUNPOD_GPU_WAIT_MINUTES:g} min)")
                said = time.time()
            time.sleep(GPU_RETRY_S)


def _on(pod: dict, gpus) -> bool:
    """Did the pod get one of these GPU types?"""
    return gpu_name(pod) in gpus


def _reusable(pod, big, gpus, log):
    """The found pod if animate can use it: a stopped or dead leftover is terminated, and so is a small pod when the clip needs a
    96 GB GPU (the runpod lane runs one job at a time, so it has nothing of ours on it)."""
    if pod and pod.get("status") not in ALIVE:
        log(f"terminating the leftover pod {pod['id']} ({pod.get('status')})")
        terminate(pod["id"])
        return None
    if pod and big and not _on(pod, gpus):
        log(f"clips over {config.RUNPOD_BIG_SECONDS:g} s need a 96 GB GPU: replacing pod {pod['id']} ({gpu_name(pod)})")
        terminate(pod["id"])
        return None
    return pod


def _wait_ready(pod, created, log) -> bool:
    """Wait until the pod's ComfyUI answers. False = a reused pod turned out to be gone (its watchdog deleted it)."""
    server, said = url(pod["id"]), time.time()
    boot = time.time()  # the boot clock starts once the pod exists, not while waiting for a free GPU
    while not answers(server):
        if time.time() - boot > READY_TIMEOUT:
            terminate(pod["id"])
            raise PodError(f"pod {pod['id']}'s ComfyUI did not answer at {server} within {READY_TIMEOUT // 60} min, so it was deleted "
                           f"-- run again for a fresh pod")
        try:
            state = (_api(f"/pods/{pod['id']}") or {}).get("status")
        except PodError as e:  # find() can still list a pod its watchdog is deleting: that one is gone, so start a new one
            if created or ": 404 " not in str(e):
                raise
            log(f"pod {pod['id']} is gone (its watchdog deleted it), starting a new one")
            return False
        if state not in ALIVE:
            raise PodError(f"pod {pod['id']} stopped while booting -- see the RunPod console")
        if time.time() - said > 60:
            log(f"  still booting ({time.time() - boot:.0f}s)")
            said = time.time()
        time.sleep(10)
    log(f"RunPod pod ready in {time.time() - boot:.0f}s: {server}")
    return True


def ensure_up(log=print, big=False) -> str:
    """The URL of a pod whose ComfyUI answers: the running one, else a new one (a stopped or dead leftover is terminated first).
    big = a clip longer than config.RUNPOD_BIG_SECONDS: only RUNPOD_BIG_GPUS will do, so a smaller running pod is replaced."""
    _wake()
    t, created = time.time(), False
    gpus = config.RUNPOD_BIG_GPUS if big else config.RUNPOD_GPUS
    pod = _reusable(find(), big, gpus, log)
    if pod:
        log(f"RunPod pod {pod['id']} is running on {gpu_name(pod)} (${pod.get('cost', '?')}/hr)")
    else:
        log(f"starting RunPod pod… (first free of {len(gpus)} {'96 GB ' if big else ''}Blackwell types, {config.RUNPOD_DATACENTER})")
        pod = _create_waiting(log, gpus)
        created = True
        log(f"pod {pod['id']} created on {gpu_name(pod)} (${pod.get('cost', '?')}/hr while up; it deletes itself after "
            f"{config.RUNPOD_IDLE_MINUTES:g} min idle)")
    boot = time.time()
    if not _wait_ready(pod, created, log):
        return ensure_up(log, big)
    if created:
        _save_starts(_starts() + [{"t": round(t), "pod": pod["id"], "gpu": gpu_name(pod), "ready_s": round(time.time() - boot),
                                    "cost_per_hr": pod.get("cost"), "seen": round(time.time())}])
    _wake()
    return url(pod["id"])


def _started(s):
    """startedAt (ISO in v2; v1 wrote "2026-10-30 14:32:34.386 +0000 UTC") as an aware UTC datetime."""
    try:
        return datetime.fromisoformat(s[:19].replace("T", " ")).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _wake() -> None:
    WAKE.parent.mkdir(parents=True, exist_ok=True)
    WAKE.touch()


def _wake_t() -> float:
    try:
        return WAKE.stat().st_mtime
    except OSError:
        return 0.0


_cache = {"t": 0.0, "ttl": 0.0, "pod": None, "err": None}


def _find_cached():
    """find() for the pollers, at most every STATUS_TTL: 10 s while a pod exists, 5 min while none (a create or delete anywhere
    touches WAKE, which ends the wait), 15 min after a daily-limit 429, 1 min after a per-minute one."""
    now = time.time()
    if now - _cache["t"] > _cache["ttl"] or _wake_t() > _cache["t"]:
        try:
            pod = find()
            _cache.update(pod=pod, err=None, ttl=STATUS_TTL["up" if pod else "down"])
        except PodError as e:
            limit = "minute" if "minute window" in str(e) else "limited"
            _cache.update(pod=None, err=e, ttl=STATUS_TTL[limit if ": 429 " in str(e) else "up"])
        _cache["t"] = now
    if _cache["err"]:
        raise _cache["err"]
    return _cache["pod"]


def status() -> dict:
    """For the ⚙ page and the header: {"up": False} or the pod's id, state, whether ComfyUI answers, uptime seconds, $/hr and what
    this pod has cost so far (session_cost: uptime since it was rented, boot included, x $/hr -- how RunPod bills it)."""
    pod = _find_cached()
    if not pod:
        return {"up": False, "gpu": "first free Blackwell: " + config.RUNPOD_GPUS[0], "last_start": last_start(), "last_run": last_run()}
    started = _started(pod.get("startedAt"))
    running = pod.get("status") in ALIVE
    uptime = (datetime.now(timezone.utc) - started).total_seconds() if started and running else None
    cost = pod.get("cost")
    if running:
        _seen(pod["id"], cost)
    return {"up": running, "id": pod["id"], "state": pod.get("status"), "ready": running and answers(url(pod["id"])),
            "uptime": uptime, "cost_per_hr": cost, "session_cost": uptime * cost / 3600 if uptime and cost else None,
            "gpu": gpu_name(pod), "url": url(pod["id"]), "last_start": last_start()}


def terminate(pod_id: str = None):
    """Delete the pod (the volume stays); returns its id, or None when there was none."""
    pod_id = pod_id or (find() or {}).get("id")
    if pod_id:
        _api(f"/pods/{pod_id}", method="DELETE")
        _wake()
    return pod_id
