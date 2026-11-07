"""Processing-time baselines for local and cloud ComfyUI jobs: data/logs/timings.json maps a graph signature to its recent run times,
so the next run of the same kind can say how long it should take (and the web UI can draw an accurate progress bar).

A time is ComfyUI's own execution_start -> execution_success span (queue waiting excluded); for the cloud, the job's
execution_start_time -> execution_end_time. The first job after
`targets.local.free` also loads its models, so cold and warm runs are kept apart."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py
import config  # noqa: E402

PATH = config.LOGS_DIR / "timings.json"
KEEP = 10  # runs kept per signature and temperature
SKIP_KEYS = {"seed", "noise_seed", "text", "prompt", "image", "filename_prefix", "lora_name", "strength_model", "wildcard"}
SKIP_CLASSES = {"LoraLoaderModelOnly", "LoadImage", "SaveImage"}  # LoRAs shift the time a little; leaving them out keeps one baseline per pass


def signature(graph: dict, label: str) -> str:
    """Pass label + a hash of what sets the run time: every node's class and plain (non-link) inputs, minus prompts, seeds and LoRAs."""
    parts = sorted(
        (n["class_type"], json.dumps({k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v  # ComfyUI's history turns 1 into 1.0
                                      for k, v in n["inputs"].items() if k not in SKIP_KEYS and not isinstance(v, list)}, sort_keys=True))
        for n in graph.values() if n["class_type"] not in SKIP_CLASSES)
    return f"{label}:{hashlib.md5(json.dumps(parts).encode()).hexdigest()[:10]}"


def sig(graph: dict) -> str:
    """signature() labelled by guess_label(): the key every caller uses, so a run and its estimate always agree."""
    return signature(graph, guess_label(graph))


def _load() -> dict:
    try:
        return json.loads(PATH.read_text())
    except (OSError, ValueError):
        return {}


def record(sig: str, seconds: float, cold: bool, job_id: str = None) -> bool:
    """Add one run. A `job_id` already recorded is skipped (backfills can be re-run); returns whether it was added."""
    data = _load()
    seen = data.setdefault("_seen", [])
    if job_id and job_id in seen:
        return False
    runs = data.setdefault(sig, {}).setdefault("cold" if cold else "warm", [])
    runs.append(round(seconds, 1))
    del runs[:-KEEP]
    if job_id:
        seen.append(job_id)
        del seen[:-3000]
    PATH.parent.mkdir(exist_ok=True)
    PATH.write_text(json.dumps(data, indent=1))
    return True


def record_cloud(graph: dict, job: dict) -> None:
    """Record a finished Comfy Cloud job (the dict from GET /api/jobs/{id}) under its graph's signature; jobs without timestamps are ignored."""
    start, end = job.get("execution_start_time"), job.get("execution_end_time")
    if start and end and end > start:
        record(sig(graph), (end - start) / 1000, False, job.get("id"))


def expected(sig: str, cold: bool):
    """Median of the recent runs of the same temperature (else of the other one), or None if never run."""
    entry = _load().get(sig, {})
    runs = entry.get("cold" if cold else "warm") or entry.get("warm" if cold else "cold")
    return round(statistics.median(runs), 1) if runs else None


def guess_label(graph: dict) -> str:
    """Which pass a submitted graph was (history has no label): by the node classes it contains."""
    classes = {n["class_type"] for n in graph.values()}
    if any("ImageToVideo" in c for c in classes):
        return "animate"
    if any("SeedVR" in c for c in classes):
        return "seedvr"
    if any("FaceDetailer" in c for c in classes) and not any(c.startswith("ControlNet") for c in classes):
        return "face"
    if "LoadImage" in classes:
        return "il"
    return "full" if any(c.startswith("ControlNet") for c in classes) else "krea2"


def backfill(server: str = config.LOCAL_COMFY_URL, limit: int = 200) -> int:
    """Seed data/logs/timings.json from the local ComfyUI's job history (its execution_start/success stamps), so the first estimates
    don't wait for a fresh run. A job is cold when its model loader wasn't cached. Returns the number of runs added; run twice = duplicates."""
    import urllib.request
    hist = json.load(urllib.request.urlopen(f"{server}/history?max_items={limit}"))
    added = 0
    for v in sorted(hist.values(), key=lambda h: h["prompt"][0]):
        msgs = {m[0]: m[1] for m in v["status"].get("messages", [])}
        if v["status"].get("status_str") != "success" or "execution_start" not in msgs or "execution_success" not in msgs:
            continue
        graph = v["prompt"][2]
        loaders = [k for k, n in graph.items() if n["class_type"] in ("UNETLoader", "CheckpointLoaderSimple")]
        cold = any(k not in msgs.get("execution_cached", {}).get("nodes", []) for k in loaders)
        added += record(sig(graph), (msgs["execution_success"]["timestamp"] - msgs["execution_start"]["timestamp"]) / 1000, cold, v["prompt"][1])
    return added


def backfill_cloud(limit: int = 150) -> int:
    """Seed the baselines from the newest `limit` Comfy Cloud jobs (one request each for the graph; needs COMFY_CLOUD_KEY). Returns runs added."""
    import comfy_api
    added = 0
    for j in comfy_api.get("/api/jobs?limit=%d" % limit).get("jobs", []):
        if j.get("status") != "completed" or j["id"] in _load().get("_seen", []):
            continue
        job = comfy_api.get(f"/api/jobs/{j['id']}")
        graph = (job.get("workflow") or {}).get("prompt")
        if graph:
            before = len(_load().get("_seen", []))
            record_cloud(graph, job)
            added += len(_load().get("_seen", [])) > before
    return added


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if sys.argv[1:] == ["backfill"]:
        print(f"{backfill()} past local runs and {backfill_cloud()} cloud runs added to {PATH}")
    else:
        sys.exit("usage: timings.py backfill   (seed the baselines from the local ComfyUI's job history and the newest Comfy Cloud jobs)")
