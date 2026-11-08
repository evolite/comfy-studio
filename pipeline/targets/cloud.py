"""Comfy Cloud as a target: the REST client (auth, submit, poll, upload, download) and CloudTarget, which queues a batch of
stage jobs up front, polls them together (resubmitting a job stuck in a not-yet-running state) and downloads each as it lands.

API notes: auth is the X-API-Key header; /api/job/{id}/status has state only,
/api/jobs/{id} (plural) has the outputs map (node_id -> {key: [file dicts]}).
"""
import json
import os
import queue
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: config.py
import config  # noqa: E402
import lora_registry  # noqa: E402
import timings  # noqa: E402
from stagegraph import stage_graph  # noqa: E402

from .base import StageJob, Target  # noqa: E402
from .models import local_models  # noqa: E402

ROOT = config.ROOT
OUTPUT_DIR = config.PIPELINES_DIR
CLOUD_BASE = "https://cloud.comfy.org"
CLIENT_ID = "comfycloud-pipeline-cloud"  # cloud sends step progress only to the submitting client id; dashboard.py listens as this one
TERMINAL = {"success", "error", "non_retryable_error", "lost", "cancelled"}
STALL_SECONDS = 60  # a job that sits in the same not-yet-running state this long is cancelled and resubmitted
MAX_RESTARTS = 3
# states that are not a stall: running, or legitimately waiting behind the account's concurrency cap
NOT_STALLED = TERMINAL | {"executing", "executed", "queued_limited"}
FINAL_PNG = "final.png"


def api_key() -> str:
    key = os.environ.get("COMFY_CLOUD_KEY")
    if not key:
        sys.exit("COMFY_CLOUD_KEY is not set: export it in your shell or add it to .env "
                 "(don't paste the key into chat).")
    return key


def _request(path, data=None, headers=None, method="GET"):
    req = urllib.request.Request(f"{CLOUD_BASE}{path}", data=data, method=method,
                                 headers={"X-API-Key": api_key(), **(headers or {})})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 402:  # the account can't pay for jobs (e.g. the subscription ended); no retry will help
            body = e.read().decode(errors="replace")[:300]
            sys.exit(f"Comfy Cloud refused the job: 402 Payment Required -- check the subscription/credits at "
                     f"https://cloud.comfy.org (local passes still work). {body}")
        raise


def get(path: str) -> dict:
    return json.loads(_request(path))


def submit(graph: dict):
    result = json.loads(_request("/api/prompt", json.dumps({"prompt": graph, "client_id": CLIENT_ID}).encode(),
                                 {"Content-Type": "application/json"}, "POST"))
    prompt_id = result.get("prompt_id") or result.get("id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id in submit response: {result}")
    return prompt_id, result


def job_state(prompt_id: str):
    status = get(f"/api/job/{prompt_id}/status")
    return status.get("status") or status.get("state"), status


def poll_status(prompt_id: str, interval: float = 5.0, timeout: float = 900.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state, status = job_state(prompt_id)
        print(f"  [{time.strftime('%H:%M:%S')}] status: {state}", flush=True)
        if state in TERMINAL:
            return status
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for job {prompt_id}")


def stalled(seen: dict, prompt_id: str, state: str, limit: float = STALL_SECONDS) -> bool:
    """True once `prompt_id` has stayed in one not-yet-running state for `limit` seconds. `seen` is the caller's scratch dict."""
    prev = seen.get(prompt_id)
    if not prev or prev[0] != state:
        seen[prompt_id] = (state, time.time())
        return False
    return state not in NOT_STALLED and time.time() - prev[1] > limit


def cancel(prompt_id: str) -> None:
    """Pull a job off the cloud queue (best effort)."""
    try:
        _request("/api/queue", json.dumps({"delete": [prompt_id]}).encode(), {"Content-Type": "application/json"}, "POST")
    except urllib.error.URLError:
        pass


def fetch_job(prompt_id: str) -> dict:
    return get(f"/api/jobs/{prompt_id}")


def upload_image(path: Path) -> str:
    """POST /api/upload/image; returns the name to put in a LoadImage node."""
    boundary = "----ComfyCloudBoundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    return json.loads(_request("/api/upload/image", body,
                               {"Content-Type": f"multipart/form-data; boundary={boundary}"}, "POST"))["name"]


def download(file: dict, dest: Path):
    url = (f"/api/view?filename={file['filename']}&subfolder={file.get('subfolder', '')}"
           f"&type={file.get('type', 'output')}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_request(url))


def _save_node_outputs(node_id, node_outputs, run_dir: Path, labels: dict, fetch, saved) -> None:
    """Download every file one output node produced into images/<type>/<node>_<label>[_<key>]/."""
    label = labels.get(node_id, node_id.replace(":", "_"))
    for key, files in node_outputs.items():
        if not (isinstance(files, list) and files and isinstance(files[0], dict) and "filename" in files[0]):
            continue  # e.g. a text preview output
        suffix = "" if key == "images" else f"_{key}"
        node_dir = run_dir / "images" / files[0].get("type", "output") / f"{node_id.replace(':', '_')}_{label}{suffix}"
        for f in files:
            dest = node_dir / (f.get("display_name") or f["filename"])
            try:
                fetch(f, dest)
                saved.append((node_id, dest))
                print(f"  saved: {dest}", flush=True)
            except urllib.error.HTTPError as e:
                print(f"  failed to fetch {dest.name} (node {node_id}): {e}", flush=True)


def save_outputs(job: dict, run_dir: Path, labels: dict = None, final_node: str = None, fetch=download):
    """Download every output file to run_dir/images/... and copy final_node's image to run_dir/final.png.
    Returns [(node_id, path)]. `fetch(file, dest)` = where the files come from: Comfy Cloud, or a ComfyUI
    server for a /history entry (same outputs shape)."""
    labels = labels or {}
    saved = []
    for node_id, node_outputs in job.get("outputs", {}).items():
        _save_node_outputs(node_id, node_outputs, run_dir, labels, fetch, saved)
    finals = [d for nid, d in saved if nid == final_node]
    if finals:
        final = run_dir / FINAL_PNG
        shutil.copyfile(finals[0], final)
        print(f"  final image: {final}", flush=True)
    return saved


class CloudTarget(Target):
    name = "cloud"
    lane = "cloud"
    params = {}

    def available(self) -> bool:
        return config.cloud_key_present()

    def check(self, graph: dict) -> list:
        """A Krea2 UNet, VAE or a LoRA that only exists on this machine would fail on the cloud, after the credits are spent on the queue."""
        local_only = set(local_models()["local_only_unets"])
        problems = [f"krea_unet {n['inputs']['unet_name']} exists only on this machine: run it locally (--preview --local), or set another "
                    f"krea_unet (e.g. krea2_turbo_fp8_scaled.safetensors) for the cloud"
                    for n in graph.values() if n["class_type"] == "UNETLoader" and n["inputs"].get("unet_name") in local_only]
        problems += [f"krea_vae {n['inputs']['vae_name']} exists only on this machine: run it locally (--preview --local), or unset "
                     f"krea_vae for the cloud" for n in graph.values()
                     if n["class_type"] == "VAELoader" and n["inputs"].get("vae_name") in local_models().get("local_only_vaes", [])]
        local_loras = lora_registry.local_only_files()
        problems += [f"{n['_meta'].get('title', 'LoRA')} ({n['inputs']['lora_name']}) is local-only (cloud: false in loras.json): "
                     f"set it to 0, or run locally (--preview --local)"
                     for n in graph.values() if n["class_type"] == "LoraLoaderModelOnly" and n["inputs"].get("lora_name") in local_loras]
        return problems

    def upload(self, path: Path, name: str = None) -> str:
        return upload_image(path)

    def submit(self, job: StageJob) -> None:
        job.debug_dir.mkdir(parents=True, exist_ok=True)
        (job.debug_dir / "graph.json").write_text(json.dumps(job.graph, indent=2))
        prompt_id, resp = submit(job.graph)
        (job.debug_dir / "submit_response.json").write_text(json.dumps(resp, indent=2))
        eta = timings.expected(timings.sig(job.graph), False)
        print(f"  [{job.debug_dir.name}] submitted -> {prompt_id}" + (f" expected {eta:.0f}s" if eta else ""), flush=True)
        job.handle = {"id": prompt_id, "state": "submitted", "restarts": 0, "t0": time.time()}

    def _finish(self, job: StageJob) -> None:
        h = job.handle
        if h["state"] != "success":
            job.error = f"state={h['state']}"
            return
        full = fetch_job(h["id"])
        (job.debug_dir / "job_full.json").write_text(json.dumps(full, indent=2))
        timings.record_cloud(job.graph, full)
        final = job.debug_dir / FINAL_PNG
        final.unlink(missing_ok=True)  # never reuse a previous run's final
        save_outputs(full, job.debug_dir, job.node_labels, job.final_node)
        if final.exists():
            job.final = final
        else:
            job.error = f"succeeded but no {FINAL_PNG} found -- check {job.debug_dir}"

    def _poll_job(self, prompt_id: str, j: StageJob, pending: dict, seen: dict) -> str | None:
        """One poll round of a pending job: state-change log, terminal hand-off ("done"), or stall resubmit."""
        h = j.handle
        state, status = job_state(prompt_id)
        if state != h["state"]:
            print(f"  [{j.debug_dir.name}] {state}", flush=True)
            h["state"] = state
        if state in TERMINAL:
            (j.debug_dir / "final_status.json").write_text(json.dumps(status, indent=2))
            del pending[prompt_id]
            return "done"
        if h["restarts"] < MAX_RESTARTS and stalled(seen, prompt_id, state):
            print(f"  [{j.debug_dir.name}] still '{state}' after {STALL_SECONDS}s, resubmitting", flush=True)
            cancel(prompt_id)
            del pending[prompt_id]
            h["restarts"] += 1
            h["id"], _ = submit(j.graph)
            h["state"] = "submitted"
            pending[h["id"]] = j
        return None

    def collect(self, jobs: list):
        """Poll every job together; yield each one as soon as it is downloaded."""
        pending = {j.handle["id"]: j for j in jobs}
        done_q, seen, outstanding = queue.Queue(), {}, 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            while pending or outstanding:
                outstanding += self._poll_ready(pending, seen, pool, done_q)
                try:
                    while True:
                        yield done_q.get(timeout=5 if pending else 0.2)
                        outstanding -= 1
                except queue.Empty:
                    pass

    def _poll_ready(self, pending: dict, seen: dict, pool, done_q) -> int:
        """One poll pass over the pending jobs: the finished ones start downloading in the pool. The loop iterates a
        snapshot because _poll_job drops the entry it returns "done" for; returns how many downloads started."""
        started = 0
        for prompt_id in pending.copy():
            j = pending[prompt_id]
            if self._poll_job(prompt_id, j, pending, seen) == "done":
                started += 1
                pool.submit(self._finish_async, j, done_q)
        return started

    def _finish_async(self, j, done_q) -> None:
        try:
            self._finish(j)
        except Exception as e:  # noqa: BLE001  # a failed download must not lose the other stages
            j.error = f"{type(e).__name__}: {e}"
        j.waited = time.time() - j.handle["t0"]
        done_q.put(j)


def run_batch(engine, stages: list, common: dict, pipeline_name: str = "pipeline",
              batch_dir: Path = None, indices: list = None) -> dict:
    """Run an engine over raw stage dicts on the cloud (pipeline/run_pipeline.py). The project commands go through pipeline/runner.py;
    this keeps the old shape: submit all, poll, save stage_N.png flat in batch_dir with everything else under _debug/stage_N/."""
    if batch_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        batch_dir = OUTPUT_DIR / f"{ts}_{pipeline_name}"
    indices = indices or list(range(1, len(stages) + 1))
    target = CloudTarget()
    jobs = [StageJob(i, stage_graph(engine, s, common), "", batch_dir / "_debug" / f"stage_{i}", engine.FINAL_IMAGE_NODE, engine.NODE_LABELS)
            for i, s in zip(indices, stages)]
    print(f"=== Submitting {len(jobs)} jobs -> {batch_dir} ({engine.__name__}) ===", flush=True)
    for j in jobs:
        target.submit(j)
    print(f"\n=== Polling {len(jobs)} jobs; downloading each as it finishes ===", flush=True)
    result = []
    for j in target.collect(jobs):
        final_path = None
        if j.final:
            final_path = batch_dir / f"stage_{j.n}.png"
            shutil.copyfile(j.final, final_path)
            print(f"  [{j.debug_dir.name}] -> {final_path}", flush=True)
        else:
            print(f"  [{j.debug_dir.name}] {j.error}", flush=True)
        result.append({"name": j.debug_dir.name, "index": j.n, "final_path": final_path, "state": j.handle["state"]})
    return {"batch_dir": batch_dir, "jobs": sorted(result, key=lambda r: r["index"])}
