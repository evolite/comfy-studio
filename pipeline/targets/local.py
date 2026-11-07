"""A local ComfyUI as a target: the same interface as Comfy Cloud, plus what a machine of your own needs -- cloud model/node names
mapped to local files, the Krea2 step on an fp8 checkpoint, models unloaded between batches that need different ones, a
pre-flight check against the server's /object_info, and free/restart controls.

Graphs arrive in cloud-canonical form (what the engines build); prepare() is the only place that knows this machine's file names,
which live in defaults/local_models.json (override per key in data/library/local_models.json).
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: config.py
import config  # noqa: E402
import timings  # noqa: E402

from .base import StageJob, Target, TargetUnavailable  # noqa: E402
from .models import local_lora, local_models  # noqa: E402

CLIENT_ID = "comfycloud-pipeline"  # ComfyUI sends step progress only to the submitting client id; dashboard.py listens as this one
NAME_FIELDS = ("ckpt_name", "control_net_name", "model_name", "lora_name")
JSON_HDR = {"Content-Type": "application/json"}
MODEL_LOADERS = {"UNETLoader": "unet_name", "CheckpointLoaderSimple": "ckpt_name", "CLIPLoader": "clip_name", "VAELoader": "vae_name",
                 "ControlNetLoader": "control_net_name", "UpscaleModelLoader": "model_name"}  # LoRAs excluded: ComfyUI re-patches those cheaply


def _call(server, path, data=None, headers=None):
    # a User-Agent of our own: the RunPod proxy in front of a pod's ComfyUI answers urllib's default one with 403
    req = urllib.request.Request(f"{server}{path}", data=data, headers={"User-Agent": "comfy-studio", **(headers or {})})
    with urllib.request.urlopen(req) as r:
        return r.read()


def free(server: str) -> None:
    """Unload every model and free cached memory on the server; a batch that needs other models starts with this."""
    _call(server, "/free", json.dumps({"unload_models": True, "free_memory": True}).encode(), JSON_HDR)
    time.sleep(3)


def models_of(graph: dict) -> frozenset:
    """The model files a graph loads (loader class, file), what decides whether the last run left the right ones in memory."""
    return frozenset((n["class_type"], n["inputs"][MODEL_LOADERS[n["class_type"]]]) for n in graph.values()
                     if n["class_type"] in MODEL_LOADERS and isinstance(n["inputs"].get(MODEL_LOADERS[n["class_type"]]), str))


def same_as_last(server: str, graph: dict) -> bool:
    """Did the newest job in the server's history load exactly this graph's models (so they are still in memory)? False when unknown
    (empty history after a restart, unreachable server)."""
    try:
        hist = json.loads(_call(server, "/history?max_items=1"))
    except (OSError, ValueError):
        return False
    last = next(iter(hist.values()), None)
    return bool(last) and last["status"].get("status_str") == "success" and models_of(last["prompt"][2]) == models_of(graph)


def swap_used_pct() -> float:
    """This machine's swap use in percent (0 without swap or /proc/meminfo)."""
    try:
        m = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    except (OSError, ValueError):
        return 0.0
    return 100 * (m["SwapTotal"] - m["SwapFree"]) / m["SwapTotal"] if m.get("SwapTotal") else 0.0


def memory_tight(server: str):
    """Why the server should be freed even though its models are the right ones (None = memory is fine): its free RAM is under
    LOCAL_FREE_MIN_RAM_GB, or, when it runs on this machine, swap use is over LOCAL_FREE_MAX_SWAP_PCT. ComfyUI only trims its
    cache once RAM is nearly gone, while zram swap fills first and systemd-oomd kills it at 90% swap."""
    try:
        free_gb = json.loads(_call(server, "/system_stats"))["system"]["ram_free"] / 2 ** 30
    except (OSError, ValueError, KeyError):
        free_gb = None
    if free_gb is not None and free_gb < config.LOCAL_FREE_MIN_RAM_GB:
        return f"{free_gb:.1f} GB RAM free"
    if urllib.parse.urlparse(server).hostname in ("127.0.0.1", "localhost", "::1"):
        swap = swap_used_pct()
        if swap > config.LOCAL_FREE_MAX_SWAP_PCT:
            return f"swap {swap:.0f}% used"
    return None


def free_if_changed(server: str, graph: dict) -> bool:
    """Unload everything unless the last run used the same models and memory isn't tight; returns whether it freed."""
    if same_as_last(server, graph):
        why = memory_tight(server)
        if not why:
            return False
        print(f"  same models as the last run, but {why} -- freeing memory first", flush=True)
    free(server)
    return True


TEMPLATE_KREA_VAE = "qwen_image_vae.safetensors"


def localize_krea(graph: dict) -> dict:
    """The Krea2 nodes on local files: the UNet loader becomes the local fp8 checkpoint (unless `krea_unet` is one of the local-only
    UNets, then that UNet is loaded as it is), the template's VAE (a `krea_vae` setting is kept as is) and the Krea2 chain's LoRA
    names (30:3xx) are mapped."""
    m = local_models()
    g = json.loads(json.dumps(graph))
    if g["30:10"]["inputs"].get("unet_name") not in m["local_only_unets"]:
        g["30:10"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": m["krea_checkpoint"]}}
    if g["30:12"]["inputs"]["vae_name"] == TEMPLATE_KREA_VAE:
        g["30:12"]["inputs"]["vae_name"] = m["krea_vae"]
    for nid, node in g.items():
        name = node["inputs"].get("lora_name")
        if nid.startswith("30:") and isinstance(name, str):
            node["inputs"]["lora_name"] = local_lora(name, krea=True)
    return g


def map_names(g: dict) -> dict:
    """Cloud model names -> local files, in place (an unmapped IL checkpoint is assumed to be in the local IL folder, a bare
    LoRA name in `il_lora_folder`)."""
    m = local_models()
    for node in g.values():
        ins = node["inputs"]
        for k, v in ins.items():
            match (k, v):
                case (_, name) if k in NAME_FIELDS and name in m["model_names"]:
                    ins[k] = m["model_names"][name]
                case ("lora_name", str() as name) if "/" not in name:  # Krea2 ones were mapped by localize_krea
                    ins[k] = local_lora(name, krea=False)
                case ("ckpt_name", str() as name) if "/" not in name and not name.startswith("krea"):
                    ins[k] = f"{m['il_checkpoint_folder']}{name}"
    return g


def _drop_inert_grain(g: dict) -> None:
    """WAS Image Film Grain rejects a density/intensity of 0: bypass the node when grain is off, rewiring its inputs to its output."""
    inert = [n for n, v in g.items() if v["class_type"] == "Image Film Grain" and 0 in (v["inputs"]["density"], v["inputs"]["intensity"])]
    for nid in inert:
        src = g.pop(nid)["inputs"]["image"]
        for node in g.values():
            for k, v in node["inputs"].items():
                if isinstance(v, list) and v[0] == nid:
                    node["inputs"][k] = src


def localize(graph: dict) -> dict:
    """Any cloud-canonical graph on this machine's names: Krea2 nodes (if present) -> local, model names mapped, Film Grain -> WAS
    Image Film Grain (skipped when grain is off: WAS rejects 0)."""
    g = localize_krea(graph) if "30:10" in graph else json.loads(json.dumps(graph))
    map_names(g)
    for node in g.values():
        if node["class_type"] == "Film Grain":
            node["class_type"] = "Image Film Grain"
            node["inputs"].pop("repeats", None)
    _drop_inert_grain(g)
    return g


def _check_node(nid, node: dict, oi: dict, fields, problems: list) -> None:
    """One node's problems: an unknown class, or a model file it doesn't list (in `fields`)."""
    c = node["class_type"]
    if c not in oi:
        problems.append(f"{nid}: node class {c!r} not installed")
        return
    spec = {**oi[c]["input"].get("required", {}), **oi[c]["input"].get("optional", {})}
    for f in fields:
        v = node["inputs"].get(f)
        if isinstance(v, str) and f in spec:
            opts = spec[f][0]
            opts = opts if isinstance(opts, list) and opts != ["COMBO"] else spec[f][1].get("options", [])
            if v not in opts:
                problems.append(f"{nid}: {f} {v!r} not in the server's list")


def check(graph: dict, server: str, fields=NAME_FIELDS, oi: dict = None) -> list:
    """Problems the server would reject: unknown node classes, model files it doesn't list (in `fields`). Pass `oi` to reuse one
    /object_info fetch across many graphs."""
    oi = oi or json.loads(_call(server, "/object_info"))
    problems = []
    for nid, node in graph.items():
        _check_node(nid, node, oi, fields, problems)
    return problems


def upload(server: str, path: Path, name: str = None) -> str:
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name or path.name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + path.read_bytes() + (
        f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n").encode()
    return json.loads(_call(server, "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"}))["name"]


def cancel(server: str, pid: str) -> None:
    """Drop prompt `pid` from the queue, or interrupt it if it's the one running, so a timed-out stage doesn't hold up the rest."""
    _call(server, "/queue", json.dumps({"delete": [pid]}).encode(), JSON_HDR)
    _call(server, "/interrupt", json.dumps({"prompt_id": pid}).encode(), JSON_HDR)


def submit(server: str, graph: dict, label: str = "") -> str:
    """Queue a graph and return its prompt id. `label` (e.g. "fairy_con stage 3") shows in dashboard.py."""
    return json.loads(_call(server, "/prompt", json.dumps({"prompt": graph, "client_id": CLIENT_ID, "extra_data": {"label": label}}).encode(),
                            JSON_HDR))["prompt_id"]


def processing_seconds(hist: dict):
    """ComfyUI's own execution_start -> execution_success span for a finished job (queue waiting excluded), or None."""
    stamps = {m[0]: m[1]["timestamp"] for m in hist["status"].get("messages", []) if m[0] in ("execution_start", "execution_success")}
    if len(stamps) == 2:
        return (stamps["execution_success"] - stamps["execution_start"]) / 1000
    return None


def download(server: str, file: dict, dest: Path) -> None:
    """One output file of a /history entry ({filename, subfolder, type}) to dest."""
    q = urllib.parse.urlencode({"filename": file["filename"], "subfolder": file.get("subfolder", ""), "type": file.get("type", "output")})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_call(server, f"/view?{q}"))


def wait(server: str, pid: str, final_node: str, dest: Path, timeout: float = 1800.0):
    """Wait for a queued job, save final_node's first image to dest, and return its processing seconds (None if unknown)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = json.loads(_call(server, f"/history/{pid}")).get(pid)
        if hist and hist["status"].get("completed") is not None:
            if hist["status"]["status_str"] != "success":
                raise RuntimeError(f"local job failed: {json.dumps(hist['status'])[:800]}")
            download(server, hist["outputs"][final_node]["images"][0], dest)
            return processing_seconds(hist)
        time.sleep(2)
    raise TimeoutError(f"local job {pid} did not finish in {timeout:.0f}s")


class LocalTarget(Target):
    name = "local"
    lane = "local"

    def __init__(self, server: str = None):
        self.server = (server or config.LOCAL_COMFY_URL).rstrip("/")

    @property
    def params(self):
        m = local_models()
        return {"seedvr_model": m["seedvr_model"], "seedvr_color": m["seedvr_color"]}

    def _unreachable(self, e, what="can't reach"):
        return TargetUnavailable(f"{what} the local ComfyUI at {self.server} ({getattr(e, 'reason', e)}) -- start it, or pass --server")

    def available(self) -> bool:
        try:
            _call(self.server, "/system_stats")
            return True
        except (OSError, ValueError):
            return False

    def prepare(self, graph: dict) -> dict:
        return localize(graph)

    def check(self, graph: dict) -> list:
        try:
            return check(graph, self.server)
        except (urllib.error.URLError, ConnectionError) as e:
            raise self._unreachable(e)

    def upload(self, path: Path, name: str = None) -> str:
        try:
            return upload(self.server, path, name)
        except (urllib.error.URLError, ConnectionError) as e:
            raise self._unreachable(e)

    def before_batch(self, jobs: list) -> bool:
        try:
            freed = free_if_changed(self.server, jobs[0].graph)
        except (urllib.error.URLError, ConnectionError) as e:
            raise self._unreachable(e)
        if not freed:
            print("  same models as the last run -- keeping them loaded", flush=True)
        jobs[0].cold = freed  # the first job after free() also loads the models
        return freed

    def same_as_last(self, graph: dict) -> bool:
        return same_as_last(self.server, graph)

    def submit(self, job: StageJob) -> None:
        job.debug_dir.mkdir(parents=True, exist_ok=True)
        (job.debug_dir / "graph.json").write_text(json.dumps(job.graph, indent=2))
        try:
            job.handle = submit(self.server, job.graph, job.label)
        except (urllib.error.URLError, ConnectionError) as e:
            raise self._unreachable(e)
        eta = timings.expected(timings.sig(job.graph), job.cold)
        print(f"  [{job.debug_dir.name}] queued" + (f" expected {eta:.0f}s" if eta else ""), flush=True)

    def collect(self, jobs: list):
        """ComfyUI runs its queue in order, so wait for the jobs in order."""
        for job in jobs:
            t = time.time()
            final = job.debug_dir / "final.png"
            try:
                job.seconds = wait(self.server, job.handle, job.final_node, final)
                if job.seconds:
                    timings.record(timings.sig(job.graph), job.seconds, job.cold)
                job.final = final
            except (urllib.error.URLError, ConnectionError) as e:
                raise self._unreachable(e, "lost") from e
            except TimeoutError as e:
                cancel(self.server, job.handle)
                job.error = f"{e}; cancelled it on the server"
            except RuntimeError as e:
                job.error = str(e)
            job.waited = time.time() - t
            yield job
