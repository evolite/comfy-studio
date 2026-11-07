"""Civitai-ready copies of project stills: the A1111-style `parameters` text Civitai reads on upload (prompt, sampler, seed,
size, model + LoRA AutoV2 hashes, so it links the resources), built from the graph that made the image.

Where the graph comes from, per folder:
    preview  the take's <id>.graph.json (pipeline/takes.py); a take without one gets no metadata
    hires    the current preview take's graph (the base render) + `Hires upscaler: SesquiLSR`
    IL       the same + `Hires upscaler: <the ESRGAN model in IL/_debug/stage_N/graph.json>`
    finish   the same + `Hires upscaler: <its source's upscaler> + SeedVR2` (source: hires/, else IL/)
Hashes come from the files under config.COMFYUI_MODELS_DIR and are cached in logs/model_hashes.json; a model that isn't on this
machine keeps its name without a hash. Each hash is also looked up on Civitai once (logs/civitai_versions.json) for a
`Civitai resources` field naming the exact model versions: the upload links resources from it even where the hash/name match
misses one (it dropped Zaey Ultra while linking the others). The originals are never rewritten: tagged_png returns a copy.
"""
import hashlib
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: config.py
import config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import takes  # noqa: E402

SAMPLERS = {"euler": "Euler", "euler_ancestral": "Euler a", "heun": "Heun", "dpm_2": "DPM2", "dpmpp_2m": "DPM++ 2M",
            "dpmpp_2m_sde": "DPM++ 2M SDE", "dpmpp_sde": "DPM++ SDE", "dpmpp_3m_sde": "DPM++ 3M SDE", "lms": "LMS", "ddim": "DDIM",
            "uni_pc": "UniPC", "lcm": "LCM", "er_sde": "ER SDE", "res_multistep": "Res Multistep"}
MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".sft")
HIRES_UPSCALER = "Hires upscaler"
_index = None


def _models_index() -> dict:
    """basename -> path of every model file under the local ComfyUI models folder (built once per process)."""
    global _index
    if _index is None:
        _index = {}
        root = config.COMFYUI_MODELS_DIR
        for dirpath, _, files in os.walk(root, followlinks=True) if root.is_dir() else ():
            for f in files:
                if f.endswith(MODEL_EXT):
                    _index.setdefault(f, Path(dirpath) / f)
    return _index


def _cached(file: str, key: str, compute):
    """compute() memoized under key in logs/<file>; an exception from compute is passed on and nothing is cached."""
    path = config.LOGS_DIR / file
    try:
        cache = json.loads(path.read_text())
    except (OSError, ValueError):
        cache = {}
    if key not in cache:
        cache[key] = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1))
    return cache[key]


def model_hash(name: str):
    """AutoV2 (first 10 hex of the file's sha256, upper case) of a model named as in a graph (`krea/x.safetensors` or `x.safetensors`),
    or None when the file isn't on this machine. Cached by path + size + mtime."""
    path = _models_index().get(Path(name or "").name)
    if not path:
        return None
    st = path.stat()
    def sha():
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()[:10].upper()
    return _cached("model_hashes.json", f"{path}|{st.st_size}|{st.st_mtime_ns}", sha)


def _fetch_version(h: str):
    """Civitai model-version id for an AutoV2 hash; None if Civitai doesn't know it. Raises OSError when Civitai can't be reached."""
    req = urllib.request.Request(f"https://civitai.com/api/v1/model-versions/by-hash/{h}",
                                 headers={"User-Agent": "comfy-studio"})  # Civitai answers urllib's default agent with 403
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r).get("id")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def version_id(h: str):
    """Cached _fetch_version; an unreachable Civitai gives None without caching, so the next copy retries."""
    try:
        return _cached("civitai_versions.json", h, lambda: _fetch_version(h))
    except (OSError, ValueError):
        return None


def _same(a: Path, b: Path) -> bool:
    return a.is_file() and b.is_file() and a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()


def _preview_graph(d: Path, sid: int, take: str = None):
    """The graph that made the preview (its current take, or the run that made a take from before takes kept graphs)."""
    tid = take or takes.current_id(d, sid)
    p = takes.stage_dir(d, sid) / f"{tid}.graph.json" if tid else None
    if p and p.is_file():
        return json.loads(p.read_text()), {}
    dbg = d / "preview" / "_debug" / f"stage_{sid}"  # a take from before takes kept graphs: the run that made it
    if tid and _same(takes.stage_dir(d, sid) / f"{tid}.png", dbg / "final.png"):
        return json.loads((dbg / "graph.json").read_text()), {}
    return None, {}


def _upscale_extra(d: Path, folder: str, sid: int, g: dict):
    """The base render's graph plus the `Hires upscaler` field naming what the folder's pass ran through."""
    if folder == "hires":
        return g, {HIRES_UPSCALER: "SesquiLSR"}
    if folder == "IL":
        dbg = d / "IL" / "_debug" / f"stage_{sid}" / "graph.json"
        model = next((n["inputs"].get("model_name") for n in json.loads(dbg.read_text()).values()
                      if n.get("class_type") == "UpscaleModelLoader"), None) if dbg.is_file() else None
        return g, {HIRES_UPSCALER: Path(model).stem} if model else {}
    if folder == "finish":
        src = next((f for f in ("hires", "IL") if (d / f / f"stage_{sid}.png").is_file()), None)
        before = _graph_for(d, src, sid)[1].get(HIRES_UPSCALER) if src else None
        return g, {HIRES_UPSCALER: f"{before} + SeedVR2" if before else "SeedVR2"}
    return None, {}


def _graph_for(d: Path, folder: str, sid: int, take: str = None):
    """(graph, extra fields) describing <folder>/stage_N.png (or a preview take), or (None, {})."""
    if folder == "preview":
        return _preview_graph(d, sid, take)
    # an upscale is described by the base render (the current preview take) plus the upscale model it went through
    g, _ = _graph_for(d, "preview", sid)
    if not g or not (d / folder / f"stage_{sid}.png").is_file():
        return None, {}
    return _upscale_extra(d, folder, sid, g)


def _loras_of(g: dict) -> list:
    """[(name, file, strength)] of the graph's active LoRAs (the node title preferred, the file stem as fallback)."""
    out = []
    for n in g.values():
        if n.get("class_type") == "LoraLoaderModelOnly" and n["inputs"].get("strength_model"):
            title = n.get("_meta", {}).get("title", "")
            name = title[len("Load LoRA ("):-1] if title.startswith("Load LoRA (") else Path(n["inputs"]["lora_name"]).stem
            out.append((name, n["inputs"]["lora_name"], n["inputs"]["strength_model"]))
    return out


def params(d: Path, folder: str, sid: int, take: str = None):
    """The A1111 `parameters` text for one still, or None when nothing records how it was made."""
    g, extra = _graph_for(d, folder, sid, take)
    if not g:
        return None
    ks = g.get("30:3", {}).get("inputs", {})
    prompt = g.get("30:6", {}).get("inputs", {}).get("text", "")
    unet = g.get("30:10", {}).get("inputs", {}).get("unet_name", "")
    loras = _loras_of(g)
    fields = {"Steps": ks.get("steps"), "Sampler": SAMPLERS.get(ks.get("sampler_name"), ks.get("sampler_name")),
              "Schedule type": (ks.get("scheduler") or "").replace("_", " ").title() or None, "CFG scale": ks.get("cfg"),
              "Seed": ks.get("seed"), "Size": _size(d, folder, sid, take)}
    mh = model_hash(unet)
    if mh:
        fields["Model hash"] = mh
    fields["Model"] = Path(unet).stem or None
    lora_h = [(name, h, s) for name, file, s in loras if (h := model_hash(file))]
    if lora_h:
        fields["Lora hashes"] = '"' + ", ".join(f"{name}: {h}" for name, h, _ in lora_h) + '"'
    fields.update(extra)
    fields["Version"] = "ComfyUI"
    res = [{"type": "checkpoint", "modelVersionId": v} for v in [mh and version_id(mh)] if v]
    res += [{"type": "lora", "weight": s, "modelVersionId": v} for _, h, s in lora_h if (v := version_id(h))]
    if res:
        fields["Civitai resources"] = json.dumps(res, separators=(",", ":"))
    lora_tags = "".join(f" <lora:{name}:{s:g}>" for name, _, s in loras)
    body = ", ".join(f"{k}: {v:g}" if isinstance(v, float) else f"{k}: {v}" for k, v in fields.items() if v not in (None, ""))
    return f"{prompt}{lora_tags}\nNegative prompt: \n{body}"


def _size(d, folder, sid, take):
    from PIL import Image
    p = takes.stage_dir(d, sid) / f"{take}.png" if folder == "preview" and take else d / folder / f"stage_{sid}.png"
    with Image.open(p) as im:
        return f"{im.width}x{im.height}"


def tagged_png(path: Path, text: str) -> bytes:
    """A copy of the PNG with a `parameters` tEXt chunk (its other text chunks, e.g. ComfyUI's `prompt`, kept)."""
    from PIL import Image, PngImagePlugin
    with Image.open(path) as im:
        info = PngImagePlugin.PngInfo()
        for k, v in im.info.items():
            if isinstance(v, str) and k != "parameters":
                info.add_text(k, v)
        if text:
            info.add_text("parameters", text)
        buf = io.BytesIO()
        im.save(buf, "PNG", pnginfo=info)
        return buf.getvalue()


def clip_params(d: Path, folder: str, sid: int):
    """Civitai doesn't read generation data from video files, so a clip's parameters are only offered as text to paste: the motion
    prompt, LoRAs, seed and size from the render graph ComfyUI wrote into the mp4's `prompt` tag. None without one."""
    import subprocess
    mp4 = (d / folder if folder else d) / "videos" / f"stage_{sid}.mp4"
    if not mp4.is_file():
        return None
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames,r_frame_rate:format_tags",
                            "-of", "json", str(mp4)], capture_output=True, text=True)
    try:
        info = json.loads(probe.stdout)
        g = json.loads(info["format"]["tags"]["prompt"])
    except (ValueError, KeyError):
        return None
    st = (info.get("streams") or [{}])[0]
    i2v = next((n["inputs"] for n in g.values() if n.get("class_type") == "MiniMaxH3ImageToVideo"), {})
    seed = next((n["inputs"]["noise_seed"] for n in g.values() if "noise_seed" in n.get("inputs", {})), None)
    sampler = next((n["inputs"]["sampler_name"] for n in g.values() if "sampler_name" in n.get("inputs", {})), None)
    loras = "".join(f" <lora:{Path(n['inputs']['lora_name']).stem}:{n['inputs']['strength_model']:g}>" for n in g.values()
                    if n.get("class_type") == "LoraLoaderModelOnly" and n["inputs"].get("strength_model"))
    fps = st.get("r_frame_rate", "").split("/")[0]
    fields = {"Steps": 8, "Sampler": SAMPLERS.get(sampler, sampler), "Seed": seed,
              "Size": f"{st['width']}x{st['height']}" if st.get("width") else None, "Frames": st.get("nb_frames"), "FPS": fps or None,
              "Model": "MiniMax H3 image-to-video (turbo 8-step)", "Version": "ComfyUI"}
    body = ", ".join(f"{k}: {v}" for k, v in fields.items() if v not in (None, ""))
    return f"{i2v.get('prompt', '')}{loras}\nNegative prompt: \n{body}"
