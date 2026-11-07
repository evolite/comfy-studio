"""The LoRAs page: the library (loras.json via lora_registry) + where each file is available + which stages use it, the local
ComfyUI's LoRA files nothing registers yet ("discovered"), the combos (named LoRA sets), and the edits (create/update, delete (taking it
off any stages and combos using it), ignore a file, put/delete a combo)."""
import json
import re
import threading
import urllib.request

import projects as P  # first: puts the repo root and pipeline/ on sys.path
import config  # noqa: E402
import lora_registry  # noqa: E402
from targets.models import local_lora  # noqa: E402

PASSES = {"image": lora_registry.IMAGE_PASSES, "video": ("file",)}
_lock = threading.Lock()


def local_files():
    """The local ComfyUI's loras folder as it lists it (sub/folder/name.safetensors), or None while it is down."""
    try:
        with urllib.request.urlopen(config.LOCAL_COMFY_URL + "/models/loras", timeout=2) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def _usage_of(pj, out) -> None:
    """One project's stage LoRA usages into `out` (strength 0 = off, not counted)."""
    try:
        stages = json.loads(pj.read_text()).get("stages", [])
    except (OSError, ValueError):
        return
    for pos, s in enumerate(stages, 1):
        for kind, field in (("image", "loras"), ("video", "video_loras")):
            for name, v in (s.get(field) or {}).items():
                if v:
                    out[kind].setdefault(name, []).append({"project": pj.parent.name, "stage": s.get("id", pos)})


def usage() -> dict:
    """{"image"|"video": {name: [{"project", "stage"}]}} over every project's stages (strength 0 = off, not counted)."""
    out = {"image": {}, "video": {}}
    for pj in sorted(config.PIPELINES_DIR.glob("*/project.json")):
        _usage_of(pj, out)
    return out


def _local_name(kind, key, file):
    return local_lora(file, krea=key == "krea") if kind == "image" else file


def _guess(path: str) -> dict:
    """Register-form prefill for a discovered local file: name slug, pass from the folder (None = ask), trigger from a
    `-trigger-xxx` part, and the file as the entry stores it (bare when it sits in that pass's default folder)."""
    folder, _, base = path.rpartition("/")
    stem = base.removesuffix(".safetensors")
    if folder in ("krea", "moss"):
        pass_ = folder
    elif folder.lower() == "il":
        pass_ = "il"
    else:
        pass_ = None
    trig = re.search(r"-trigger-([A-Za-z0-9]+)", stem)
    name = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")[:40] or "lora"
    files = {k: (base if local_lora(base, krea=k == "krea") == path else path) for k in ("krea", "il")} | {"moss": path}
    return {"path": path, "name": name, "pass": pass_, "files": files, "trigger": trig.group(1) if trig else None}


def _entry(kind, name, spec, local, used, referenced) -> dict:
    """One library entry: its spec plus per-pass local availability (i2v only runs on the cloud: no local check) and usages."""
    avail = {}
    for key in PASSES[kind]:
        if spec.get(key):
            path = _local_name(kind, key, spec[key])
            referenced.add(path)
            if kind == "image":
                avail[key] = None if local is None else path in local
    return {**spec, "local": avail, "used": used[kind].get(name, [])}


def library() -> dict:
    data = lora_registry.load()
    local = local_files()
    used = usage()
    referenced = set()
    entries = {kind: {name: _entry(kind, name, spec, local, used, referenced) for name, spec in data[kind].items()}
               for kind in ("image", "video")}
    ignore = set(data["ignore"])
    discovered = [_guess(f) for f in (local or []) if f not in referenced and f not in ignore]
    literal = sorted(n for n in used["video"] if n.endswith(".safetensors"))  # literal video_loras keys: offer to register
    return {**entries, "combos": data["combos"], "ignore": data["ignore"], "discovered": discovered, "literal_video": literal, "local_up": local is not None}


def put(kind: str, name: str, spec: dict) -> dict:
    if kind not in PASSES:
        raise P.ApiError(f"kind must be image or video, got {kind!r}")
    spec = {k: v for k, v in spec.items() if v not in (None, "")}  # blank form fields = unset
    with _lock:
        data = lora_registry.load()
        data[kind][name] = spec  # existing names keep their place: entry order is the default-on chain order
        try:
            lora_registry.save(data)
        except ValueError as e:
            raise P.ApiError(str(e))
    return spec


def _strip_from(project: str, field: str, name: str) -> None:
    """Take `name` out of `field` on every stage of one project (project.json.bak kept)."""
    with P._edit_lock:
        path = config.PIPELINES_DIR / project / "project.json"
        cfg = json.loads(path.read_text())
        for s in cfg.get("stages", []):
            if name in (s.get(field) or {}):
                del s[field][name]
        P._write_json(path, cfg)


def delete(kind: str, name: str, strip: bool = False) -> list:
    """Remove a LoRA from the library. In use: 409, unless `strip`, which first takes it off every stage using it (project.json
    .bak kept). Returns the stages it was taken off."""
    uses = usage().get(kind, {}).get(name) or []
    if uses and not strip:
        where = ", ".join(f"{u['project']} #{u['stage']}" for u in uses[:5]) + (" ..." if len(uses) > 5 else "")
        raise P.ApiError(f"{name} is used by {len(uses)} stage(s): {where}", 409)
    field = "loras" if kind == "image" else "video_loras"
    for project in sorted({u["project"] for u in uses}):
        _strip_from(project, field, name)
    with _lock:
        data = lora_registry.load()
        if data.get(kind, {}).pop(name, None) is None:
            raise P.ApiError(f"no {kind} LoRA {name!r}", 404)
        for cname, c in sorted(data["combos"].items()):  # off its combos too; one left empty goes
            if c.get("kind") == kind and c["loras"].pop(name, None) is not None and not c["loras"]:
                del data["combos"][cname]
        lora_registry.save(data)
    return uses


def ignore(file: str, on: bool) -> list:
    with _lock:
        data = lora_registry.load()
        data["ignore"] = [f for f in data["ignore"] if f != file] + ([file] if on else [])
        lora_registry.save(data)
    return data["ignore"]


def put_combo(name: str, spec: dict) -> dict:
    spec = {k: v for k, v in spec.items() if v not in (None, "")}
    with _lock:
        data = lora_registry.load()
        data["combos"][name] = spec
        try:
            lora_registry.save(data)
        except ValueError as e:
            raise P.ApiError(str(e))
    return spec


def delete_combo(name: str) -> None:
    """Stages that applied it keep their LoRAs: a combo is only a preset."""
    with _lock:
        data = lora_registry.load()
        if data["combos"].pop(name, None) is None:
            raise P.ApiError(f"no combo {name!r}", 404)
        lora_registry.save(data)
