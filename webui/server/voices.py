"""The Voices page: the voice library (voices.json via voice_registry), which stages use each voice, whether its sample clip
exists, and the edits (create/update, delete, new entry from a stage, make the voice's sample stage)."""
import shutil
import threading

import projects as P  # first: puts the repo root and pipeline/ on sys.path
import config  # noqa: E402
import voice_registry  # noqa: E402

_lock = threading.Lock()


def _sample_path(sample):
    if not sample:
        return None
    d = config.PIPELINES_DIR / sample["project"]
    f = (d / sample["folder"] if sample.get("folder") else d) / "videos" / f"stage_{sample['stage']}.mp4"
    return f if f.is_file() else None


def library() -> dict:
    used = voice_registry.usage()
    out = {}
    for name, spec in voice_registry.load()["voices"].items():
        f = _sample_path(spec.get("sample"))
        out[name] = {**spec, "used": used.get(name, []), "sample_mtime": int(f.stat().st_mtime) if f else None}
    return {"voices": out, "slots": list(voice_registry.SLOTS)}


def put(name: str, spec: dict) -> dict:
    spec = {k: v for k, v in spec.items() if v not in (None, "", [])}  # blank form fields = unset
    with _lock:
        data = voice_registry.load()
        data["voices"][name] = spec
        try:
            voice_registry.save(data)
        except ValueError as e:
            raise P.ApiError(str(e)) from e
    return spec


def delete(name: str) -> None:
    """Remove a voice; refused (409) while a stage or project uses it."""
    uses = voice_registry.usage().get(name) or []
    if uses:
        where = ", ".join(f"{u['project']} #{u['stage']}" for u in uses[:5]) + (" ..." if len(uses) > 5 else "")
        raise P.ApiError(f"{name} is used by {len(uses)} stage(s): {where}", 409)
    with _lock:
        data = voice_registry.load()
        if data["voices"].pop(name, None) is None:
            raise P.ApiError(f"no voice {name!r}", 404)
        voice_registry.save(data)


def from_stage(project: str, stage: int, name: str, folder: str = "") -> dict:
    """A new library entry whose sample is this stage's clip; its slots start from the stage's current voice (if any), to be
    rewritten for the new voice."""
    cfg = P.load(project)
    s = next((x for x in cfg["stages"] if x.get("id") == stage), None)
    if s is None:
        raise P.ApiError(f"no stage {stage}", 404)
    base = voice_registry.load()["voices"].get(voice_registry.voice_of(s, cfg) or "", {})
    spec = {k: base[k] for k in voice_registry.SLOTS if base.get(k)} or {"voice": "describe the voice"}
    spec["sample"] = {"project": project, "stage": stage, "folder": folder}
    with _lock:
        data = voice_registry.load()
        if name in data["voices"]:
            raise P.ApiError(f"there is already a voice {name!r}", 409)
    return put(name, spec)


@P.locked
def make_sample(name: str) -> dict:
    """The voice's stage in the samples project (config.VOICE_SAMPLES_PROJECT), created on first use: a copy of the project's
    first stage (its still, and its motion with the {voice} slots and the sample line) with `voice` = this voice. Every call
    re-copies the first stage, so editing its line or still and re-making a sample picks the change up. The library entry's
    `sample` then points at the stage's Animate smooth clip. Returns where to animate it; animating is the caller's (paid) step."""
    if name not in voice_registry.load()["voices"]:
        raise P.ApiError(f"no voice {name!r}", 404)
    project = config.VOICE_SAMPLES_PROJECT
    d = P.project_dir(project)
    cfg = P.load(project)
    base = cfg["stages"][0]
    still = d / "preview" / f"stage_{base['id']}.png"
    if not still.is_file():
        raise P.ApiError(f"{project} has no preview of stage {base['id']} yet: render the base still first")
    stage = next((s for s in cfg["stages"] if s.get("voice") == name), None)
    if stage is None:
        stage = {"id": max(s["id"] for s in cfg["stages"]) + 1}
        cfg["stages"].append(stage)
    if stage["id"] != base["id"]:
        stage.update({k: v for k, v in base.items() if k not in ("id", "voice", "voices", "done")})
        shutil.copyfile(still, d / "preview" / f"stage_{stage['id']}.png")
    stage["voice"] = name
    stage["done"] = ["preview"]
    P._write_json(d / P.PROJECT_JSON, cfg)
    with _lock:
        data = voice_registry.load()
        data["voices"][name]["sample"] = {"project": project, "stage": stage["id"], "folder": "interp"}
        voice_registry.save(data)
    return {"project": project, "stage": stage["id"], "folder": "preview"}
