"""The LoRA library, `loras.json` (data/library, else the defaults/ starter; config.library_file): every image and i2v LoRA with its files, triggers and notes, in one file
the web UI's LoRAs page edits. Read fresh on every call (a few KB), so a UI edit applies to the next job without a restart.

    {"image": {name: {"krea": file, "il": file, "krea_trigger": str, "il_trigger": str, "default": n, ...}},
     "video": {name: {"file": file, ...}},
     "combos": {name: {"kind": "image"|"video", "loras": {lora name: strength}, "note": str, "tips": str}},
     "ignore": [local file names the LoRAs page shouldn't offer to register]}

Shared fields: `category` (concept | style | character; groups the UI), `suggested` (strength the UI adds it at), `cloud` (uploaded to Comfy Cloud: false = local only, a cloud run
refuses it), `note` (one line), `tips` (plain text: prompt phrasing, tested strengths), `url` (source page), `min` / `max` = the
strength range of a slider LoRA (strength is the dial, often negative; the stage editor shows a drag slider over it).
Image: `krea` / `il` / `moss` = the file for the Krea2 / Illustrious pass / the MOSS-Image engine (at least one; a MOSS
file is local only and keeps its `moss/` folder), `*_trigger` = words appended to that pass's prompt while active, `default` = strength when a stage doesn't mention it (none set at the moment). Video: `file`,
`engine` = the i2v model it was trained for (VIDEO_ENGINES; absent = minimax), `trigger` = words put in front of the motion
prompt while the LoRA is on (video_triggers()).
A combo is a tested set of LoRAs with their strengths (e.g. three i2v LoRAs that work well together on one motion). It is a preset:
the stage editor copies its LoRAs into the stage's `loras` / `video_loras`, so the pipeline never sees combos and a stage can
still tune one strength afterwards. Members must be entries of that kind (video ones all for the same i2v model).
Files are bare cloud names; the local target maps them to this machine's folders. Entry order = chain order for
default-on LoRAs, so saves keep it.
"""
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: config.py
import config  # noqa: E402

KINDS = ("image", "video")
COMMON = {"category", "suggested", "cloud", "disabled", "note", "tips", "url", "min", "max"}
CATEGORIES = ("concept", "style", "character")
IMAGE_PASSES = ("krea", "il", "moss")  # image LoRA file fields, one per model family (image_loras.ENGINE_FIELDS)
FIELDS = {"image": COMMON | set(IMAGE_PASSES) | {f"{p}_trigger" for p in IMAGE_PASSES} | {"default"}, "video": COMMON | {"file", "engine", "trigger"}}
VIDEO_ENGINES = ("minimax", "ltx")  # = video/animate_pipeline.py VIDEO_ENGINES (tests/test_loras.py checks)
NAME = re.compile(r"^[a-z0-9_]+$")


def load() -> dict:
    f = config.LORAS_FILE if config.LORAS_FILE.is_file() else config.DEFAULTS_DIR / config.LORAS_FILE.name
    data = json.loads(f.read_text())
    return {"image": data.get("image", {}), "video": data.get("video", {}), "combos": data.get("combos", {}), "ignore": data.get("ignore", [])}


def _validate_files(label: str, kind: str, spec: dict) -> None:
    """An entry's file fields: at least one, each a bare .safetensors name."""
    files = [spec.get(k) for k in (IMAGE_PASSES if kind == "image" else ("file",)) if spec.get(k) is not None]
    if not files:
        raise ValueError(f"{label}: needs a file" + (" (krea, il and/or moss)" if kind == "image" else ""))
    for f in files:
        if not isinstance(f, str) or not f.endswith(".safetensors"):
            raise ValueError(f"{label}: file {f!r} must be a .safetensors name")


def _validate_numbers(label: str, spec: dict) -> None:
    """suggested/default/min/max: finite numbers; a slider needs both min and max, min below max."""
    for k in ("suggested", "default", "min", "max"):
        v = spec.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)):
            raise ValueError(f"{label}: {k} must be a number, got {v!r}")
    if ("min" in spec) != ("max" in spec) or ("min" in spec and spec["min"] >= spec["max"]):
        raise ValueError(f"{label}: a slider needs both min and max, min below max")


def _validate_entry(kind: str, name, spec) -> None:
    """One image/video LoRA entry."""
    label = f"{kind} LoRA {name!r}"
    if not NAME.match(name):
        raise ValueError(f"{label}: names are lowercase letters, digits and _")
    if not isinstance(spec, dict):
        raise ValueError(f"{label}: must be an object")
    extra = set(spec) - FIELDS[kind]
    if extra:
        raise ValueError(f"{label}: unknown field(s) {', '.join(sorted(extra))}")
    _validate_files(label, kind, spec)
    _validate_numbers(label, spec)
    if spec.get("category", "concept") not in CATEGORIES:
        raise ValueError(f"{label}: category must be one of {', '.join(CATEGORIES)}")
    if kind == "video" and spec.get("engine", "minimax") not in VIDEO_ENGINES:
        raise ValueError(f"{label}: engine must be one of {', '.join(VIDEO_ENGINES)}")
    for k in ("cloud", "disabled"):
        if not isinstance(spec.get(k, False), bool):
            raise ValueError(f"{label}: {k} must be true or false")
    for k in ("note", "tips", "url", "krea_trigger", "il_trigger", "moss_trigger", "trigger"):
        if not isinstance(spec.get(k, ""), str):
            raise ValueError(f"{label}: {k} must be text")


def _validate_combo_loras(label: str, kind: str, loras, data: dict) -> None:
    """A combo's loras map: names in the library, numeric finite strengths, and one i2v engine for a video combo."""
    if not isinstance(loras, dict) or not loras:
        raise ValueError(f"{label}: loras must be a non-empty object like {{\"name\": 0.7}}")
    for n, v in loras.items():
        if n not in data[kind]:
            raise ValueError(f"{label}: no {kind} LoRA {n!r} in the library")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{label}: strength for {n!r} must be a number, got {v!r}")
    if kind == "video" and len({data["video"][n].get("engine", "minimax") for n in loras}) > 1:
        raise ValueError(f"{label}: its video LoRAs are for different i2v models")


def _validate_combo(name, c, data: dict) -> None:
    label = f"combo {name!r}"
    if not NAME.match(name):
        raise ValueError(f"{label}: names are lowercase letters, digits and _")
    if not isinstance(c, dict) or set(c) - {"kind", "loras", "note", "tips"}:
        raise ValueError(f"{label}: an object with kind, loras, note, tips")
    if c.get("kind") not in KINDS:
        raise ValueError(f"{label}: kind must be image or video")
    _validate_combo_loras(label, c["kind"], c.get("loras"), data)
    for k in ("note", "tips"):
        if not isinstance(c.get(k, ""), str):
            raise ValueError(f"{label}: {k} must be text")


def validate(data: dict) -> None:
    """Raises ValueError naming the first bad entry."""
    for kind in KINDS:
        for name, spec in data[kind].items():
            _validate_entry(kind, name, spec)
    for name, c in data["combos"].items():
        _validate_combo(name, c, data)
    if not all(isinstance(f, str) for f in data["ignore"]):
        raise ValueError("ignore must be a list of file names")


def save(data: dict) -> None:
    validate(data)
    config.save_library(config.LORAS_FILE, data)


def image() -> dict:
    return load()["image"]


def video_files() -> dict:
    """{name: file} of the i2v LoRAs."""
    return {n: s["file"] for n, s in load()["video"].items()}


def with_video_triggers(motion: str, loras: dict) -> str:
    """The motion prompt with the triggers of its active i2v LoRAs in front (once: a trigger already in the text isn't repeated)."""
    lib = load()["video"]
    missing = [t for n, v in (loras or {}).items() if v and (t := lib.get(n, {}).get("trigger")) and t not in motion]
    return f"{', '.join(missing)}\n\n{motion}" if missing else motion


def video_engines() -> dict:
    """{name: engine} of the i2v LoRAs."""
    return {n: s.get("engine", "minimax") for n, s in load()["video"].items()}


def local_only_files() -> set:
    """Files of entries marked `cloud: false` (both kinds)."""
    data = load()
    return {s[k] for kind, keys in (("image", IMAGE_PASSES), ("video", ("file",)))
            for s in data[kind].values() if s.get("cloud") is False for k in keys if s.get(k)}


def disabled(kind: str) -> set:
    """Names of the `kind` entries switched off on the LoRAs page: retired, never picked for a stage (stages already using
    one still render; the dry-run and `project.py show` flag them)."""
    return {n for n, s in load()[kind].items() if s.get("disabled")}


def disabled_in(loras: dict, kind: str) -> list:
    """The switched-off library LoRAs a stage's `loras` / `video_loras` still has on (strength 0 = off, not counted)."""
    off = disabled(kind)
    return [n for n, v in (loras or {}).items() if v and n in off]


def _local_listing():
    """The local ComfyUI's loras folder as a set of paths, or None while it is down."""
    import urllib.request
    try:
        with urllib.request.urlopen(config.LOCAL_COMFY_URL + "/models/loras", timeout=2) as r:
            return set(json.loads(r.read()))
    except Exception:  # noqa: BLE001
        return None


def _where(kind: str, spec: dict, local) -> str:
    """'cloud+local', 'local only', ... with a MISSING mark when the local ComfyUI is up and lacks an image file."""
    where = "local only" if spec.get("cloud") is False else ("cloud" if kind == "video" else "cloud+local")
    if kind == "image" and local is not None:
        from targets.models import local_lora
        lacking = [k for k in IMAGE_PASSES if spec.get(k) and local_lora(spec[k], krea=k == "krea") not in local]
        if lacking:
            where += f" (MISSING locally: {', '.join(lacking)})"
    return where


def catalog() -> str:
    """The menu to pick from: every switched-on LoRA per kind with strength, where it runs, note and tips; the combos whose
    members are all on; the voices that are on; then the names that are off (don't use them)."""
    import voice_registry
    data, local = load(), _local_listing()
    out = [f"# Available LoRAs (loras.json; local ComfyUI {'down: local files unchecked' if local is None else 'up: files checked'})"]
    for kind, field in (("image", "loras"), ("video", "video_loras")):
        out.append(f"\n## {kind} -> stage `{field}`")
        for n, s in data[kind].items():
            if s.get("disabled"):
                continue
            dial = f"{s['min']}..{s['max']}" if "min" in s else f"{s.get('suggested', 1.0)}"
            eng = f", {s['engine']}" if s.get("engine", "minimax") != "minimax" else ""
            out.append(f"- {n} [{s.get('category', 'concept')}{eng}] {dial}, {_where(kind, s, local)}: {s.get('note', '')}")
            if s.get("tips"):
                out.append("    tips: " + " ".join(s["tips"].split()))
    on = {k: set(data[k]) - disabled(k) for k in KINDS}
    combos = [f"- {n} ({c['kind']}): {', '.join(f'{m} {v}' for m, v in c['loras'].items())}. {c.get('note', '')}"
              for n, c in data["combos"].items() if set(c["loras"]) <= on[c["kind"]]]
    out += ["\n## combos (presets: copy their LoRAs into the stage)"] + (combos or ["- none with every member on"])
    voices = voice_registry.load()["voices"]
    out.append("\n## voices -> stage/project `voice`: " + ", ".join(n for n, v in voices.items() if not v.get("disabled")))
    off = {k: sorted(disabled(k)) for k in KINDS}
    out.append(f"\n## switched off (never pick): image {', '.join(off['image']) or '-'}; video {', '.join(off['video']) or '-'}; "
               f"voices {', '.join(sorted(n for n, v in voices.items() if v.get('disabled'))) or '-'}")
    return "\n".join(out)


if __name__ == "__main__":
    print(catalog())
