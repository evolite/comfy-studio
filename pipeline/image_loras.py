#!/usr/bin/env python3
"""
Image-pipeline LoRAs, the same model as video's `video_loras`: templates carry no LoRA nodes; a stage's
`loras` ({"name": strength}, decimals ok) is the only switch, and `apply()` chains one LoraLoaderModelOnly
per active entry (dict order) onto the graph at build time.

Registry: loras.json (see lora_registry.py): `krea` / `il` = LoRA file for the Krea2 pass / the Illustrious pass,
`*_trigger` = words appended to that pass's scene text while the LoRA is active,
`default` = strength used when a stage doesn't mention it (`{"name": 0}` forces it off).
`moss` = the file for the MOSS-Image engine (another model family: its LoRAs chain off the same 30:10 base node).
ENGINE_FIELDS = which of those files each image engine loads (the first: the base pass off 30:10; `il`: the Illustrious pass,
only in `full`). The web UI's stage editor offers only LoRAs that fit.

Node ids are hand-traced from the templates (see AGENTS.md "Node-ID fragility").
"""
import math
import sys

import lora_registry


def __getattr__(name):
    """`IMAGE_LORAS` = the image half of loras.json ({name: spec}), read fresh so a web-UI edit applies without a restart."""
    if name == "IMAGE_LORAS":
        return lora_registry.image()
    raise AttributeError(name)


ENGINE_FIELDS = {"full": ("krea", "il"), "krea2": ("krea",), "bgswap": ("krea",), "moss": ("moss",)}  # = run_pipeline.ENGINES

KREA_BASE = "30:10"                # UNETLoader; chain tail feeds the Krea2 KSampler
KREA_CONSUMERS = ["30:3"]
IL_BASE = "433:368"                # Illustrious checkpoint (full engine); chain tail feeds these
IL_CONSUMERS = ["433:366", "434:392", "434:421"]  # IL KSampler, upscale sampler, FaceDetailer


def check_loras(loras, label: str) -> dict:
    if not isinstance(loras, dict):
        raise SystemExit(f"{label}: loras must be an object like {{\"background_detail\": 0.6}}, got {loras!r}")
    known = lora_registry.image()
    for name, strength in loras.items():
        if name not in known:
            raise SystemExit(f"{label}: unknown LoRA {name!r}; known: {', '.join(known)}")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength):
            raise SystemExit(f"{label}: strength for {name!r} must be a number (decimals ok), got {strength!r}")
    off = lora_registry.disabled_in(loras, "image")
    if off:
        print(f"  warning: {label}: {', '.join(off)} switched off on the LoRAs page (still applied; pick another)", file=sys.stderr)
    return loras


def resolve(loras: dict = None) -> dict:
    """Registry defaults overlaid with the stage's explicit strengths."""
    resolved = {n: s["default"] for n, s in lora_registry.image().items() if "default" in s}
    resolved.update(check_loras(loras or {}, "loras"))
    return resolved


def _chain(graph: dict, base: str, consumers: list, entries: list, prefix: str) -> None:
    prev = [base, 0]
    for k, (name, file, strength) in enumerate(entries):
        node_id = f"{prefix}:{300 + k}"
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": file, "strength_model": strength, "model": prev},
            "_meta": {"title": f"Load LoRA ({name})"},
        }
        prev = [node_id, 0]
    for node in consumers:
        graph[node]["inputs"]["model"] = prev


def _il_entries(loras: dict, il_scene: str):
    """The active LoRAs that have an Illustrious file as [(name, file, strength)], plus il_scene with their triggers appended."""
    entries = []
    specs = lora_registry.image()
    for name, strength in resolve(loras).items():
        spec = specs[name]
        if strength != 0 and "il" in spec:
            entries.append((name, spec["il"], strength))
            trig = spec.get("il_trigger")
            if trig and trig.split(",")[0] not in il_scene:
                il_scene = f"{il_scene} {trig},"
    return entries, il_scene


def apply_il(graph: dict, base: str, consumers: list, loras: dict, il_scene: str, prefix: str) -> str:
    """Chain the active Illustrious LoRAs from checkpoint node `base` onto `consumers` (ids `prefix:300+`); returns il_scene with triggers."""
    entries, il_scene = _il_entries(loras, il_scene)
    _chain(graph, base, consumers, entries, prefix)
    return il_scene


def apply(graph: dict, engine: str, loras: dict, krea_scene: str, il_scene: str):
    """Chain the active LoRAs onto the graph; returns (krea_scene, il_scene) with triggers appended.
    engine: a key of ENGINE_FIELDS (which passes' files it loads)."""
    fields = ENGINE_FIELDS[engine]
    base = fields[0]  # the pass whose files chain off 30:10
    krea = []
    specs = lora_registry.image()
    for name, strength in resolve(loras).items():
        spec = specs[name]
        if strength != 0 and base in spec:
            krea.append((name, spec[base], strength))
            trig = spec.get(f"{base}_trigger")
            if trig and trig.split(",")[0] not in krea_scene:
                krea_scene = f"{krea_scene} {trig}."
    _chain(graph, KREA_BASE, KREA_CONSUMERS, krea, "30")
    if "il" in fields:
        il_scene = apply_il(graph, IL_BASE, IL_CONSUMERS, loras, il_scene, "433")
    return krea_scene, il_scene
