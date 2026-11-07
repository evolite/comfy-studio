"""
Full engine: Krea2 (text2img, natural-language caption) -> HED/normal/canny
ControlNets from that render -> Illustrious (Danbooru tags) -> Impact Pack
upscale -> optional FaceDetailer -> sharpen -> film grain.

Only builds graphs; comfy_api.py submits them. Run via pipeline/run_pipeline.py
(stage sets) or pipeline/generate_one.py (single image).
"""
import json
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graph_settings
import image_loras

TEMPLATE_PATH = HERE / "templates" / "krea2_area_template.json"

# --- Node IDs in the exported API-format graph (Krea2 Area workflow) -------
# Traced by hand from the "Save (API Format)" export. If the workflow is
# re-saved from ComfyUI with structural changes, these may shift — re-derive
# with the same node-tracing approach (grep class_type/_meta.title in the
# exported JSON) if a patch silently does nothing.
NODE_KREA_PROMPT = "30:6"          # CLIPTextEncode.text (whole Krea2 prompt: character + scene, written out per stage)
NODE_IL_PROMPT = "433:372"         # CLIPTextEncode.text (whole Illustrious prompt: quality prefix + stage's il_scene)
NODE_IL_NEGATIVE = "433:403"       # CLIPTextEncode.text (Illustrious negative)
NODE_FACE_DETAILER_SWITCH = "434:418"  # ComfySwitchNode: switch=true -> FaceDetailer(434:421) output, false -> raw upscaled image (434:394), skipping FaceDetailer entirely
# LoRA node ids live in image_loras.py.
NODE_UPSCALE_SAMPLER = "434:392"  # PixelKSampleUpscalerProvider
SEED_NODES = [
    ("30:3", "seed"),        # Krea2 KSampler
    ("433:366", "seed"),     # Illustrious KSampler
    ("434:421", "seed"),     # FaceDetailer
]

# Friendly labels for output nodes, used only to sort/name downloaded files.
# "final" marks the node whose image is the finished deliverable.
NODE_LABELS = {
    "436": "final",
}
FINAL_IMAGE_NODE = "436"

# The settled full-engine look (the con pipeline's settings); `project.py new` writes these into project.json.
TUNED = {
    "il_denoise": 0.2, "il_steps": 32, "il_cfg": 3.5,
    "upscale_factor": 1, "upscale_model": "4x_NMKD-Siax_2d00k.pth", "upscale_steps": 36, "upscale_cfg": 3.5,
    "upscale_denoise": 0.1, "upscale_iterations": 2,
    "sharpen": 0.8, "grain_density": 0, "grain_intensity": 0,
}
# Where the intended default differs from the template's value; every other tunable defaults to the template
# (keys: pipeline/graph_settings.py FULL).
DEFAULTS = {
    "face_wildcard": "detailed anime face, clean lineart, thick outline, cel shading, anime coloring",
    **TUNED,
}

DEFAULT_IL_NEGATIVE = (
    "lowres, worst quality, bad anatomy, bad hands, extra fingers, extra limbs, "
    "deformed teeth, blurry, watermark, censored, 2girls"
)
QUALITY_PREFIX = (
    "masterpiece, best quality, amazing quality, absurdres, highres, "
    "extremely detailed, solo, lineart, outline, cel shading, anime coloring, scenery, "
)


def load_template() -> dict:
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def _rewire(graph: dict, old: str, new: "dict | str") -> None:
    """Repoint every input reading node `old` (any output slot) to new[slot] (a dict of slots, or a node id for all of them)."""
    for node in graph.values():
        for k, v in node["inputs"].items():
            if isinstance(v, list) and v[0] == old:
                node["inputs"][k] = new[v[1]] if isinstance(new, dict) else [new, v[1]]


def _prune_edges(graph: dict, edges: bool) -> dict:
    """The conditioning the samplers read, and the ControlNet nodes kept: the HED/normal/canny chain's when `edges` is on,
    the plain IL prompts' otherwise (those ControlNets dropped)."""
    if edges:
        return {0: ["433:369:356", 0], 1: ["433:369:356", 1]}
    cond = {0: ["433:372", 0], 1: ["433:403", 0]}
    for nid in [n for n in graph if n.startswith("433:369:") and n != "433:369:371"]:
        del graph[nid]
    del graph["433:369:371"]
    return cond


def _prune_face(graph: dict) -> None:
    """FaceDetailer off: the switch becomes a passthrough and its detector/SAM loaders go."""
    _rewire(graph, "434:418", "434:394")
    for nid in ("434:418", "434:421", "434:419", "434:420"):
        del graph[nid]


def _plain_upscale(graph: dict, no_model: bool) -> None:
    """IL result -> (ESRGAN ->) lanczos to upscale_factor; no second IL sampling (the Impact upscaler divides by denoise,
    so 0 can't go through it)."""
    factor = graph["434:393"]["inputs"]["upscale_factor"]
    if no_model:
        graph["434:394"] = {"class_type": "ImageScaleBy", "inputs": {"image": ["433:367", 0], "upscale_method": "lanczos", "scale_by": factor}}
    else:
        graph["434:390"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["434:397", 0], "image": ["433:367", 0]}}
        graph["434:394"] = {"class_type": "ImageScaleBy", "inputs": {"image": ["434:390", 0], "upscale_method": "lanczos", "scale_by": factor / 4}}
    del graph["434:393"], graph["434:392"]


def _drop_upscale_model(graph: dict) -> None:
    """The Impact upscaler's optional model input off: the hop becomes a plain lanczos resize + the IL resample."""
    if "434:392" in graph:
        graph["434:392"]["inputs"].pop("upscale_model_opt", None)
    del graph["434:397"]


def _prune(graph: dict, edges: bool, plain_upscale: bool = False, upscale_only: bool = False, face: bool = False, no_model: bool = False) -> None:
    """Drop what a run doesn't need: the Krea2 preview save + compare nodes, the unused IL decode (the IL passes read Krea2's VAEDecode
    directly), the HED/normal/canny ControlNets when
    edge_controlnet_strength is 0 (samplers then read the plain IL conditioning). upscale_only: no base IL pass --
    the ESRGAN upscale + IL resample (434:393/434:392) runs straight on Krea2's encoded image."""
    _rewire(graph, "29", "30:8")
    for nid in ("29", "435", "434:422") + (() if plain_upscale else ("433:367",)):  # 434:422 compares against FaceDetailer, so left in it would force that pass even when off
        del graph[nid]
    cond = _prune_edges(graph, edges)
    for nid in ("433:366", "434:392", "434:421"):
        graph[nid]["inputs"]["positive"], graph[nid]["inputs"]["negative"] = cond[0], cond[1]
    if not face:
        _prune_face(graph)
    if upscale_only:
        graph["434:393"]["inputs"]["samples"] = ["433:440", 0]
        for nid in ("433:366", "433:367"):
            graph.pop(nid, None)
    if plain_upscale:
        _plain_upscale(graph, no_model)
    if no_model:
        _drop_upscale_model(graph)


def build_graph(
    krea_scene: str,
    il_scene: str,
    il_negative: str = None,
    seed: int = None,
    prepend_quality_prefix: bool = True,
    loras: dict = None,
    face_detailer_enabled: bool = False,
    **settings,
) -> dict:
    """Patch the template with prompts, seeds and settings. Extra keyword args are the `settings` keys in
    graph_settings.FULL (upscale_factor, il_steps, upscale_model, ...); unknown ones are ignored, so any
    stage JSON field can be passed.

    krea_scene / il_scene: the complete prompts -- each stage writes out its own
        character description (Krea2) and character tags (Illustrious); nothing
        is prepended except the Illustrious quality prefix and LoRA triggers.

    seed: one value for every sampler (None = independent random seeds).
    il_denoise (setting): Illustrious pass strength (0.2 default; 0.05-0.1 has a verified
        ghost/noise failure rate -- re-roll the seed if one appears).
    loras: {"name": strength} chained onto the model (see image_loras.py; an entry's
        `default` in loras.json applies when unset, {"name": 0} turns it off).
    face_detailer_enabled: False routes the raw upscale past FaceDetailer.
    edge_controlnet_strength (setting): overrides the HED/normal/canny passes derived from
        Krea2's render; 0.0 frees the composition (use with il_denoise=1.0 and
        explicit colour tags in il_scene).
    """
    graph = load_template()

    krea_scene, il_scene = image_loras.apply(graph, "full", loras, krea_scene, il_scene)

    graph[NODE_KREA_PROMPT]["inputs"]["text"] = krea_scene

    if prepend_quality_prefix and not il_scene.strip().startswith("masterpiece"):
        il_scene = QUALITY_PREFIX + il_scene
    graph[NODE_IL_PROMPT]["inputs"]["text"] = il_scene

    graph[NODE_IL_NEGATIVE]["inputs"]["text"] = il_negative or DEFAULT_IL_NEGATIVE
    graph[NODE_FACE_DETAILER_SWITCH]["inputs"]["switch"] = bool(face_detailer_enabled)
    settings = {**DEFAULTS, **{k: v for k, v in settings.items() if v is not None and v != ""}}
    graph_settings.apply(graph, graph_settings.FULL, settings)
    _prune(graph, edges=settings.get("edge_controlnet_strength") != 0,
           plain_upscale=settings.get("upscale_denoise") == 0 and settings.get("il_mode") != "upscale_only",
           upscale_only=settings.get("il_mode") == "upscale_only", face=bool(face_detailer_enabled),
           no_model=settings.get("upscale_model") == "none")

    graph_settings.add_seedvr(graph, settings)

    for nid, key in SEED_NODES:
        if nid in graph:
            graph[nid]["inputs"][key] = seed if seed is not None else secrets.randbits(48)
    if NODE_UPSCALE_SAMPLER in graph:
        graph[NODE_UPSCALE_SAMPLER]["inputs"]["seed"] = secrets.randbits(48)  # own noise, not the base pass's

    return graph
