#!/usr/bin/env python3
"""
Krea2 character + Illustrious background engine ("bgswap").

Pipeline: Krea2 text2img (photoreal character, same as the krea2 engine) ->
BiRefNet person matte -> Illustrious *inpaints everything except the
person* (Fooocus inpaint patch + Differential Diffusion, optional depth
ControlNet from the Krea2 render so the room's perspective/floor plane still
fits her pose) -> the original Krea2 person pixels are pasted back on top
with a feathered mask -> ESRGAN upscale -> sharpen -> grain. Before the
paste, the person is soft-light blended with a heavily blurred copy of the
IL background (harmonize) so she picks up the room's colour/brightness
instead of keeping Krea2's neutral indoor light. Finally a low-denoise
(default 0.1) Illustrious img2img pass runs over the *whole* composite
("unify"), prompted with the stage's il_scene (character + pose tags) + the
room tags, to give her a slight anime sheen and tie both halves together. Set
unify_denoise=0 to skip it (she then stays 100% Krea2 pixels).

The person is never resampled by any model, so she stays exactly photoreal;
only the background is anime. Because IL paints *around* her (inpainting,
not a separately generated plate), it sees her silhouette and fills in
whatever she's sitting/lying on consistently.

bg_denoise:
  1.0 (default) — background generated from scratch by IL. The Krea2
      background is pre-filled (telea) first so none of it leaks through.
  ~0.5-0.75 — "merge" mode: IL restyles Krea2's own background instead
      (keeps its layout/lighting, so the character's lighting matches
      better, but the room is Krea2's room, not IL's). The pre-fill is
      skipped in this mode, since it would erase what we're restyling.

Stage files: needs an `il_background` field per stage (Danbooru tags for the
room only — no character tags; verify them against the Danbooru tag list), or set one
shared background as a `settings` key. il_scene feeds
only the unify pass; face_wildcard is accepted and ignored.

Only builds graphs; comfy_api.py submits them (run_pipeline.py --engine bgswap).
"""
import json
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graph_settings
import image_loras

TEMPLATE_PATH = HERE / "templates" / "krea2_il_bgswap_template.json"

NODE_KREA_PROMPT = "30:6"  # CLIPTextEncode.text (whole Krea2 prompt, character written out per stage)
NODE_INPAINT_MASK = "700:2"     # GrowMaskWithBlur: expand=-edge_overlap, output[1] = inverted (background) mask
NODE_PREFILL = "700:3"          # INPAINT_MaskedFill (only used when bg_denoise >= 1.0)
NODE_IL_POSITIVE = "700:12"
NODE_IL_NEGATIVE = "700:14"
NODE_DEPTH_CN = "700:23"        # ControlNetApplyAdvanced (depth)
NODE_INPAINT_ENCODE = "700:30"  # INPAINT_VAEEncodeInpaintConditioning
NODE_IL_SAMPLER = "700:33"
NODE_AMBIENT_MAP = "700:60"     # ImageScaleBy: IL background shrunk to a tiny colour map (scale_by = ambient detail)
NODE_UNIFY_POSITIVE = "700:70"
NODE_UNIFY_NEGATIVE = "700:71"
NODE_UNIFY_SAMPLER = "700:73"   # KSampler over the whole composite, denoise = unify_denoise
UNIFY_NODES = ("700:70", "700:71", "700:72", "700:73", "700:74")
SEED_NODES = [("30:3", "seed"), ("700:33", "seed"), ("700:73", "seed")]

NODE_LABELS = {
    "29": "krea2_base",
    "700:50": "il_background_raw",
    "700:52": "inpaint_mask",
    "700:64": "composite_no_harmonize",
    "700:75": "composite_pre_unify",
    "436": "final",
}
FINAL_IMAGE_NODE = "436"

# Style tags (all verified against the Danbooru tag list) are pushed hard on purpose:
# the room should read as unmistakably anime next to the photoreal person.
BG_QUALITY_PREFIX = (
    "masterpiece, best quality, amazing quality, absurdres, highres, "
    "extremely detailed, official art, anime screencap, anime coloring, "
    "colorful, bloom, lens flare, light particles, scenery, no humans, "
)
DEFAULT_BG_NEGATIVE = (
    "lowres, worst quality, blurry, watermark, text, "
    "1girl, 2girls, multiple girls, 1boy, extra limbs, bad anatomy, "
    "photorealistic, realistic, 3d, muted color, monochrome, greyscale, "
    "simple background"
)
DEFAULT_BG_DENOISE = 1.0
DEFAULT_DEPTH_STRENGTH = 0.35
DEFAULT_EDGE_OVERLAP = 6     # px the IL inpaint reaches into the person's edge (repaints Krea2's background halo)
# Harmonize defaults picked from a local A/B preview on the first bgswap test
# batch: soft_light alone mostly darkens/adds contrast (0.5 was barely
# visible); it takes a small normal-mode colour wash on top to actually give
# her the room's hue cast. Wash >0.2 starts to look hazy/washed out.
# Unify pass: positive = quality/style prefix + il_scene + il_background.
# Its negative must NOT carry the background pass's "1girl/no humans"-style
# exclusions — she's in frame for this pass.
UNIFY_QUALITY_PREFIX = (
    "masterpiece, best quality, amazing quality, absurdres, highres, "
    "extremely detailed, official art, anime coloring, colorful, bloom, "
)
DEFAULT_UNIFY_NEGATIVE = (
    "lowres, worst quality, blurry, watermark, text, bad anatomy, bad hands, "
    "extra fingers, deformed teeth, bad teeth, 2girls, censored, muted color, monochrome"
)
DEFAULT_UNIFY_DENOISE = 0.1  # NOTE: the full engine saw occasional ghost/noise output at 0.05-0.1 IL denoise — re-roll the seed if one appears


def load_template() -> dict:
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def _apply_unify(graph: dict, unify, il_scene, il_background) -> None:
    """The optional whole-image Illustrious pass over the finished composite (off: its nodes go, the composite routes past it)."""
    if unify > 0:
        parts = [(il_scene or "").strip().rstrip(","), il_background]
        unify_text = UNIFY_QUALITY_PREFIX + ", ".join(p for p in parts if p)
        graph[NODE_UNIFY_POSITIVE]["inputs"]["text"] = unify_text
        graph[NODE_UNIFY_NEGATIVE]["inputs"]["text"] = DEFAULT_UNIFY_NEGATIVE
        graph[NODE_UNIFY_SAMPLER]["inputs"]["denoise"] = unify
        graph[NODE_UNIFY_SAMPLER]["inputs"]["cfg"] = graph[NODE_IL_SAMPLER]["inputs"]["cfg"]
    else:
        graph["30:966"]["inputs"]["image"] = ["700:41", 0]
        for nid in UNIFY_NODES + ("700:75",):
            del graph[nid]


def _apply_depth(graph: dict, strength) -> None:
    """The depth ControlNet from the Krea2 render on the IL pass (0: IL reads the plain prompts, the chain's nodes go)."""
    if strength > 0:
        graph[NODE_DEPTH_CN]["inputs"]["strength"] = strength
    else:
        graph[NODE_INPAINT_ENCODE]["inputs"]["positive"] = [NODE_IL_POSITIVE, 0]
        graph[NODE_INPAINT_ENCODE]["inputs"]["negative"] = [NODE_IL_NEGATIVE, 0]
        for nid in ("700:20", "700:21", "700:22", NODE_DEPTH_CN):
            del graph[nid]


def build_graph(
    krea_scene: str,
    il_background: str = None,
    il_background_negative: str = None,
    seed: int = None,
    loras: dict = None,
    upscale_factor: float = None,
    bg_denoise: float = None,
    depth_strength: float = None,
    edge_overlap: int = None,
    unify_denoise: float = None,
    il_scene: str = None,
    **settings,  # graph_settings.BGSWAP keys; other stage fields are ignored
) -> dict:
    """Patch the bgswap template.

    il_background: Danbooru tags for the room only (required).
    bg_denoise: see module docstring (1.0 = from scratch, ~0.5-0.75 = restyle
        Krea2's own background).
    depth_strength: depth ControlNet (from the Krea2 render) on the IL pass;
        0.0 removes it from the graph entirely, giving IL total freedom
        over the room layout (perspective may then not match her pose).
    edge_overlap / paste_feather: seam tuning. If a halo of Krea2's
        original background shows around her, raise edge_overlap; if her
        silhouette looks cut-out/hard, raise paste_feather.
    matte_model: BiRefNetRMBG variant (default BiRefNet-HR; try
        BiRefNet-HR-matting for finer hair edges). BiRefNet is salient-object
        matting, so furniture she's in contact with (a chair, a bed) may get
        kept as "foreground" — that's usually fine, sometimes not.
    harmonize: 0-1, how strongly her colours are pulled toward the room's
        light (soft-light blend with a blurred copy of the IL background).
        A non-harmonized composite is always saved to _debug for A/B.
    harmonize_mode: ImageBlend mode (soft_light default; "multiply" darkens
        more, "overlay" is punchier).
    tint: 0-1 straight colour wash of the room's ambient colour over her,
        applied after harmonize (default 0.15).
    il_scene: the stage's character + pose tags, used only
        by the unify pass's prompt — at higher unify_denoise, e.g. 0.3, IL
        needs the pose/clothing tags or it drifts.
    unify_denoise: whole-image Illustrious pass over the finished composite
        (default 0.1). 0 removes the pass from the graph.
    il_cfg: IL background sampler CFG (default 5.5; higher = more saturated,
        eventually burnt).
    """
    if not il_background:
        raise ValueError("bgswap engine needs il_background (Danbooru tags for the room) per stage or as a `settings` key")
    graph = load_template()

    # Krea2 side of this template is identical to the krea2 engine's: Krea2-pass LoRAs only.
    krea_scene, _ = image_loras.apply(graph, "krea2", loras, krea_scene, "")

    graph[NODE_KREA_PROMPT]["inputs"]["text"] = krea_scene
    graph_settings.apply(graph, graph_settings.BGSWAP, settings)

    bg_text = il_background
    if not bg_text.strip().startswith("masterpiece"):
        bg_text = BG_QUALITY_PREFIX + bg_text
    graph[NODE_IL_POSITIVE]["inputs"]["text"] = bg_text
    graph[NODE_IL_NEGATIVE]["inputs"]["text"] = il_background_negative or DEFAULT_BG_NEGATIVE

    denoise = DEFAULT_BG_DENOISE if bg_denoise is None else bg_denoise
    graph[NODE_IL_SAMPLER]["inputs"]["denoise"] = denoise
    if denoise < 1.0:
        # merge mode: encode Krea2's real background, not the telea pre-fill
        graph[NODE_INPAINT_ENCODE]["inputs"]["pixels"] = ["30:8", 0]
        del graph[NODE_PREFILL]

    unify = DEFAULT_UNIFY_DENOISE if unify_denoise is None else unify_denoise
    _apply_unify(graph, unify, il_scene, il_background)

    strength = DEFAULT_DEPTH_STRENGTH if depth_strength is None else depth_strength
    _apply_depth(graph, strength)

    graph[NODE_INPAINT_MASK]["inputs"]["expand"] = -(DEFAULT_EDGE_OVERLAP if edge_overlap is None else edge_overlap)

    if upscale_factor is not None:
        graph["30:967"]["inputs"]["scale_by"] = upscale_factor / 4.0  # relative to the 4x ESRGAN output

    graph_settings.add_seedvr(graph, settings)

    for nid, key in SEED_NODES:
        if nid not in graph:
            continue
        graph[nid]["inputs"][key] = seed if seed is not None else secrets.randbits(48)

    return graph
