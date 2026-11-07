"""
Krea2-only engine: Krea2 render -> Impact-Pack iterative upscale to 1.5x (ESRGAN 4x
in pixel space, then a low-denoise Krea2 re-diffusion with the Krea2 model/VAE/CLIP) ->
sharpen -> film grain. No Illustrious pass, no FaceDetailer, no ControlNet -- pure photoreal.

Earlier this engine avoided a second Krea2 pass because the INT8 turbo checkpoint
(8 steps, cfg 1) went blotchy on re-noise. If that shows up again, lower
`upscale_denoise` (project.json settings) toward 0.1.

Only builds graphs; comfy_api.py submits them.
"""
import json
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graph_settings
import image_loras

TEMPLATE_PATH = HERE / "templates" / "krea2_only_template.json"

NODE_KREA_PROMPT = "30:6"  # CLIPTextEncode.text (whole Krea2 prompt, character written out per stage)
NODE_BASE_SAMPLER = "30:3"
NODE_UPSCALE_SAMPLER = "30:968"  # always its own random seed, so the upscale noise isn't the base pass's noise

NODE_LABELS = {
    "29": "krea2_base_preview",
    "436": "final",
}
FINAL_IMAGE_NODE = "436"
# LoRA node ids live in image_loras.py.


def load_template() -> dict:
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def build_graph(
    krea_scene: str,
    seed: int = None,
    loras: dict = None,
    **settings,  # graph_settings.KREA2 keys; other stage fields (il_scene, face_wildcard...) are ignored
) -> dict:
    """Patch the Krea2-only template. `**settings` lets this share stage JSON
    files written for the two-pass (Krea2+Illustrious) pipeline — any
    il_*/face_wildcard fields in a stage dict are simply not used here.

    loras: see generate.py's build_graph docstring (image_loras.py); only the
    Krea2-pass LoRAs apply in this engine.

    Upscale (upscale_factor, upscale_model, upscale_steps/cfg/sampler/scheduler/denoise settings): the
    iterative ESRGAN + low-denoise Krea2 pass; final size = upscale_factor x the base render (default 1.5).
    """
    graph = load_template()

    krea_scene, _ = image_loras.apply(graph, "krea2", loras, krea_scene, "")

    graph[NODE_KREA_PROMPT]["inputs"]["text"] = krea_scene
    graph_settings.apply(graph, graph_settings.KREA2, settings)

    graph_settings.add_seedvr(graph, settings)

    graph[NODE_BASE_SAMPLER]["inputs"]["seed"] = seed if seed is not None else secrets.randbits(48)
    graph[NODE_UPSCALE_SAMPLER]["inputs"]["seed"] = secrets.randbits(48)

    return graph
