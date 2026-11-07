"""
MOSS-Image engine (e.g. the Dirty MOSS checkpoint): one base render, nothing after it. The template reuses the Krea2 node ids
(30:10 UNet, 30:11 text encoder, 30:12 VAE, 30:6 prompt, 30:3 sampler, 30:8 decode, 49 resolution), so preview, takes,
`krea_*` settings and the local runner treat it like a Krea2 base pass; 30:13 is a real negative (MOSS runs at CFG ~2).

Needs ComfyUI >= 0.42 (CLIPLoader type moss_image, EmptyFlux2LatentImage), the Qwen3.5 9B text encoder and the FLUX.2 VAE
(Comfy-Org/MOSS-Image on Hugging Face). Local only: the checkpoints are in local_only_unets. Krea2 LoRAs don't fit this
model; only MOSS LoRAs chain off 30:10.

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

TEMPLATE_PATH = HERE / "templates" / "moss_template.json"

NODE_PROMPT = "30:6"
NODE_SAMPLER = "30:3"
NODE_LABELS = {"436": "final"}
FINAL_IMAGE_NODE = "436"


def load_template() -> dict:
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def build_graph(krea_scene: str, seed: int = None, loras: dict = None, **settings) -> dict:
    """krea_scene = the MOSS caption; other stage fields (il_scene, ...) are ignored."""
    graph = load_template()
    krea_scene, _ = image_loras.apply(graph, "moss", loras, krea_scene, "")
    graph[NODE_PROMPT]["inputs"]["text"] = krea_scene
    graph_settings.apply(graph, graph_settings.MOSS, settings)
    graph_settings.add_seedvr(graph, settings)
    graph[NODE_SAMPLER]["inputs"]["seed"] = seed if seed is not None else secrets.randbits(48)
    return graph
