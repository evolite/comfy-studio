"""
Face engine: an existing image (the IL + upscale result) -> Illustrious FaceDetailer -> save. Only builds
graphs; project.py `generate --face` submits them (input IL/stage_N.png, passed as `_image`).

The detailer's prompt is the stage's `il_scene` (quality prefix + IL LoRA triggers added, like the full engine) with
`face_wildcard` as the Impact wildcard; the tunables are the face_* / il_checkpoint keys in graph_settings.FACE.
"""
import json
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate
import graph_settings
import image_loras

TEMPLATE_PATH = HERE / "templates" / "il_face_template.json"
NODE_LABELS = {"436": "final"}
FINAL_IMAGE_NODE = "436"


def build_graph(_image: str, il_scene: str = "", il_negative: str = None, seed: int = None, loras: dict = None, **settings) -> dict:
    """_image: the uploaded input's name. loras: only the Illustrious files apply (chained onto the detailer's model).
    Other stage fields / settings keys are ignored."""
    graph = json.loads(TEMPLATE_PATH.read_text())
    il_scene = image_loras.apply_il(graph, "2", ["7"], loras, il_scene, "face")
    if not il_scene.strip().startswith("masterpiece"):
        il_scene = generate.QUALITY_PREFIX + il_scene
    graph["1"]["inputs"]["image"] = _image
    graph["3"]["inputs"]["text"] = il_scene
    graph["4"]["inputs"]["text"] = il_negative or generate.DEFAULT_IL_NEGATIVE
    graph["7"]["inputs"]["seed"] = seed if seed is not None else secrets.randbits(48)
    settings = {"face_wildcard": generate.DEFAULTS["face_wildcard"], **{k: v for k, v in settings.items() if v is not None and v != ""}}
    graph_settings.apply(graph, graph_settings.FACE, settings)
    return graph
