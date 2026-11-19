"""The i2v graph library: what a clip's graph is, per video engine (MiniMax H3, LTX 2.3). The animate step
(pipeline/steps.py, through pipeline/runner.py and a target) builds each stage's clip with build_graph(); this module
holds the engines' node maps and templates, the frame grid, the LoRA chain, RIFE, keyframes and the clip's aspect.
"""
import json
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
import config
import interpolate
import lora_registry

DEFAULT_TEMPLATE = HERE / "templates" / "video_minimax_h3_i2v_template.json"
AUDIO_TEMPLATE = HERE / "templates" / "video_minimax_h3_i2v_audio_template.json"  # same node ids + a generated soundtrack

# i2v models: project.json `video_engine` (a stage's own `video_engine` wins). Each names the node ids build_graph patches in
# its template: image/prompt/seed = (node, input); length = (node, input, "seconds" | "frames"); size = the ResolutionSelector;
# the LoRA chain runs from model_src's model input into every model_dst; video/save = CreateVideo/SaveVideo; frames on a
# grid*k + offset grid at fps (k >= min_k); `mute` = (CreateVideo, nodes to drop) when the model always makes audio.
VIDEO_ENGINES = {
    "minimax": {"label": "MiniMax H3", "template": DEFAULT_TEMPLATE, "audio_template": AUDIO_TEMPLATE,
                "fps": 24, "grid": 17, "offset": 5, "min_k": 2,
                "image": ("114", "image"), "prompt": ("105:104", "prompt"), "length": ("105:111", "value", "seconds"),
                "seed": ("105:15", "noise_seed"), "size": "115", "model_src": "105:16", "model_dst": ("105:9", "105:16"),
                "lora_prefix": "105:3", "video": "105:91", "save": "92"},
    "ltx": {"label": "LTX 2.3 (Sulphur 2)", "template": HERE / "templates" / "video_ltx23_i2v_template.json", "audio_template": None,
            "fps": 24, "grid": 8, "offset": 1, "min_k": 2,
            "image": ("ltx:image", "image"), "prompt": ("ltx:prompt", "text"), "length": ("ltx:length", "value", "frames"),
            "seed": ("ltx:seed", "noise_seed"), "size": "ltx:size", "model_src": "ltx:guider", "model_dst": ("ltx:guider",),
            "lora_prefix": "ltx:lora", "video": "ltx:video", "save": "ltx:save", "mute": ("ltx:video", ("ltx:audio_decode",))},
}
DEFAULT_ENGINE = "minimax"


def engine_spec(engine) -> dict:
    if engine not in VIDEO_ENGINES:
        raise SystemExit(f"unknown video_engine {engine!r}; known: {', '.join(VIDEO_ENGINES)}")
    return VIDEO_ENGINES[engine]


def frames(engine, seconds) -> int:
    """Frame count a clip of `seconds` renders at: snapped up to the engine's grid*k + offset (k >= min_k)."""
    e = engine_spec(engine)
    return e["grid"] * max(e["min_k"], math.ceil((math.ceil(seconds * e["fps"]) - e["offset"]) / e["grid"])) + e["offset"]


def template_for(engine, audio=False) -> dict:
    e = engine_spec(engine)
    return json.loads((e["audio_template"] if audio and e["audio_template"] else e["template"]).read_text())

# Extra i2v LoRAs on top of the turbo one already in the base template: the video half of loras.json (lora_registry.py).
# Key = name used in a stage's `video_loras` ({"name": strength}); a key ending .safetensors is taken as a literal filename.


def __getattr__(name):
    """`LORA_FILES` = {name: file}, read fresh from loras.json so a web-UI edit applies without a restart."""
    if name == "LORA_FILES":
        return lora_registry.video_files()
    raise AttributeError(name)


SEED_NODE = VIDEO_ENGINES["minimax"]["seed"][0]


def check_loras(loras, label: str, engine: str = None) -> dict:
    """With `engine`, a library LoRA made for another i2v model is refused too (loras.json `engine`, default minimax)."""
    if not isinstance(loras, dict):
        raise SystemExit(f"{label}: video_loras must be an object like {{\"motion_lora\": 1.0}}, got {loras!r}")
    known = lora_registry.video_engines()
    for name, strength in loras.items():
        if name not in known and not name.endswith(".safetensors"):
            raise SystemExit(f"{label}: unknown LoRA {name!r}; known: {', '.join(known)} (or a literal .safetensors filename)")
        if engine and strength and name in known and known[name] != engine:
            raise SystemExit(f"{label}: video LoRA {name!r} is for the {known[name]} model, this stage animates with {engine}")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength):
            raise SystemExit(f"{label}: strength for {name!r} must be a number (decimals ok), got {strength!r}")
    off = lora_registry.disabled_in(loras, "video")
    if off:
        print(f"  warning: {label}: {', '.join(off)} switched off on the LoRAs page (still applied; pick another)", file=sys.stderr)
    return loras


def add_lora_chain(graph: dict, loras: dict, engine: str = DEFAULT_ENGINE) -> None:
    """Chain one LoraLoaderModelOnly per entry (in order) after whatever model the guider currently uses,
    then point the scheduler and guider at the tail. Strength 0 is skipped."""
    e = engine_spec(engine)
    prev = graph[e["model_src"]]["inputs"]["model"]
    files = lora_registry.video_files()
    for k, (name, strength) in enumerate(loras.items()):
        if strength == 0:
            continue
        node_id = f"{e['lora_prefix']}{k:02d}"
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": files.get(name, name), "strength_model": strength, "model": prev},
            "_meta": {"title": f"Load LoRA ({name})"},
        }
        prev = [node_id, 0]
    for nid in e["model_dst"]:
        graph[nid]["inputs"]["model"] = prev


RIFE_SAVE = "rife:save"


def add_rife(graph: dict, engine: str = DEFAULT_ENGINE) -> None:
    """Animate + RIFE: interpolate the decoded frames x interpolate.MULTIPLIER in the same job and save only that clip (node
    RIFE_SAVE, with the clip's audio when the template has some): the template's 24 fps SaveVideo is dropped, so the 24 fps
    clip is never written or downloaded."""
    e = engine_spec(engine)
    src = graph[e["video"]]["inputs"]
    graph["rife:vfi"] = {"class_type": "RIFE VFI", "inputs": interpolate.rife_inputs(src["images"]), "_meta": {"title": "RIFE VFI"}}
    graph["rife:video"] = {"class_type": "CreateVideo", "inputs": interpolate.video_inputs(["rife:vfi", 0], src["fps"], src.get("audio")),
                           "_meta": {"title": "Create Video (RIFE)"}}
    save = graph[e["save"]]["inputs"]
    graph[RIFE_SAVE] = {"class_type": "SaveVideo", "inputs": {**save, "video": ["rife:video", 0],
                                                              "filename_prefix": save["filename_prefix"] + "_rife"}, "_meta": {"title": "Save Video (RIFE)"}}
    del graph[e["save"]]


# the ResolutionSelector's aspect ratios (the templates default to 2:3 portrait)
ASPECTS = ["2:3 (Portrait Photo)", "3:4 (Portrait Standard)", "9:16 (Portrait Widescreen)", "1:1 (Square)",
           "3:2 (Photo)", "4:3 (Standard)", "16:9 (Widescreen)", "21:9 (Ultrawide)"]


def aspect_of(png: Path) -> str:
    """The ResolutionSelector aspect closest to the still's width/height (PNG header), so a landscape still animates landscape."""
    w, h = struct.unpack(">II", Path(png).read_bytes()[16:24])
    def ratio(a):
        x, y = a.split(" ")[0].split(":")
        return int(x) / int(y)
    return min(ASPECTS, key=lambda a: abs(math.log(ratio(a) * h / w)))


def build_graph(template: dict, image_name: str, prompt: str, duration_value: float,
                loras: dict | None = None, seed: int | None = None, megapixels: float | None = None, rife: bool = False,
                engine: str = DEFAULT_ENGINE, audio: bool = False, aspect: str | None = None, keyframes: list | None = None) -> dict:
    """`template` = template_for(engine, audio). `audio` only matters for an engine that always makes sound (`mute`).
    `aspect` (aspect_of(still)) sets the clip's aspect ratio; None keeps the template's.
    `keyframes` = [(uploaded image name, frame index)] anchored after the first frame (add_keyframes; MiniMax only)."""
    e = engine_spec(engine)
    graph = json.loads(json.dumps(template))
    graph[e["image"][0]]["inputs"][e["image"][1]] = image_name
    graph[e["prompt"][0]]["inputs"][e["prompt"][1]] = prompt
    node, field, unit = e["length"]
    graph[node]["inputs"][field] = duration_value if unit == "seconds" else frames(engine, duration_value)
    if megapixels:
        graph[e["size"]]["inputs"]["megapixels"] = megapixels
    if aspect:
        graph[e["size"]]["inputs"]["aspect_ratio"] = aspect
    if loras:
        add_lora_chain(graph, loras, engine)
    if seed is not None:
        graph[e["seed"][0]]["inputs"][e["seed"][1]] = seed
    if "mute" in e and not audio:
        video, drop = e["mute"]
        graph[video]["inputs"].pop("audio", None)
        for nid in drop:
            graph.pop(nid, None)
    if keyframes:
        add_keyframes(graph, keyframes, engine, frames(engine, duration_value))
    if rife:
        add_rife(graph, engine)
    return graph


def keyframe_frames(engine, seconds, ats: list) -> list:
    """Frame index per keyframe from its `at` (seconds; negative = from the end; None = spread evenly, the last one on the
    clip's last frame), clamped inside the clip. The stage's own still is frame 0."""
    total, fps, n = frames(engine, seconds), engine_spec(engine)["fps"], len(ats)
    out = []
    for i, at in enumerate(ats):
        if at is None:
            f = total - 1 if i == n - 1 else round(seconds * (i + 1) / n * fps)
        else:
            f = round(at * fps)
        f = total + f if f < 0 else f
        out.append(min(max(f, 1), total - 1))
    return out


I2VA_LINE = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."


def keyframe_refs(prompt: str, frames_at: list, fps: int = 24) -> str:
    """The motion with MiniMax's FL2VA alignment line (VIDEO_PROMPT_WRITING_GUIDE: first + last frame) in place of the I2VA
    first-frame line: Picture 1 at 0.00 s, then Picture 2.. at each keyframe's time."""
    marks = "; ".join(["Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video"] +
                      [f"Picture {i + 2} (from Shot 1) aligns with the {f / fps:.2f}-second mark of the target video" for i, f in enumerate(frames_at)])
    body = prompt.replace(I2VA_LINE, "").lstrip()
    return f"How the reference pictures align with the target video \u2014 {marks}.\n\n{body}"


def add_keyframes(graph: dict, keyframes: list, engine: str = DEFAULT_ENGINE, total: int | None = None) -> None:
    """More stills in a MiniMax clip. One on the clip's last frame goes into MiniMaxH3ImageToVideo's `last_frame` (H3's own
    first + last frame task: the text encoder sees it as <Picture 2>); any other is pinned at its frame by a
    MiniMaxH3AddGuide chained in front of the guider (the latent anchor only)."""
    if engine != "minimax":
        raise SystemExit(f"keyframes need the minimax engine, not {engine!r}")
    e = engine_spec(engine)
    i2v = graph[e["prompt"][0]]["inputs"]
    guider = graph[e["model_dst"][1]]["inputs"]
    sampler = next(n for n in graph.values() if n["class_type"] == "SamplerCustomAdvanced")["inputs"]
    cond = guider["conditioning"]
    for i, (image, frame) in enumerate(keyframes):
        k = f"kf{i}"
        graph[f"{k}:load"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        if total is not None and frame in (total - 1, -1):
            i2v["last_frame"] = [f"{k}:load", 0]
            continue
        graph[f"{k}:scale"] = {"class_type": "ImageScale", "inputs": {"image": [f"{k}:load", 0], "upscale_method": "lanczos",
                                "width": [e["size"], 0], "height": [e["size"], 1], "crop": "center"}}
        graph[f"{k}:guide"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": cond, "latent": sampler["latent_image"],
                                "frame_idx": frame, "vae": i2v["vae"], "image": [f"{k}:scale", 0]}}
        cond = [f"{k}:guide", 0]
    guider["conditioning"] = cond


STOCK_H3_UNET = config.STOCK_H3_UNET
