"""
Table-driven tunables. Each engine maps a settings key to the template node input(s) it patches; a key
absent from `settings` leaves the template's value alone, so the templates hold the defaults. Users set
keys in project.json (`"settings": {...}` run-wide, or as a stage field to override one stage).

Node ids are hand-traced from the templates (see AGENTS.md "Node-ID fragility") -- this file is the only
place tunables' ids live. Prompts, seeds, LoRAs and node pruning stay in each engine's build_graph.
"""
import secrets

# Krea2 base pass + final polish -- the same nodes in all three templates.
_KREA = {
    "krea_unet": ("30:10", "unet_name"),
    "krea_vae": ("30:12", "vae_name"),  # e.g. the HDR VAE (local file); the template default maps to local_models' krea_vae
    "krea_steps": ("30:3", "steps"),
    "krea_cfg": ("30:3", "cfg"),
    "krea_sampler": ("30:3", "sampler_name"),
    "krea_scheduler": ("30:3", "scheduler"),
    "aspect_ratio": ("49", "aspect_ratio"),
    "megapixels": ("49", "megapixels"),
}


def _upscale(model, sampler=None, factor=None) -> dict:
    """The upscale group. Same key names in every engine so a project's upscale settings carry over when the
    engine changes (each template supplies its own defaults: the sampler differs per engine, so explicit
    steps/cfg/sampler/denoise values only make sense for the engine's own model)."""
    keys = {"upscale_model": (model, "model_name")}
    if factor:
        keys["upscale_factor"] = (factor, "upscale_factor")
    if sampler:
        keys.update({f"upscale_{f}": (sampler, n) for f, n in
                     (("steps", "steps"), ("cfg", "cfg"), ("sampler", "sampler_name"), ("scheduler", "scheduler"), ("denoise", "denoise"))})
    return keys


_POLISH = {  # krea2 + bgswap: sharpen, grain
    "sharpen": ("30:963", "amount"),
    "grain_density": ("30:964", "density"),
    "grain_intensity": ("30:964", "intensity"),
}

SEEDVR_KEYS = {"seedvr", "seedvr_resolution"}  # optional SeedVR2 3B finish (add_seedvr), same in every engine
HIRES_KEYS = {"hires_scale", "hires_denoise", "hires_vae", "hires_lora_scale", "hires_prompt", "finish_resolution"}  # the local SesquiLSR latent-upscale pass (hires_graph; hires_lora_scale/prompt: steps._hires) + its SeedVR finish step, same in every engine

FULL = {
    **_KREA,
    "il_checkpoint": ("433:368", "ckpt_name"),
    "il_steps": ("433:366", "steps"),
    "il_cfg": ("433:366", "cfg"),
    "il_sampler": ("433:366", "sampler_name"),
    "il_scheduler": ("433:366", "scheduler"),
    "il_denoise": ("433:366", "denoise"),
    "hed_strength": ("433:369:356", "strength"),
    "normal_strength": ("433:369:357", "strength"),
    "canny_strength": ("433:369:358", "strength"),
    "edge_controlnet_strength": [("433:369:356", "strength"), ("433:369:357", "strength"), ("433:369:358", "strength")],
    **_upscale("434:397", sampler="434:392", factor="434:393"),
    "upscale_step_mode": ("434:393", "step_mode"),  # simple | geometric
    "upscale_iterations": ("434:393", "steps"),  # IterativeLatentUpscale hops (upscale_steps is the sampler's)
    "face_wildcard": ("434:421", "wildcard"),
    "face_steps": ("434:421", "steps"),
    "face_cfg": ("434:421", "cfg"),
    "face_denoise": ("434:421", "denoise"),
    "sharpen": ("434:405", "amount"),
    "grain_density": ("434:406", "density"),
    "grain_intensity": ("434:406", "intensity"),
}
FULL_SPECIAL = {"face_detailer_enabled", "il_mode"} | SEEDVR_KEYS | HIRES_KEYS  # handled in code (pruning)

FACE = {  # the face engine (generate_face.py): FaceDetailer pass on an existing image
    "il_checkpoint": ("2", "ckpt_name"),
    "face_wildcard": ("7", "wildcard"),
    "face_steps": ("7", "steps"),
    "face_cfg": ("7", "cfg"),
    "face_denoise": ("7", "denoise"),
}

KREA2 = {
    **_KREA,
    **_POLISH,
    **_upscale("30:965", sampler="30:968", factor="30:969"),
    "upscale_step_mode": ("30:969", "step_mode"),  # simple | geometric
    "upscale_iterations": ("30:969", "steps"),  # IterativeLatentUpscale hops (upscale_steps is the sampler's)
}
KREA2_SPECIAL = SEEDVR_KEYS | HIRES_KEYS

BGSWAP = {
    **_KREA,
    **_POLISH,
    **_upscale("30:965"),
    "il_checkpoint": ("700:10", "ckpt_name"),
    "il_steps": ("700:33", "steps"),
    "il_cfg": ("700:33", "cfg"),
    "il_sampler": ("700:33", "sampler_name"),
    "il_scheduler": ("700:33", "scheduler"),
    "unify_steps": ("700:73", "steps"),
    "matte_model": ("700:1", "model"),
    "paste_feather": ("700:40", "blur_radius"),
    "harmonize": ("700:62", "blend_factor"),
    "harmonize_mode": ("700:62", "blend_mode"),
    "tint": ("700:65", "blend_factor"),
}
BGSWAP_SPECIAL = {"il_background", "il_background_negative", "upscale_factor", "bg_denoise", "depth_strength", "edge_overlap", "unify_denoise"} | SEEDVR_KEYS | HIRES_KEYS  # code: prunes / sign flips

MOSS = {  # generate_moss.py: the Krea2 node ids, plus the negative prompt
    **_KREA,
    "moss_negative": ("30:13", "text"),
}
MOSS_SPECIAL = SEEDVR_KEYS

ENGINE_KEYS = {
    "full": (FULL, FULL_SPECIAL),
    "krea2": (KREA2, KREA2_SPECIAL),
    "bgswap": (BGSWAP, BGSWAP_SPECIAL),
    "moss": (MOSS, MOSS_SPECIAL),
}


def apply(graph: dict, table: dict, settings: dict) -> None:
    """Patch every table key present (and not None) in `settings`; other keys are ignored here."""
    for key, targets in table.items():
        if settings.get(key) is None:
            continue
        for node_id, field in targets if isinstance(targets, list) else [targets]:
            graph[node_id]["inputs"][field] = settings[key]


def check_settings(engine: str, settings) -> dict:
    """Validate a project.json `settings` block: an object whose keys some engine knows. Keys the current
    engine has no node for (e.g. il_steps under krea2) are accepted and ignored, so one block survives an
    engine switch; only names no engine knows (typos) are rejected."""
    if not isinstance(settings, dict):
        raise SystemExit(f"settings must be an object, got {settings!r}")
    valid = set().union(*(set(t) | sp for t, sp in ENGINE_KEYS.values()))
    unknown = sorted(set(settings) - valid)
    if unknown:
        raise SystemExit(f"settings: unknown key(s) {unknown}; valid: {sorted(valid)}")
    return settings


def seedvr_nodes(src, resolution=None, model="seedvr2_ema_3b_fp8_e4m3fn.safetensors", color="lab") -> dict:
    """SeedVR2 upscale of image `src` ([node, slot]) to `resolution` (short edge, default 1680); output is "seedvr:up"."""
    return {
        "seedvr:dit": {"class_type": "SeedVR2LoadDiTModel", "inputs": {
            "model": model, "device": "cuda:0", "blocks_to_swap": 0, "swap_io_components": False,
            "offload_device": "none", "cache_model": False, "attention_mode": "sdpa"}},
        "seedvr:vae": {"class_type": "SeedVR2LoadVAEModel", "inputs": {
            "model": "ema_vae_fp16.safetensors", "device": "cuda:0", "encode_tiled": True, "encode_tile_size": 1024,
            "encode_tile_overlap": 128, "decode_tiled": True, "decode_tile_size": 1024, "decode_tile_overlap": 128,
            "tile_debug": "false", "offload_device": "none", "cache_model": False}},
        "seedvr:up": {"class_type": "SeedVR2VideoUpscaler", "inputs": {
            "image": src, "dit": ["seedvr:dit", 0], "vae": ["seedvr:vae", 0], "seed": secrets.randbits(31),
            "resolution": resolution or 1680, "max_resolution": 0, "batch_size": 1,
            "uniform_batch_size": False, "color_correction": color, "temporal_overlap": 0, "prepend_frames": 0,
            "input_noise_scale": 0, "latent_noise_scale": 0, "offload_device": "none", "enable_debug": False}},
    }


def hires_graph(krea_graph: dict, image_name: str, scale=None, denoise=None, vae=None) -> dict:
    """The Krea2 base graph (project.preview_graph: loaders, LoRAs, prompt, base KSampler 30:3) turned into a latent upscale of an
    existing still: LoadImage -> VAEEncode -> SesquiLSR (Wan 2.1 latents = Krea2's VAE) x`scale` (default 1.5, max 2.0) -> Krea2
    re-diffusion at `denoise` (default 0.05) with the stage's prompt/LoRAs -> tiled decode -> save node 436. Local only:
    SesquiLatentUpscale isn't on Comfy Cloud. `vae` (setting `hires_vae`, a Wan 2.1-latent VAE) replaces the preview's on 30:12 for
    this pass's encode + decode (the base decode 30:8 is dropped, so nothing else reads it); unset = the preview's `krea_vae`."""
    g = {k: v for k, v in krea_graph.items() if k not in ("30:5", "49", "30:8", "436")}
    if vae:
        g["30:12"] = {**g["30:12"], "inputs": {**g["30:12"]["inputs"], "vae_name": vae}}
    sampler = g.pop("30:3")
    g.update({
        "hires:load": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "hires:enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["hires:load", 0], "vae": ["30:12", 0]}},
        "hires:up": {"class_type": "SesquiLatentUpscale", "inputs": {
            "latent": ["hires:enc", 0], "model_format": "Wan 2.1", "scale": scale or 1.5, "half_precision": True}},
        "hires:ks": {"class_type": "KSampler", "inputs": {**sampler["inputs"], "latent_image": ["hires:up", 0],
                                                         "seed": secrets.randbits(48), "denoise": denoise or 0.05}},
        "hires:dec": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["hires:ks", 0], "vae": ["30:12", 0], "tile_size": 512,  # 1024 needs ~3.4 GB on top of the resident Krea2 UNet: CUDA OOM on 16 GB
                                                                "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}},
        "436": {"class_type": "SaveImage", "inputs": {"images": ["hires:dec", 0], "filename_prefix": "Hires/"}},
    })
    return g


def add_seedvr(graph: dict, settings: dict, final_node: str = "436") -> None:
    """`seedvr: true` -> SeedVR2 3B upscales the final image to `seedvr_resolution` (short edge, default 1680)
    just before the save node. It runs after sharpen/grain, so set grain to 0 and re-add it only if wanted."""
    if not settings.get("seedvr"):
        return
    graph.update(seedvr_nodes(graph[final_node]["inputs"]["images"], settings.get("seedvr_resolution")))
    graph[final_node]["inputs"]["images"] = ["seedvr:up", 0]
