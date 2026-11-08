#!/usr/bin/env python3
"""
One-off: turn a finished image pipeline's stage_N.png files into short i2v
clips with a VIDEO_ENGINES model (per stage via --engines; default MiniMax H3, or --template), sized per-stage from a JSON
duration map (frame counts already snapped to the engine's grid, see frames()), and
save them to <pipeline_dir>/videos/stage_N.mp4.

Renders on Comfy Cloud, or with --server on a ComfyUI of our own (the RunPod pod, pipeline/runpod_pod.py) with the same
cloud-canonical graphs: checked against its /object_info first (a missing model fails before anything renders), then
submitted, polled and downloaded through pipeline/targets/local.py.

Usage (normally via `project.py animate`, which writes these files from project.json):
  python3 video/animate_pipeline.py <pipeline_dir> \
      --motion-prompts <pipeline_dir>/motion_prompts.json \
      --durations /path/to/stage_durations.json \
      [--loras loras.json] [--seeds seeds.json] [--engines engines.json]
"""
import argparse
import json
import math
import os
import secrets
import sys
import time
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
import comfy_api as api
import config
import interpolate
import lora_registry
import runpod_pod
import timings
from targets import local

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
    """Animate + RIFE: interpolate the decoded frames x interpolate.MULTIPLIER in the same cloud job and save that clip too
    (node RIFE_SAVE), with the clip's audio when the template has some. The 24 fps path (CreateVideo -> SaveVideo) is untouched."""
    e = engine_spec(engine)
    src = graph[e["video"]]["inputs"]
    graph["rife:vfi"] = {"class_type": "RIFE VFI", "inputs": interpolate.rife_inputs(src["images"]), "_meta": {"title": "RIFE VFI"}}
    graph["rife:video"] = {"class_type": "CreateVideo", "inputs": interpolate.video_inputs(["rife:vfi", 0], src["fps"], src.get("audio")),
                           "_meta": {"title": "Create Video (RIFE)"}}
    save = graph[e["save"]]["inputs"]
    graph[RIFE_SAVE] = {"class_type": "SaveVideo", "inputs": {**save, "video": ["rife:video", 0],
                                                              "filename_prefix": save["filename_prefix"] + "_rife"}, "_meta": {"title": "Save Video (RIFE)"}}


def build_graph(template: dict, image_name: str, prompt: str, duration_value: float,
                loras: dict | None = None, seed: int | None = None, megapixels: float | None = None, rife: bool = False,
                engine: str = DEFAULT_ENGINE, audio: bool = False) -> dict:
    """`template` = template_for(engine, audio). `audio` only matters for an engine that always makes sound (`mute`)."""
    e = engine_spec(engine)
    graph = json.loads(json.dumps(template))
    graph[e["image"][0]]["inputs"][e["image"][1]] = image_name
    graph[e["prompt"][0]]["inputs"][e["prompt"][1]] = prompt
    node, field, unit = e["length"]
    graph[node]["inputs"][field] = duration_value if unit == "seconds" else frames(engine, duration_value)
    if megapixels:
        graph[e["size"]]["inputs"]["megapixels"] = megapixels
    if loras:
        add_lora_chain(graph, loras, engine)
    if seed is not None:
        graph[e["seed"][0]]["inputs"][e["seed"][1]] = seed
    if "mute" in e and not audio:
        video, drop = e["mute"]
        graph[video]["inputs"].pop("audio", None)
        for nid in drop:
            graph.pop(nid, None)
    if rife:
        add_rife(graph, engine)
    return graph


STOCK_H3_UNET = config.STOCK_H3_UNET


def pod_unet(graph: dict, unet: str) -> None:
    """On the pod, H3 graphs load `unet` (config.runpod_h3_unet(): a distilled turbo finetune) in place of the stock UNet; Comfy Cloud keeps the stock one.
    It is distilled already: the turbo switch's LoRA branch is skipped (still 8 steps) and the sampler is euler (author's pick)."""
    for nid, node in graph.items():
        if node.get("class_type") == "UNETLoader" and node["inputs"].get("unet_name") == STOCK_H3_UNET:
            node["inputs"]["unet_name"] = unet
            for other in graph.values():  # the ComfySwitchNode picking UNet vs UNet + turbo LoRA
                if other.get("class_type") == "ComfySwitchNode" and other["inputs"].get("on_false") == [nid, 0]:
                    other["inputs"]["on_true"] = [nid, 0]
        elif node.get("class_type") == "KSamplerSelect":
            node["inputs"]["sampler_name"] = "euler"


MAX_ATTEMPTS = 3
PENDING = "i2v_pending"  # <pipeline_dir>/_debug/i2v_pending/stage_N.json: a submitted clip not saved yet (see recover())
CHECK_FIELDS = local.NAME_FIELDS + ("unet_name", "clip_name", "vae_name")  # model inputs the --server preflight checks
SERVER_GRACE = 120  # seconds a --server run tolerates the pod not answering (proxy hiccups) before giving up


def pending_path(pipeline_dir: Path, stage_num: int) -> Path:
    return pipeline_dir / "_debug" / PENDING / f"stage_{stage_num}.json"


def write_pending(pipeline_dir: Path, r: dict, prompt_id: str, rife_dir: Path = None, server: str = None) -> None:
    """Record a submitted clip with everything save_clip needs, so it can still be saved if this process dies (studio restart).
    `server` = the ComfyUI it went to (None = Comfy Cloud)."""
    f = pending_path(pipeline_dir, r["stage"])
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"prompt_id": prompt_id, "stage": r["stage"], "engine": r["engine"], "loras": r["loras"], "seed": r["seed"],
                             "audio": r["audio"], "rife_dir": str(rife_dir) if rife_dir else None, "server": server, "pid": os.getpid(),
                             "submitted": time.time()}, indent=2))


def drop_pending(pipeline_dir: Path, stage_num: int) -> None:
    pending_path(pipeline_dir, stage_num).unlink(missing_ok=True)


def _alive(pid) -> bool:
    """Is the animate run that wrote a record still going (then it saves the clip itself)?"""
    try:
        return "animate_pipeline" in Path(f"/proc/{int(pid)}/cmdline").read_bytes().decode(errors="ignore")
    except (OSError, ValueError, TypeError):
        return False


def orphans(pipeline_dir: Path) -> list:
    """Pending records whose animate process is gone."""
    out = []
    for f in sorted((pipeline_dir / "_debug" / PENDING).glob("stage_*.json")):
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not _alive(rec.get("pid")):
            out.append(rec)
    return out


def server_job(server: str, pid: str):
    """(state, /history entry) of a prompt on a ComfyUI server: "success" / "error" once it ran, "running" while it is queued
    or running, "lost" when the server no longer knows it (restarted). Raises URLError when the server doesn't answer."""
    hist = json.loads(local._call(server, f"/history/{pid}")).get(pid)
    if hist and hist["status"].get("completed") is not None:
        return ("success" if hist["status"].get("status_str") == "success" else "error"), hist
    q = json.loads(local._call(server, "/queue"))
    return ("running" if hist or any(i[1] == pid for i in q["queue_running"] + q["queue_pending"]) else "lost"), None


def _recover_one(pipeline_dir: Path, rec: dict) -> bool:
    """Wait out one orphaned job (on Comfy Cloud, or the record's `server`) and save it like the dead run would have.
    True when it failed or was lost (the old clip stays in Previous versions)."""
    n, pid, server = rec["stage"], rec["prompt_id"], rec.get("server")
    print(f"[stage_{n}] recovering {'job on ' + server if server else 'cloud job'} {pid}", flush=True)
    if server:
        try:
            while (st := server_job(server, pid))[0] == "running":
                time.sleep(5)
            state, job = st
        except urllib.error.URLError as e:
            state = f"server gone: {getattr(e, 'reason', e)}"
    else:
        while (state := api.job_state(pid)[0]) not in api.TERMINAL:
            time.sleep(5)
        job = None
    if state == "success":
        save_clip(job or api.fetch_job(pid), n, pipeline_dir, rec["loras"], rec["seed"], rec["audio"],
                  Path(rec["rife_dir"]) if rec.get("rife_dir") else None, rec["engine"], server)
        return False
    where = f"the pod {server}" if server else "the cloud"
    print(f"[stage_{n}] failed or lost on {where} ({state}), nothing to save -- animate the stage again", flush=True)
    return True


def recover(pipeline_dir: Path) -> int:
    """Save the clips of orphaned records; returns how many stages failed or were lost, e.g. a pod that is gone."""
    failed = 0
    for rec in orphans(pipeline_dir):
        if _recover_one(pipeline_dir, rec):
            failed += 1
        drop_pending(pipeline_dir, rec["stage"])
    return failed


def save_clip(job: dict, stage_num: int, pipeline_dir: Path, loras: dict, seed: int, with_audio: bool, rife_dir: Path = None,
              engine: str = DEFAULT_ENGINE, server: str = None) -> None:
    """The 24 fps clip -> <pipeline_dir>/videos/stage_N.mp4; with rife_dir (Animate + RIFE) the interpolated one ->
    <rife_dir>/videos/stage_N.mp4 plus the source record project.py interpolate also writes. `job` = a Comfy Cloud job, or
    with `server` that ComfyUI's /history entry (the same node_id -> {key: [files]} outputs)."""
    debug_dir = pipeline_dir / "_debug" / f"i2v_stage_{stage_num}"
    saved = api.save_outputs(job, debug_dir, fetch=(lambda f, dest: local.download(server, f, dest)) if server else api.download)
    (debug_dir / "render_settings.json").write_text(json.dumps({"engine": engine, "loras": loras, "seed": seed, "audio": with_audio, "rife": bool(rife_dir)}, indent=2))
    clips = {nid: d for nid, d in saved if d.suffix == ".mp4"}
    main = clips.get(engine_spec(engine)["save"]) or next((d for nid, d in clips.items() if nid != RIFE_SAVE), None)
    if not main:
        print(f"[stage_{stage_num}] WARNING: no .mp4 output found among saved outputs", flush=True)
        return
    out_path = pipeline_dir / "videos" / f"stage_{stage_num}.mp4"
    out_path.write_bytes(main.read_bytes())
    print(f"[stage_{stage_num}] -> {out_path}", flush=True)
    if rife_dir:
        if RIFE_SAVE not in clips:
            print(f"[stage_{stage_num}] WARNING: no RIFE clip among the outputs", flush=True)
            return
        dest = rife_dir / "videos" / f"stage_{stage_num}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(clips[RIFE_SAVE].read_bytes())
        src = rife_dir / "_debug" / f"stage_{stage_num}"
        src.mkdir(parents=True, exist_ok=True)
        folder = "" if pipeline_dir.resolve() == rife_dir.parent.resolve() else pipeline_dir.name
        (src / "source.json").write_text(json.dumps({"folder": folder}, indent=2) + "\n")
        print(f"[stage_{stage_num}] -> {dest} (RIFE x{interpolate.MULTIPLIER})", flush=True)


def _check_local_only(stage_loras: dict, stage_engines: dict, server) -> None:
    """A local-only i2v LoRA on a cloud run would fail after the credits are spent; it is named here instead."""
    local_only = lora_registry.local_only_files()
    files = lora_registry.video_files()
    for k, v in stage_loras.items():
        check_loras(v, f"stage {k}", stage_engines.get(k, DEFAULT_ENGINE))
        for name, strength in v.items():
            if strength and files.get(name, name) in local_only and not server:
                raise SystemExit(f"stage {k}: video LoRA {name!r} is marked local-only (cloud: false in loras.json) and i2v runs "
                                 f"on Comfy Cloud: upload it and tick 'cloud' on the LoRAs page, or set it to 0")


def _render_record(args, png: Path, prompts, durations, stage_loras, stage_seeds, stage_audio, stage_engines, custom,
                   videos_dir, server):
    """One stage's render record, or None when it is skipped: a clip already exists, or no motion/duration entry."""
    stage_num = int(png.stem[6:])
    key = str(stage_num)
    if (videos_dir / f"stage_{stage_num}.mp4").exists():
        print(f"[stage_{stage_num}] already exists, skipping", flush=True)
        return None
    if key not in prompts or key not in durations:
        print(f"[stage_{stage_num}] no motion prompt/duration entry, skipping", flush=True)
        return None
    duration_value = durations[key]["duration_value"]
    engine = stage_engines.get(key, DEFAULT_ENGINE)
    print(f"[stage_{stage_num}] uploading {png.name} ({engine}, duration={duration_value}s, tier={durations[key]['tier_frames']}f)", flush=True)
    loras = stage_loras.get(key, {})
    seed = stage_seeds.get(key)
    if seed is None:
        seed = secrets.randbits(32)
    with_audio = bool(stage_audio.get(key))
    print(f"[stage_{stage_num}] loras={loras or 'none'} seed={seed}{' audio' if with_audio else ''}", flush=True)
    template = custom if custom and engine == DEFAULT_ENGINE and not with_audio else template_for(engine, with_audio)
    image = local.upload(server, png, f"{secrets.token_hex(4)}_{png.name}") if server else api.upload_image(png)  # pod: one inputs/ for every project
    graph = build_graph(template, image, prompts[key], duration_value, loras, seed, args.megapixels,
                        rife=bool(args.rife_dir), engine=engine, audio=with_audio)
    if server and engine == "minimax" and (unet := config.runpod_h3_unet()):
        pod_unet(graph, unet)
    return {"stage": stage_num, "graph": graph, "engine": engine, "loras": loras, "seed": seed, "audio": with_audio,
            "attempts": 0}


def _build_renders(args, prompts, durations, stage_loras, stage_seeds, stage_audio, stage_engines, custom, videos_dir, server) -> list:
    """A render record per stage that needs a clip (existing clips and stages without a prompt/duration are skipped)."""
    pngs = sorted((p for p in args.pipeline_dir.glob("stage_*.png") if p.stem[6:].isdigit()),
                  key=lambda p: int(p.stem[6:]))  # skip extras like stage_21a_facial_before.png
    return [r for png in pngs if (r := _render_record(args, png, prompts, durations, stage_loras, stage_seeds, stage_audio,
                                                      stage_engines, custom, videos_dir, server))]


def _poll_state(server, prompt_id: str, answered: list):
    """(state, /history entry) of one pending render; (None, None) while a silent pod is still within its grace period."""
    if server:
        try:
            state, job = server_job(server, prompt_id)
            answered[0] = time.time()
            return state, job
        except urllib.error.URLError as e:
            if time.time() - answered[0] > SERVER_GRACE:
                raise SystemExit(f"lost {server} ({getattr(e, 'reason', e)}); `project.py <name> recover` saves the clips "
                                 f"if the pod comes back") from e
            return None, None
    state, _ = api.job_state(prompt_id)
    return state, None


def _finish_render(args, server, prompt_id: str, job, state, r, failed: list, submit) -> None:
    """A render left the queue: save a success (logging its timings), resubmit a recoverable end state, else count it failed."""
    if state == "success":
        if not server:  # pod runs stay out of the baselines: another GPU than the cloud's
            job = api.fetch_job(prompt_id)
            timings.record_cloud(r["graph"], job)
        else:  # but a cold pod's first clips are logged with its start (runpod_pod.STARTS, the header's RunPod menu)
            runpod_pod.record_clip(server, local.processing_seconds(job))
        save_clip(job, r["stage"], args.pipeline_dir, r["loras"], r["seed"], r["audio"], args.rife_dir, r["engine"], server)
        drop_pending(args.pipeline_dir, r["stage"])
    elif r["attempts"] < MAX_ATTEMPTS:
        print(f"  [stage_{r['stage']}] ended in state '{state}', resubmitting...", flush=True)
        submit(r)
    else:
        print(f"  [stage_{r['stage']}] failed after {MAX_ATTEMPTS} attempts (state '{state}')", flush=True)
        failed.append(r["stage"])
        drop_pending(args.pipeline_dir, r["stage"])


def _submitter(args, server, pending: dict):
    """The submit closure for one poll run: queue a render's graph, write its pending record, log the id + ETA."""
    def submit(r):
        r["attempts"] += 1
        prompt_id = local.submit(server, r["graph"], f"stage {r['stage']} animate") if server else api.submit(r["graph"])[0]
        write_pending(args.pipeline_dir, r, prompt_id, args.rife_dir, server)
        eta = timings.expected(timings.sig(r["graph"]), False)
        print(f"  [stage_{r['stage']}] submitted -> {prompt_id} (attempt {r['attempts']})" + (f" expected {eta:.0f}s" if eta else ""), flush=True)
        pending[prompt_id] = r
    return submit


def _poll_and_save(args, server, renders) -> list:
    """Submit every render, then poll and save each as it finishes; returns the failed stage numbers."""
    pending = {}  # prompt_id -> render
    submit = _submitter(args, server, pending)

    for r in renders:
        submit(r)

    failed = []
    seen = {}
    answered = [time.time()]  # --server: when the pod last answered (a list so _poll_state can update it)
    while pending:
        for prompt_id, r in pending.copy().items():
            state, job = _poll_state(server, prompt_id, answered)
            if state is None or (server and state == "running"):
                continue
            if not server and state not in api.TERMINAL:
                _resubmit_stalled(prompt_id, r, state, pending, seen, submit)
                continue
            del pending[prompt_id]
            _finish_render(args, server, prompt_id, job, state, r, failed, submit)
        if pending:
            time.sleep(5)
    return failed


def _resubmit_stalled(prompt_id: str, r: dict, state, pending: dict, seen: dict, submit) -> None:
    """A cloud job still in a non-terminal state: cancel it and queue it again once it passed STALL_SECONDS (a restart
    doesn't use up a failure attempt); otherwise leave it queued and keep polling."""
    if r.get("restarts", 0) >= api.MAX_RESTARTS or not api.stalled(seen, prompt_id, state):
        return
    print(f"  [stage_{r['stage']}] still '{state}' after {api.STALL_SECONDS}s, resubmitting", flush=True)
    api.cancel(prompt_id)
    del pending[prompt_id]
    r["restarts"] = r.get("restarts", 0) + 1
    r["attempts"] -= 1  # a restart doesn't use up a failure attempt
    submit(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline_dir", type=Path)
    ap.add_argument("--motion-prompts", type=Path, required=True)
    ap.add_argument("--durations", type=Path, required=True)
    ap.add_argument("--template", type=Path, default=None,
                    help="MiniMax i2v base graph instead of video/templates/video_minimax_h3_i2v_template.json "
                         "(add LoRAs per stage with --loras)")
    ap.add_argument("--engines", type=Path, default=None,
                    help=f"JSON {{\"<stage>\": \"<engine>\"}} (engines: {', '.join(VIDEO_ENGINES)}; default {DEFAULT_ENGINE})")
    ap.add_argument("--loras", type=Path, default=None,
                    help="JSON {\"<stage>\": {\"<lora>\": strength, ...}} chained onto the template per stage "
                         f"(names: {', '.join(lora_registry.video_files())})")
    ap.add_argument("--audio", type=Path, default=None,
                    help="JSON {\"<stage>\": true} for stages whose clip should come with generated audio (uses the audio template)")
    ap.add_argument("--seeds", type=Path, default=None,
                    help="JSON {\"<stage>\": int} to pin a stage's noise seed (default: random per render, logged)")
    ap.add_argument("--rife-dir", type=Path, default=None,
                    help="Animate + RIFE: also interpolate each clip in the same cloud job and save it to <dir>/videos/ (project.py: <project>/interp)")
    ap.add_argument("--server", default=None,
                    help="render on this ComfyUI (the RunPod pod's URL, project.py animate --target runpod) instead of Comfy Cloud")
    ap.add_argument("--megapixels", type=float, default=None,
                    help="clip size via the template's ResolutionSelector (default: template's 0.4 = 512x768 at 2:3; 1.04 = 832x1248)")
    args = ap.parse_args()
    server = args.server.rstrip("/") if args.server else None

    custom = json.loads(args.template.read_text()) if args.template else None
    prompts = json.loads(args.motion_prompts.read_text())
    durations = json.loads(args.durations.read_text())
    stage_loras = json.loads(args.loras.read_text()) if args.loras else {}
    stage_seeds = json.loads(args.seeds.read_text()) if args.seeds else {}
    stage_audio = json.loads(args.audio.read_text()) if args.audio else {}
    stage_engines = json.loads(args.engines.read_text()) if args.engines else {}
    _check_local_only(stage_loras, stage_engines, server)

    videos_dir = args.pipeline_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Submit every missing clip up front, then poll and save each as it finishes.
    renders = _build_renders(args, prompts, durations, stage_loras, stage_seeds, stage_audio, stage_engines, custom, videos_dir, server)

    if server:  # a model or node the pod lacks fails here, before anything renders
        oi = json.loads(local._call(server, "/object_info"))
        problems = sorted({p for r in renders for p in local.check(r["graph"], server, CHECK_FIELDS, oi)})
        if problems:
            raise SystemExit(f"{server} can't run these graphs:\n  " + "\n  ".join(problems))

    failed = _poll_and_save(args, server, renders)
    if failed:
        raise SystemExit(f"Failed stages: {failed}")
    print("All stages animated.", flush=True)


if __name__ == "__main__":
    main()
