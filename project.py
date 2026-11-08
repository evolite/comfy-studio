#!/usr/bin/env python3
"""
Per-project workflow. Everything for a project lives in data/pipelines/<name>/; ALL settings + prompts live in data/pipelines/<name>/project.json ({"engine": ..., "settings": {...}, "stages": [...]}):
`settings` = run-wide engine tunables (upscale_factor, upscale_model, il_steps, ... -- keys in pipeline/graph_settings.py; the
template's value applies when a key is absent); any of them can also be set on a single stage, which wins.
each stage = id (stable unique int; stage_N.png/mp4), krea_scene, il_scene, `loras` (image LoRAs, {"name": strength}; names in the LoRA library), ... plus `motion` (i2v prompt) and optional `duration` (seconds, 1.0-10.0, snapped up to the i2v model's frame grid (MiniMax 17k+5, LTX 8k+1); the web UI offers lengths that fit the chosen music;
default: 2.0, 2.9 for the last; aim for 2-4 s), `video_loras` (extra i2v LoRAs chained after the turbo one, as
{"name": strength}, decimals ok, e.g. {"motion_lora": 1.0, "other_lora": 0.6}; names in the LoRA library) and
optional `audio` (true = the i2v clip gets a generated soundtrack), optional `voice` (a voice-library name filling the motion's
{voice} / {voice.delivery} / {voice.moans} / {voice.sound}; a top-level `voice` is the project default; optional `voices`
({"S2": name}, stage or project level) gives another speaker its own voice for its :S2 slots; pipeline/voice_registry.py), optional `video_seed` (pin the i2v noise seed; default is random per render, logged in _debug/i2v_stage_N/).
Top-level `video_megapixels` (i2v clip size; template default 0.4 = 512x768 at 2:3, 1.04 = 832x1248) and `video_engine`
(the i2v model, animate_pipeline.VIDEO_ENGINES: minimax (default) | ltx = LTX 2.3 / Sulphur 2; a stage's own `video_engine` wins).

  python3 project.py <name> new [--engine full|krea2|bgswap|moss] [--from data/stage_sets/x.json]
  python3 project.py <name> generate [N...] [--dry-run] [--preview|--local|--hires|--finish]   no N: generate every stage with no stage_N.png yet;
                                                     with N: regenerate those stages in place (paid);
                                                     --il: the cloud twin of --local (paid): IL + upscale + finish from preview/stage_N.png
                                                     -> IL/stage_N.png, no new Krea2
                                                     --preview: Krea2 base image only -> preview/stage_N.png; the next real
                                                     generate picks a fresh seed (--local/--hires work from the preview image itself);
                                                     --local: IL + upscale + finish on a local ComfyUI (--server) from
                                                     preview/stage_N.png -> IL/stage_N.png; stops if a preview is missing;
                                                     --local --new: new cloud Krea2 preview first, then the local pass
                                                     --preview --local: the Krea2 base on the local ComfyUI (fp8; models are unloaded first only when the last run used other ones or memory is tight)
                                                     --hires --local: SesquiLSR latent upscale (x`hires_scale`, default 1.5) + Krea2
                                                     re-diffusion (`hires_denoise`, default 0.05; stage LoRAs × `hires_lora_scale`, default 0.5; `hires_prompt` = extra text for this pass; `hires_vae`, default the preview's)
                                                     of preview/stage_N.png -> hires/stage_N.png;
                                                     local only (no stage N: every stage without a hires/stage_N.png)
                                                     --finish [--local] [--from hires|IL]: optional SeedVR2 step on an upscale result ->
                                                     finish/stage_N.png (a separate pass, so Krea2 and SeedVR never share VRAM;
                                                     `finish_resolution` short edge, default 1440). Source per stage: --from, else the
                                                     upscale ✓'d in the web UI, else hires/ if it exists, else IL/ (no N: every stage
                                                     with an upscale result and no finish/stage_N.png)
                                                     --face N..: IL FaceDetailer on IL/stage_N.png, replacing it (old -> its history); cloud (paid), --local free
                                                     (run after the IL + upscale pass; face_* / il_checkpoint settings, il_scene + IL loras)
  python3 project.py <name> animate [folder] [N...] [--rife] [--target cloud|runpod]  no N: i2v every stage without a clip;
                                                     with N: re-render those clips (paid). folder = a subfolder of the project
                                                     (finish, hires, IL, preview, ...): stills from <folder>/stage_N.png, clips into
                                                     <folder>/videos/; default = the project root (stage_N.png -> videos/)
                                                     --rife: the same cloud job also interpolates x2 -> interp/videos/ (step 5)
                                                     --target: Comfy Cloud or our RunPod pod (started on demand, bills per hour
                                                     while up, deletes itself when idle; pipeline/runpod_pod.py); default = ⚙ setting
  python3 project.py <name> interpolate [folder] [N...]   RIFE frame interpolation (x2: 24 -> 48 fps) of <folder>/videos/
                                                     stage_N.mp4 on the local ComfyUI (free) -> interp/videos/stage_N.mp4 (sources
                                                     untouched); no N: every clip without an interp/ result; needs the
                                                     ComfyUI-Frame-Interpolation node; cut it with `edit interp animate`
  python3 project.py <name> edit [folder] [animate|stills] [N...] [make_tiktok.py args]   beat-synced edit (free)
                                                     folder as for animate; `animate` = cut that folder's clips (errors if none),
                                                     `stills` = its stage_N.png; N... = only those stages, in that order;
                                                     mix: `animate ... --as-still 2 5 [--stills-from <dir>]` shows those stages' stills
                                                     edits land in <folder>/edits/ (a folder edit: <name>_<folder>.mp4)
  python3 project.py <name> status
  python3 project.py <name> show [N...]             JSON of everything about those stages (no N: all): the stage, its outputs per
                                                     folder, preview takes, past runs' _debug graph.json paths;
                                                     what an agent reads for a web-UI ref `@<name>#N`
"""
import argparse
import json
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "video"))
import animate_pipeline
import interpolate
import lora_registry
import voice_registry
import generate as full_engine
import graph_settings
import history
import image_loras
import run_pipeline
import runner
import runpod_pod
import steps
import takes
import targets
import timings
from stagegraph import check_stage_ids

# each i2v model renders on its own frame grid at 24fps (animate_pipeline.frames); a requested duration snaps up to the next grid point
MIN_DURATION, MAX_DURATION = 1.0, 10.0  # above 5.0 is untested on MiniMax


def video_engine(cfg, stage):
    return stage.get("video_engine") or cfg.get("video_engine") or animate_pipeline.DEFAULT_ENGINE
BLANK_STAGE = {"id": 0, "krea_scene": "", "il_scene": "", "motion": "", "video_loras": {}, "loras": {}, "face_detailer_enabled": True,
               "audio": True}  # new stages animate with sound
# model names (data, not code): `new_project` = what `new` writes into `settings` (il_checkpoint only where the engine has an IL
# pass), `moss_unet` = the MOSS engine's checkpoint, `choices` = the web UI's model dropdowns (cloud object_info doesn't list
# user uploads, so an upload is added there; the first krea_unet/il_checkpoint choice should be the new_project one).
# defaults/models.json, overridden per key by data/library/models.json (config.library_merged).
def model_library() -> dict:
    return config.library_merged("models.json")


MOSS_SETTINGS = {"krea_steps": 25, "krea_cfg": 2, "krea_sampler": "er_sde",
                 "krea_scheduler": "simple", "aspect_ratio": "2:3 (Portrait Photo)", "megapixels": 1}  # Dirty MOSS author's settings (+ moss_unet as krea_unet)
# what `new` writes into `settings` per engine (full: full_engine.TUNED overlaid with the dark-academia project's tuned values;
# TUNED itself stays the engine default for existing projects)
DEFAULT_SETTINGS = {"full": {**full_engine.TUNED, **{
    "krea_steps": 8, "krea_cfg": 1, "krea_sampler": "euler", "krea_scheduler": "simple",
    "aspect_ratio": "2:3 (Portrait Photo)", "megapixels": 1,
    "il_steps": 32, "il_cfg": 3.5, "il_sampler": "dpmpp_2m_sde", "il_scheduler": "karras", "il_denoise": 0.2,
    "hed_strength": 0.29, "normal_strength": 0.41, "canny_strength": 0.4,
    "upscale_model": "4x_NMKD-Siax_2d00k.pth", "upscale_steps": 36, "upscale_cfg": 3.5, "upscale_sampler": "dpmpp_2m_sde",
    "upscale_scheduler": "karras", "upscale_denoise": 0.1, "upscale_step_mode": "simple", "upscale_iterations": 2,
    "face_wildcard": "detailed face, natural skin texture, skin pores", "face_steps": 30, "face_cfg": 4.5, "face_denoise": 0.35,
    "sharpen": 0.8, "grain_density": 0.25, "grain_intensity": 0.11}}}
DEFAULT_SETTINGS["moss"] = MOSS_SETTINGS


def paths(name):
    d = config.PIPELINES_DIR / name
    return d, d / "project.json"


def load(p):
    return json.loads(p.read_text())


def load_stages(name):
    cfg = project(name)[1]
    stages = cfg["stages"]
    check_stage_ids(stages)
    for i, s in enumerate(stages, 1):
        image_loras.check_loras(s.get("loras", {}), f"stage {i}")
        animate_pipeline.engine_spec(video_engine(cfg, s))  # an unknown video_engine exits here
        animate_pipeline.check_loras(s.get("video_loras", {}), f"stage {i}")  # LoRA vs engine: checked when animating
    return stages


def save(p, data):
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def project(name):
    d, p = paths(name)
    if not p.exists():
        sys.exit(f"no project {name!r} (pipelines/{name}/project.json missing) -- run: project.py {name} new")
    return d, load(p)


def save_stages(name, stages):
    _, cfg = project(name)
    save(paths(name)[1], {**cfg, "stages": stages})


def by_id(stages):
    """{id: stage} in file order; a stage without `id` gets its 1-based position. Ids name the stage_N files
    and survive removing a stage."""
    return {s.get("id", pos): s for pos, s in enumerate(stages, 1)}


def _default_duration(i, last) -> float:
    """A stage without an explicit duration: 2 s, the closing shot a little longer (clips aim for 2-4 s)."""
    return 2.9 if i == last else 2.0


def stage_duration(stages, i, engine=animate_pipeline.DEFAULT_ENGINE):
    ids = by_id(stages)
    v = ids[i].get("duration") or _default_duration(i, next(reversed(ids)))
    if not isinstance(v, (int, float)) or not MIN_DURATION <= v <= MAX_DURATION:
        sys.exit(f"stage {i}: duration {v!r} must be a number of seconds between {MIN_DURATION} and {MAX_DURATION}")
    return {"duration_value": v, "tier_frames": animate_pipeline.frames(engine, v)}


def new_settings(engine):
    lib = model_library()
    models = {k: v for k, v in lib["new_project"].items() if k != "il_checkpoint" or engine not in ("krea2", "moss")}
    if engine == "moss":
        models["krea_unet"] = lib["moss_unet"]
    return {**models, **DEFAULT_SETTINGS.get(engine, {})}


def cmd_new(a):
    d, project_p = paths(a.name)
    if d.exists():
        sys.exit(f"{d} already exists")
    if a.__dict__["from"]:
        src = load(a.__dict__["from"])
        stages = [{"id": i, **{k: v for k, v in BLANK_STAGE.items() if k not in ("id", "krea_scene", "il_scene", "motion")},
                   **{k: v for k, v in s.items() if k not in ("id", "name")}} for i, s in enumerate(src, 1)]
    else:
        stages = [dict(BLANK_STAGE, id=i) for i in range(1, 2 if a.oneoff else 3)]
    for sub in ("videos", "_debug", "preview"):  # preview/ from the start: the board lists its takes
        (d / sub).mkdir(parents=True)
    save(project_p, {"engine": a.engine, **({"oneoff": True} if a.oneoff else {}), "settings": new_settings(a.engine), "stages": stages})
    print(f"created {d} ({len(stages)} stages, engine={a.engine})\nfill in {project_p}, then: project.py {a.name} generate")


def make_ctx(name):
    """What a step needs about a project: its folder, engine, run-wide settings and stages by id."""
    d, cfg = project(name)
    return SimpleNamespace(name=name, d=d, cfg=cfg, engine=run_pipeline.ENGINES[cfg["engine"]], engine_name=cfg["engine"],
                           stages=by_id(load_stages(name)),
                           common={"seed": cfg.get("seed", secrets.randbits(48)), **graph_settings.check_settings(cfg["engine"], cfg.get("settings", {}))})


def get_target(a):
    return targets.get("local", server=a.server) if a.local else targets.get("cloud")


def run(a, step, numbers, reroll=False, **extra):
    """Run a step on the target the flags chose; exit non-zero when any stage failed. `extra` goes onto the ctx (finish_from)."""
    ctx = make_ctx(a.name)
    vars(ctx).update(extra)
    for n in numbers:
        if n not in ctx.stages:
            sys.exit(f"stage {n} not in {list(ctx.stages)}")
    failed = runner.run_step(steps.STEPS[step], get_target(a), ctx, numbers, getattr(a, "dry_run", False), reroll,
                             takes=getattr(a, "takes", 1))
    if failed:
        sys.exit(f"\nstages {sorted(failed)} failed (rerun to retry)")


def ensure_previews(a, numbers):
    """For the passes that start from preview/stage_N.png: make any missing previews first (on the same target)."""
    d, _ = project(a.name)
    missing = [n for n in numbers if not (d / "preview" / f"stage_{n}.png").exists()]
    if missing:
        print(f"no preview for stages {missing}: making the Krea2 base first")
        generate(SimpleNamespace(name=a.name, dry_run=a.dry_run, preview=True, local=a.local, server=a.server), missing)


def _estimate_action(ctx, action, build, target, numbers, local) -> dict:
    """Seconds of one action over its stages: a graph per stage (None when that stage can't be built), plus the one-off
    model load a local run pays at its first stage (cold time minus warm)."""
    todo = [n for n in numbers if ctx.stages[n].get("motion", "").strip()] if action.startswith("animate") else numbers
    secs = []
    for i, n in enumerate(todo):
        try:
            g = build({**ctx.stages[n], "id": n})
            secs.append(timings.expected(timings.sig(g), i == 0 and local and not target.same_as_last(g)))
        # SystemExit is the builder's "this stage can't be built" signal (empty prompt, bad duration) -- the estimate
        # records None and keeps going; a real run validates before anything is spent
        except (KeyError, SystemExit):  # NOSONAR(S5754)
            secs.append(None)
    load = 0
    if todo and local:
        g = build({**ctx.stages[todo[0]], "id": todo[0]})
        cold, warm = timings.expected(timings.sig(g), True), timings.expected(timings.sig(g), False)
        load = round(cold - warm, 1) if cold and warm and cold > warm and not target.same_as_last(g) else 0
    return {"seconds": sum(x for x in secs if x), "known": sum(1 for x in secs if x), "of": len(todo),
            "parallel": not local, "load": load}


def estimate(name, numbers):
    """Expected processing seconds of each run over these stages, from the baselines in data/logs/timings.json (see pipeline/timings.py):
    {action: {"seconds": sum, "known": stages with a baseline, "of": len(numbers), "parallel": cloud jobs overlap}}. The first stage of a
    local run counts as cold (loads the models). `animate` is for the stages that have a motion prompt."""
    ctx = make_ctx(name)
    ctx.common["seed"] = 0
    all_stages = load_stages(name)
    numbers = [n for n in numbers if n in ctx.stages]
    def clip(s, rife=False):
        engine = video_engine(ctx.cfg, s)
        motion = voice_registry.expand(s.get("motion", ""), voice_registry.cast_of(s, ctx.cfg), f"stage {s['id']}")
        return animate_pipeline.build_graph(animate_pipeline.template_for(engine, s.get("audio")), "x", motion,
                                            stage_duration(all_stages, s["id"], engine)["duration_value"], s.get("video_loras", {}), None,
                                            ctx.cfg.get("video_megapixels"), rife, engine, bool(s.get("audio")))
    clip_rife = lambda s: clip(s, rife=True)  # noqa: E731
    builders = {"animate": (clip, None), "animate_rife": (clip_rife, None)}  # action -> (graph of a stage, target or None for the cloud-only clip render)
    for (step_name, target_name), (action, _) in steps.ACTION_IDS.items():
        step, target = steps.STEPS[step_name], targets.get(target_name)
        if step.supports(target_name, ctx.engine_name):
            builders[action] = (lambda s, step=step, target=target: target.prepare(step.graph(ctx, s, "x", target)), target)
    out = {}
    for action, (build, target) in builders.items():
        local = target is not None and target.lane == "local"
        out[action] = _estimate_action(ctx, action, build, target, numbers, local)
    return out


def generate(a, numbers, reroll=False, **extra):
    ctx = make_ctx(a.name)
    preview = getattr(a, "preview", False)
    for n in numbers:
        if n not in ctx.stages:
            sys.exit(f"stage {n} not in {list(ctx.stages)}")
        if not ctx.stages[n].get("krea_scene", "").strip():
            sys.exit(f"stage {n} has an empty krea_scene in {paths(a.name)[1]}")
    if ctx.engine_name == "full" and not preview:
        no_il = [n for n in numbers if not ctx.stages[n].get("il_scene", "").strip()]
        if no_il:
            sys.exit(f"stages {no_il}: empty `il_scene` in {paths(a.name)[1]} -- the full engine's IL pass needs a Danbooru tag list "
                     f"per stage (--preview only needs krea_scene)")
    if preview:
        step = "preview"
    elif getattr(a, "il", False) or a.local:
        step = "il"
    else:
        step = "main"
    run(a, step, numbers, reroll, **extra)


def _cmd_face(a):
    if a.preview or a.new or a.il:
        sys.exit("--face is its own pass (FaceDetailer on IL/stage_N.png); only --local goes with it")
    if not a.stage:
        sys.exit("--face needs stage numbers")
    return run(a, "face", a.stage)


def _check_il_args(a):
    if a.preview or a.new or a.local or a.hires or a.finish or a.src:
        sys.exit("--il is the cloud IL pass on the previews; don't combine it with --preview/--new/--local/--hires/--finish/--from "
                 "(--local is the free version)")
    if not a.stage:
        sys.exit("--il needs stage numbers")


def _cmd_hires(a, d):
    if a.preview or a.new or a.il:
        sys.exit("--hires is its own pass (SesquiLSR latent upscale of the previews); only --local goes with it")
    if not a.local:
        sys.exit("--hires runs on the local ComfyUI only (SesquiLSR isn't on Comfy Cloud): add --local")
    todo = a.stage or [i for i in by_id(load_stages(a.name)) if not (d / "hires" / f"stage_{i}.png").exists()]
    ensure_previews(a, todo)
    return run(a, "hires", todo)


def _have_upscale(d, src, i) -> bool:
    if src:
        return (d / src / f"stage_{i}.png").exists()
    return any((d / f / f"stage_{i}.png").exists() for f in steps.UPSCALES)


def _cmd_finish(a, d):
    if a.preview or a.new or a.il or a.hires:
        sys.exit("--finish is its own pass (SeedVR2 on an upscale result); only --local / --from go with it")
    todo = a.stage or [i for i in by_id(load_stages(a.name)) if _have_upscale(d, a.src, i) and not (d / "finish" / f"stage_{i}.png").exists()]
    if not todo:
        sys.exit("no stage has an upscale result without a finish/stage_N.png -- name stage numbers to redo some")
    return run(a, "finish", todo, finish_from=a.src)


def regen_local(a):
    """`generate [N..] --local --new`: fresh Krea2 preview on the cloud (cheap), then the free local IL pass -- one step for iterating.
    The new preview take becomes current automatically (even if the preview was ✓'d); a stage whose cloud preview failed keeps
    its old take and gets no local pass."""
    d, _ = project(a.name)
    numbers = a.stage or [i for i in by_id(load_stages(a.name)) if not (d / f"stage_{i}.png").exists()]
    if not numbers:
        sys.exit("every stage already has a stage_N.png -- name stage numbers to redo some")
    before = {n: takes.current_id(d, n) for n in numbers}
    opts = {"name": a.name, "dry_run": a.dry_run, "server": a.server}
    try:
        generate(SimpleNamespace(**opts, preview=True, local=False), numbers)
    # a take that landed before the failure still earns its local pass below
    except SystemExit as e:  # NOSONAR
        if a.dry_run or not any(takes.current_id(d, n) != before[n] for n in numbers):
            raise
        print(e)
    if a.dry_run:
        return
    ready = [n for n in numbers if takes.current_id(d, n) not in (None, before[n])]
    if ready:
        generate(SimpleNamespace(**opts, preview=False, local=True), ready, reroll=True)
    if len(ready) < len(numbers):
        sys.exit(f"\nno preview made for stages {[n for n in numbers if n not in ready]} (cloud step failed; rerun to retry)")


def cmd_generate(a):
    if a.target:
        a.local = a.target == "local"
    d, _ = project(a.name)
    if a.takes > 1 and not a.preview:
        sys.exit("--takes goes with --preview (several Krea2 previews per stage to choose from)")
    if a.face:
        return _cmd_face(a)
    if a.il:
        _check_il_args(a)
    elif a.hires:
        return _cmd_hires(a, d)
    elif a.finish:
        return _cmd_finish(a, d)
    elif a.src:
        sys.exit("--from goes with --finish")
    elif a.new:
        if not a.local:
            sys.exit("--new goes with --local (new cloud Krea2 preview, then the local pass)")
        return regen_local(a)
    if a.stage:
        return generate(a, a.stage, reroll=True)
    todo = [i for i in by_id(load_stages(a.name)) if not (d / f"stage_{i}.png").exists()]
    if not todo:
        sys.exit("every stage already has a stage_N.png -- name stage numbers to redo some")
    generate(a, todo)


def folder_and_stages(d, tokens):
    """[folder] [N..]: a leading subfolder of the project (seedvr, IL, ...) picks where stills/clips live; default the project root."""
    folder, nums = d, []
    for t in tokens:
        if t.isdigit():
            nums.append(int(t))
        elif folder == d and not nums and (d / t).is_dir() and not t.startswith("_"):
            folder = d / t
        else:
            sys.exit(f"{t!r}: not a stage number or a folder of {d.name} ({', '.join(p.name for p in sorted(d.iterdir()) if p.is_dir() and not p.name.startswith('_'))})")
    return folder, nums


def animate_cmd(a, src, only):
    """One sequential animate_pipeline.py run for every stage without a clip (concurrent i2v submissions
    trip a backend error). Per-stage LoRAs/seeds come from `video_loras` / `video_seed` in project.json."""
    d, cfg = project(a.name)
    stages = load_stages(a.name)
    ids = by_id(stages)
    pngs = sorted(int(p.stem[6:]) for p in src.glob("stage_*.png") if p.stem[6:].isdigit() and int(p.stem[6:]) in ids)  # a removed stage's leftover PNG is ignored
    unknown = [n for n in only if n not in pngs]
    if unknown:
        sys.exit(f"stages {unknown}: no {src.name}/stage_N.png (or not in project.json) to animate")
    todo = [n for n in pngs if (n in only if only else not (src / "videos" / f"stage_{n}.mp4").exists())]
    bad = [n for n in todo if not ids[n].get("motion", "").strip()]
    if bad:
        sys.exit(f"stages {bad}: empty `motion` in {paths(a.name)[1]}")
    (d / "_debug").mkdir(exist_ok=True)
    motion_p, dur_p, lora_p, seed_p, audio_p, engine_p = (d / "_debug" / f for f in ("motion_prompts.json", "durations.json", "loras.json", "seeds.json", "audio.json", "engines.json"))
    voices = voice_registry.load()["voices"]  # {voice...} slots filled here, so the pipeline (and _debug graphs) see the final text
    motions = {n: voice_registry.expand(ids[n]["motion"], voice_registry.cast_of(ids[n], cfg), f"stage {n}", voices) for n in todo}
    save(motion_p, {str(n): lora_registry.with_video_triggers(motions[n], ids[n].get("video_loras", {})) for n in todo})
    save(dur_p, {str(n): stage_duration(stages, n, video_engine(cfg, ids[n])) for n in todo})
    save(engine_p, {str(n): video_engine(cfg, ids[n]) for n in todo})
    save(lora_p, {str(n): ids[n].get("video_loras", {}) for n in todo})
    save(seed_p, {str(n): ids[n]["video_seed"] for n in todo if "video_seed" in ids[n]})
    save(audio_p, {str(n): True for n in todo if ids[n].get("audio")})
    return [sys.executable, str(ROOT / "video" / "animate_pipeline.py"), str(src), "--motion-prompts", str(motion_p),
            "--durations", str(dur_p), "--loras", str(lora_p), "--seeds", str(seed_p), "--audio", str(audio_p), "--engines", str(engine_p)] + (["--megapixels", str(cfg["video_megapixels"])] if cfg.get("video_megapixels") else []) \
        + (["--rife-dir", str(d / interpolate.FOLDER)] if a.rife else []) + (["--server", pod_server()] if a.target == "runpod" and todo else [])


def pod_server():
    """The RunPod pod's ComfyUI URL, starting the pod when none is up (minutes; the log says how long)."""
    try:
        return runpod_pod.ensure_up(lambda m: print(m, flush=True))
    except runpod_pod.PodError as e:
        sys.exit(f"RunPod: {e}")


def cmd_animate(a):
    d, _ = project(a.name)
    src, stages = folder_and_stages(d, a.args)
    cmd = animate_cmd(a, src, stages)  # first: a pod that fails to start leaves the current clips in place
    for n in stages:
        for mp4 in [src / "videos" / f"stage_{n}.mp4"] + ([d / interpolate.FOLDER / "videos" / f"stage_{n}.mp4"] if a.rife else []):
            history.archive(d, mp4)
    sys.exit(subprocess.call(cmd))


def cmd_recover(a):
    """Save cloud/pod clips whose animate run died before downloading them (e.g. a web-UI restart): every folder's
    _debug/i2v_pending/ records whose process is gone (video/animate_pipeline.py recover())."""
    d, _ = project(a.name)
    dirs = [p.parent.parent for p in d.glob(f"**/_debug/{animate_pipeline.PENDING}") if p.is_dir()]
    if not any(animate_pipeline.orphans(x) for x in dirs):
        print("nothing to recover", flush=True)
        return
    failed = sum(animate_pipeline.recover(x) for x in dirs)
    if failed:
        sys.exit(f"{failed} clip(s) had failed on the cloud or the pod, or the pod was gone")


def _interpolate_one(a, d, src, vids, dest, tmp, n) -> bool:
    """One clip through RIFE on the local ComfyUI; False when it failed (logged). A server that can't be reached exits."""
    out = tmp / f"stage_{n}.mp4"
    t0 = time.time()
    try:
        secs = interpolate.interpolate(vids / f"stage_{n}.mp4", out, a.server, f"{a.name} stage {n} interpolate")
    except TimeoutError as e:
        print(f"  [stage_{n}] FAILED: {e}", flush=True)
        return False
    except OSError as e:
        sys.exit(f"can't reach the local ComfyUI at {a.server} ({e}) -- start it, or pass --server")
    except RuntimeError as e:
        print(f"  [stage_{n}] FAILED: {e}", flush=True)
        return False
    dest.mkdir(parents=True, exist_ok=True)
    history.archive(d, dest / f"stage_{n}.mp4")
    shutil.move(out, dest / f"stage_{n}.mp4")
    (dest.parent / "_debug" / f"stage_{n}").mkdir(parents=True, exist_ok=True)
    save(dest.parent / "_debug" / f"stage_{n}" / "source.json", {"folder": src.name if src != d else ""})
    print(f"  [stage_{n}] done (waited {time.time() - t0:.0f}s" + (f", processed {secs:.0f}s" if secs else "") + f") -> {dest / f'stage_{n}.mp4'}", flush=True)
    return True


def cmd_interpolate(a):
    """RIFE x2 of a folder's clips (<folder>/videos/stage_N.mp4) into interp/videos/stage_N.mp4, one at a time on the local ComfyUI.
    The source clips stay as they are; no N: every clip without an interp/ result yet, with N: redo those (the old result goes into
    its history, pipeline/history.py). interp/_debug/stage_N/source.json records which folder each result came from."""
    d, _ = project(a.name)
    src, only = folder_and_stages(d, a.args)
    if src.name == interpolate.FOLDER:
        sys.exit(f"{interpolate.FOLDER}/ holds the results: name the folder whose clips to interpolate (the root, finish, hires, ...)")
    vids, dest = src / "videos", d / interpolate.FOLDER / "videos"
    have = sorted(int(p.stem[6:]) for p in vids.glob("stage_*.mp4") if p.stem[6:].isdigit())
    missing = [n for n in only if n not in have]
    if missing:
        sys.exit(f"stages {missing}: no {vids.relative_to(d)}/stage_N.mp4 -- animate them first")
    todo = only or [n for n in have if not (dest / f"stage_{n}.mp4").exists()]
    if not todo:
        sys.exit(f"nothing to interpolate in {vids.relative_to(d)}" if not have else 0)
    tmp = d / "_debug" / "interpolate"
    print(f"=== interpolate x{interpolate.MULTIPLIER} on local: {len(todo)} clip(s) from {vids.relative_to(d)} -> {dest} ===", flush=True)
    failed = [n for n in todo if not _interpolate_one(a, d, src, vids, dest, tmp, n)]
    print(f"\n{len(todo) - len(failed)}/{len(todo)} clips interpolated" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


def cmd_edit(a):
    d, _ = project(a.name)
    extra = [x for i, x in enumerate(a.extra) if not (i == 0 and x == "--")]
    n = next((i for i, x in enumerate(extra) if x.startswith("-")), len(extra))
    words, flags = extra[:n], extra[n:]
    mode = next((w for w in words if w in ("animate", "stills")), None)
    folder, stages = folder_and_stages(d, [w for w in words if w != mode])
    if mode == "animate" and not any((folder / "videos").glob("stage_*.mp4")):
        sys.exit(f"no clips in {folder.name}/videos -- run: project.py {a.name} animate {'' if folder == d else folder.name}")
    cmd = [sys.executable, str(ROOT / "video" / "make_tiktok.py"), str(folder)]
    if mode == "stills":
        cmd.append("--stills")
    if stages:
        cmd += ["--order", *map(str, stages)]
    if folder != d and "--out-name" not in flags:
        cmd += ["--out-name", f"{a.name}_{folder.name}.mp4"]  # lands in <folder>/edits/
    sys.exit(subprocess.call(cmd + flags))


def cmd_status(a):
    d, cfg = project(a.name)
    stages = load_stages(a.name)
    print(f"{a.name} | engine={cfg['engine']} | video={cfg.get('video_engine', animate_pipeline.DEFAULT_ENGINE)} | {len(stages)} stages\n  #  png  mp4  motion  dur  video_loras")
    for i, s in by_id(stages).items():
        m = "yes" if s.get("motion", "").strip() else "-"
        print(f"{i:>3}  {'yes' if (d / f'stage_{i}.png').exists() else '-':<4} {'yes' if (d / 'videos' / f'stage_{i}.mp4').exists() else '-':<4} "
              f"{m:<7} {stage_duration(stages, i)['duration_value']:<4} "
              f"{' '.join(f'{k}:{v:g}' for k, v in s.get('video_loras', {}).items()) or '(none)'}"
              + (f"  [{s['video_engine']}]" if s.get("video_engine") else ""))


def _voice_info(s, cfg):
    """The stage's voice as `show` prints it: name, where it comes from, the other speakers and the expanded motion."""
    cast = voice_registry.cast_of(s, cfg)
    if not cast:
        return None
    voice = {"name": voice_registry.voice_of(s, cfg), "from": "stage" if s.get("voice") else "project"}
    if set(cast) - {"S1"}:
        voice["speakers"] = cast
    if voice_registry.PLACEHOLDER.search(s.get("motion", "")):
        bad = voice_registry.problems(s["motion"], cast)
        voice["expanded_motion"] = "; ".join(bad) if bad else voice_registry.expand(s["motion"], cast)
    return voice


def _stage_outputs(d, folders, n):
    """{folder: {"still"/"clip": {path, mtime}}} of the stage's files, the project root as "root"."""
    outputs = {}
    for f in folders:
        got = {k: {"path": str(x.relative_to(d)), "mtime": int(x.stat().st_mtime)}
               for k, x in (("still", f / f"stage_{n}.png"), ("clip", f / "videos" / f"stage_{n}.mp4")) if x.exists()}
        if got:
            outputs[str(f.relative_to(d)) if f != d else "root"] = got
    return outputs


def _show_stage(d, cfg, s, n, folders, graphs):
    own = re.compile(rf"(?:^|_)stage_{n}(?:_|$)")
    off = lora_registry.disabled_in(s.get("loras"), "image") + lora_registry.disabled_in(s.get("video_loras"), "video")
    return {"stage": s, "video_engine": video_engine(cfg, s), "voice": _voice_info(s, cfg),
            **({"switched_off_loras": off} if off else {}),  # retired on the LoRAs page: swap them (pipeline/lora_registry.py)
            "outputs": _stage_outputs(d, folders, n),
            "takes": [{k: t[k] for k in ("id", "current", "source", "delta")} for t in takes.listing(d, n)],
            "debug_graphs": [str(g.relative_to(d)) for g in graphs if own.search(g.parent.name)][:10]}  # newest first


def cmd_show(a):
    """Read-only: no LoRA/engine validation, so a stage with a broken name can still be looked up and fixed."""
    d, cfg = project(a.name)
    ids = by_id(cfg["stages"])
    missing = [n for n in a.stage if n not in ids]
    if missing:
        sys.exit(f"{a.name}: no stage {', '.join(map(str, missing))} (ids: {', '.join(map(str, ids))})")
    folders = [d] + sorted(f for f in d.iterdir() if f.is_dir() and not f.name.startswith("_"))
    graphs = sorted(d.glob("**/_debug/*/graph.json"), key=lambda g: g.stat().st_mtime, reverse=True)
    out = {"name": a.name, "dir": str(d), "ref": f"@{a.name}",
           **{k: cfg.get(k) for k in ("engine", "video_engine", "voice", "oneoff", "music", "settings") if k in cfg},
           "stage_ids": list(ids), "stages": {n: _show_stage(d, cfg, ids[n], n, folders, graphs) for n in (a.stage or ids)}}
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(cmd, fn, **kw):
        p = sub.add_parser(cmd, **kw)
        p.set_defaults(fn=fn)
        return p

    p = add("show", cmd_show)
    p.add_argument("stage", type=int, nargs="*")
    p = add("new", cmd_new)
    p.add_argument("--engine", choices=run_pipeline.ENGINES, default="full")
    p.add_argument("--oneoff", action="store_true", help="single-stage project shown in the web UI's One-offs tab")
    p.add_argument("--from", type=Path, help="import an existing stage set JSON")
    p = add("generate", cmd_generate)
    p.add_argument("stage", type=int, nargs="*")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--local", action="store_true", help="run the IL/upscale/finish stages on a local ComfyUI from preview/stage_N.png (Krea2 step stays on the cloud via --preview)")
    p.add_argument("--server", default=config.LOCAL_COMFY_URL, help="local ComfyUI address for --local")
    p.add_argument("--target", choices=targets.NAMES, help="where to run it; --target local is the same as --local")
    p.add_argument("--new", action="store_true", help="with --local: make a fresh cloud Krea2 preview first (it becomes the current take; the old one stays a take), then run the local pass")
    p.add_argument("--preview", action="store_true", help="Krea2 base render only, into preview/ (cheap pose check); every render is kept as a take")
    p.add_argument("--takes", type=int, default=1, choices=range(1, 9), metavar="K",
                   help="with --preview: render K takes per stage (fresh seed each, round-robin); the first becomes current unless the preview is ✓'d")
    p.add_argument("--il", action="store_true", help="IL + upscale + finish in the cloud (paid) from preview/stage_N.png -> IL/stage_N.png, no new Krea2; the cloud twin of --local")
    p.add_argument("--hires", action="store_true", help="with --local: SesquiLSR latent upscale + low-denoise Krea2 pass on preview/stage_N.png -> hires/stage_N.png (makes missing previews first)")
    p.add_argument("--finish", action="store_true", help="optional SeedVR2 step on an upscale result (hires/ or IL/) -> finish/stage_N.png (`finish_resolution` short edge, default 1440); with --local, on the local ComfyUI")
    p.add_argument("--from", dest="src", choices=steps.UPSCALES, help="with --finish: the upscale to start from (default per stage: the ✓'d one, else hires/, else IL/)")
    p.add_argument("--face", action="store_true", help="IL FaceDetailer (cloud, paid; --local = free) on IL/stage_N.png, replacing it; needs stage numbers")
    p = add("animate", cmd_animate)
    p.add_argument("args", nargs="*", metavar="[folder] [N..]")
    p.add_argument("--rife", action="store_true", help="Animate + RIFE: the same cloud job also interpolates each clip x2 (48 fps) into interp/videos/")
    p.add_argument("--target", choices=("cloud", "runpod"), default=config.animate_target(),
                   help="Comfy Cloud, or our RunPod pod (started on demand; pipeline/runpod_pod.py); default: the ⚙ animate target")
    add("recover", cmd_recover, help="save cloud clips whose animate run died before downloading them")
    p = add("interpolate", cmd_interpolate)
    p.add_argument("args", nargs="*", metavar="[folder] [N..]")
    p.add_argument("--server", default=config.LOCAL_COMFY_URL, help="local ComfyUI address")
    p = add("edit", cmd_edit)
    p.add_argument("extra", nargs=argparse.REMAINDER)
    add("status", cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
