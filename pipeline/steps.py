"""The passes a project goes through, defined once. A step builds its graph in cloud-canonical form (what the engines emit) and
says where its input comes from and where its result goes; which target runs it (Comfy Cloud or a local ComfyUI) is the runner's
business (runner.py), not the step's.

    preview   Krea2 base render only                                  -> preview/
    il        IL + upscale + finish on a Krea2 preview (bgswap: its IL background + composite) -> IL/
    face      IL FaceDetailer on an IL image (replaces it)             -> IL/ (rendered in _debug/face first)
    hires     latent upscale + Krea2 re-diffusion of a preview (local) -> hires/
    finish    SeedVR2 upscale of an upscale result (hires/ or IL/)     -> finish/
    main      the full engine, start to finish                         -> project root
    animate   i2v clip of a folder's still + its RIFE x2 (cloud or the pod) -> <folder>/videos/ (24 fps), interp/videos/ (48 fps)

animate reads the folder from ctx.clip_from ("" = the project root; project.py sets it), and the stage's engine and duration
through ctx.video_engine(stage) / ctx.duration_of(n, engine).
"""
import json
import secrets
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "video"))
import animate_pipeline as clips  # noqa: E402  # the i2v graph library (engines, templates, LoRAs, RIFE, keyframes)
import generate_face as face_engine  # noqa: E402
import graph_settings  # noqa: E402
import history  # noqa: E402
import image_loras  # noqa: E402
import interpolate as rife  # noqa: E402
import lora_registry  # noqa: E402
import voice_registry  # noqa: E402
from stagegraph import stage_graph  # noqa: E402


# --- graph pieces shared by steps -------------------------------------------------------------------------------
def preview_graph(graph):
    """Just the Krea2 base render: every node the Krea2 VAEDecode (30:8) depends on, saved to node 436."""
    keep, todo = set(), ["30:8"]
    while todo:
        nid = todo.pop()
        if nid not in keep:
            keep.add(nid)
            todo += [v[0] for v in graph[nid]["inputs"].values() if isinstance(v, list)]
    g = {nid: graph[nid] for nid in keep}
    g["436"] = {"class_type": "SaveImage", "inputs": {"images": ["30:8", 0], "filename_prefix": "Preview/"}}
    return g


def preview_engine(engine):
    return SimpleNamespace(__name__=f"{engine.__name__} (krea2 preview)", NODE_LABELS={"436": "final"}, FINAL_IMAGE_NODE="436",
                           build_graph=lambda **kw: preview_graph(engine.build_graph(**{"il_scene": "", **kw})))


def replace_krea(graph: dict, image_name: str) -> dict:
    """The graph with its Krea2 render (30:8 and everything it depends on) replaced by LoadImage(image_name), so the rest
    (full: IL + upscale + finish; bgswap: matte, IL background, composite, upscale) starts from an existing still. A size
    node the rest still reads (bgswap's 49) becomes that image's size."""
    gen = set(preview_graph(graph)) - {"436"}
    g = {k: v for k, v in graph.items() if k not in gen}
    g["29"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    used = {v[0] for node in g.values() for v in node["inputs"].values() if isinstance(v, list)}
    if "49" in gen and "49" in used:
        g["49"] = {"class_type": "GetImageSize+", "inputs": {"image": ["29", 0]}}  # width, height: the same outputs as 49
    for node in g.values():
        for k, v in node["inputs"].items():
            if isinstance(v, list) and v[0] == "30:8":
                node["inputs"][k] = ["29", 0]
    return g


def seedvr_graph(image_name, resolution=None, model=None, color=None) -> dict:
    kw = {k: v for k, v in {"model": model, "color": color}.items() if v}
    return {"1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            **graph_settings.seedvr_nodes(["1", 0], resolution, **kw),
            "436": {"class_type": "SaveImage", "inputs": {"images": ["seedvr:up", 0], "filename_prefix": "SeedVR/"}}}


# --- the steps ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    name: str
    folder: str                          # result folder under the project ("" = the project root)
    graph: Callable                      # (ctx, stage, image, target) -> cloud-canonical graph
    final: Callable                      # (ctx) -> (final node id, node labels)
    needs: Optional[str | Callable] = None  # folder whose stage_N.png this step starts from, or (ctx, n) -> that folder
    needs_hint: str = ""
    targets: tuple = ("cloud", "local")
    engines: Optional[tuple] = None      # empty = every engine runs it
    suffix: str = ""                     # added to the job label (the local dashboard shows "project stage N <suffix>")
    after: Optional[Callable] = None     # (ctx, ok_numbers, out_dir) once the stages are rendered
    takes: bool = False                  # keep every render as a take of the stage, one of them current (pipeline/takes.py)
    input: str = "stage_{n}.png"         # the file in the `needs` folder it starts from
    final_for: Optional[Callable] = None  # (ctx, stage) -> final node id, when it depends on the stage (the video engine)
    keep: tuple = ()                     # output nodes saved besides the final one
    debug_prefix: str = ""               # _debug/<prefix>stage_N (animate's i2v_ keeps clear of the still's _debug/stage_N)
    save: Optional[Callable] = None      # (ctx, job, out) -> saved path; default: out/stage_N.png (or a take)

    def supports(self, target: str, engine: str) -> bool:
        return target in self.targets and (self.engines is None or engine in self.engines)


def _replace_il(ctx, ok, out):
    """FaceDetailer renders into _debug/face first, so a failed stage keeps its IL image; the old one goes into its history."""
    for n in ok:
        if (out / f"stage_{n}.png").exists():
            history.archive(ctx.d, ctx.d / "IL" / f"stage_{n}.png")
            shutil.copyfile(out / f"stage_{n}.png", ctx.d / "IL" / f"stage_{n}.png")


def _hires(ctx, stage, image, target):
    """The stage's image LoRAs (registry defaults included) on the re-diffusion, scaled by `hires_lora_scale` (stage or project, default 0.5:
    enough to keep their content, less skin texture; 1 = full strength, 0 = none).
    `hires_prompt` (stage or project) is appended to the stage's krea_scene for this pass only, e.g. "smooth flawless skin"."""
    k = stage.get("hires_lora_scale", ctx.common.get("hires_lora_scale", 0.5))
    on = {n: v * k for n, v in image_loras.resolve(stage.get("loras")).items()}
    extra = (stage.get("hires_prompt") or ctx.common.get("hires_prompt") or "").strip()
    scene = f"{stage.get('krea_scene', '').rstrip()} {extra}".strip() if extra else stage.get("krea_scene", "")
    g = stage_graph(preview_engine(ctx.engine), {**stage, "krea_scene": scene, "loras": {**dict.fromkeys(image_loras.IMAGE_LORAS, 0), **on}}, ctx.common)
    return graph_settings.hires_graph(g, image, stage.get("hires_scale", ctx.common.get("hires_scale")),
                                      stage.get("hires_denoise", ctx.common.get("hires_denoise")),
                                      stage.get("hires_vae", ctx.common.get("hires_vae")))


UPSCALES = ("hires", "IL")  # the results a finish can start from, in order of preference


def finish_source(ctx, n):
    """Where stage n's finish starts: `ctx.finish_from` (project.py --from), else the upscale ✓'d in the web UI, else hires/ if it
    exists, else IL/."""
    if getattr(ctx, "finish_from", None):
        return ctx.finish_from
    done = ctx.stages[n].get("done")
    picked = [f for f in UPSCALES if isinstance(done, list) and f in done]
    have = [f for f in UPSCALES if (ctx.d / f / f"stage_{n}.png").exists()]
    return next((f for f in picked if f in have), have[0] if have else UPSCALES[0])


def il_source(ctx, n):
    """Where stage n's IL pass starts: the Krea2 preview; bgswap takes the latent upscale (hires/) when there is one, so the
    pasted-back people keep its detail (set `upscale_factor` 1 to keep that size)."""
    if ctx.engine.__name__.endswith("bgswap") and (ctx.d / "hires" / f"stage_{n}.png").exists():
        return "hires"
    return "preview"


# --- animate / interpolate ---------------------------------------------------------------------------------------
def _clip_from(ctx, n=None):
    return getattr(ctx, "clip_from", "")


def _keyframes(ctx, stage, engine, duration, image, target):
    """The stage's `keyframes` ([{"stage": k, "at": seconds}]): those stills (same folder) uploaded, at their frames."""
    src = ctx.d / _clip_from(ctx)
    pngs = []
    for kf in stage["keyframes"]:
        png = src / f"stage_{kf['stage']}.png"
        if kf["stage"] not in ctx.stages or not png.exists():
            sys.exit(f"stage {stage['id']}: keyframe stage {kf['stage']} has no {src.name}/stage_{kf['stage']}.png")
        pngs.append(png)
    at = clips.keyframe_frames(engine, duration, [k.get("at") for k in stage["keyframes"]])
    names = [image if image == "x" else target.upload(p, f"kf_{ctx.name}_{p.name}") for p in pngs]
    return list(zip(names, at))


def _animate_graph(ctx, stage, image, target):
    """The stage's clip, built from project.json as it is now: motion (voices filled, LoRA triggers added), duration,
    `video_loras`, `video_seed` (else random), audio, engine, keyframes, the project's `video_megapixels`."""
    n, engine = stage["id"], ctx.video_engine(stage)
    loras = stage.get("video_loras", {})
    clips.check_loras(loras, f"stage {n}", engine)
    if target.name == "cloud":  # a local-only i2v LoRA would fail on the cloud after the credits are spent
        files, local_only = lora_registry.video_files(), lora_registry.local_only_files()
        bad = [k for k, v in loras.items() if v and files.get(k, k) in local_only]
        if bad:
            sys.exit(f"stage {n}: video LoRA {bad[0]!r} is marked local-only (cloud: false in loras.json) and this run is on "
                     "Comfy Cloud: upload it and tick 'cloud' on the LoRAs page, animate on the pod, or set it to 0")
    motion = voice_registry.expand(stage.get("motion", ""), voice_registry.cast_of(stage, ctx.cfg), f"stage {n}")
    motion = lora_registry.with_video_triggers(motion, loras)
    duration = ctx.duration_of(n, engine)
    audio = bool(stage.get("audio"))
    kfs = _keyframes(ctx, stage, engine, duration, image, target) if stage.get("keyframes") else None
    if kfs:
        motion = clips.keyframe_refs(motion, [f for _, f in kfs], clips.engine_spec(engine)["fps"])
    return clips.build_graph(clips.template_for(engine, audio), image, motion, duration, loras,
                             stage.get("video_seed", secrets.randbits(32)), ctx.cfg.get("video_megapixels"), rife=True,
                             engine=engine, audio=audio, aspect=clips.aspect_of(ctx.d / _clip_from(ctx) / f"stage_{n}.png"),
                             keyframes=kfs)


def _save_video(ctx, src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    history.archive(ctx.d, dest)  # the clip it replaces stays restorable
    shutil.copyfile(src, dest)
    return dest


def _note_rife_source(ctx, n):
    """interp/_debug/stage_N/source.json: which folder's clip the interpolated one was made from (stale-clip check)."""
    d = ctx.d / rife.FOLDER / "_debug" / f"stage_{n}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "source.json").write_text(json.dumps({"folder": _clip_from(ctx)}, indent=2) + "\n")


def _save_clip(ctx, job, out):
    """The clip -> <folder>/videos/stage_N.mp4 (+ its render settings in _debug), its 48 fps RIFE one -> interp/videos/."""
    engine = ctx.video_engine(job.stage)
    e = clips.engine_spec(engine)
    files = dict(job.outputs)
    seed_node, seed_field = e["seed"]
    (job.debug_dir / "render_settings.json").write_text(json.dumps(
        {"engine": engine, "loras": job.stage.get("video_loras", {}), "seed": job.graph[seed_node]["inputs"][seed_field],
         "audio": bool(job.stage.get("audio")), "rife": True}, indent=2))
    if clips.RIFE_SAVE not in files:
        raise RuntimeError(f"no RIFE clip among the outputs -- check {job.debug_dir}")
    dest = _save_video(ctx, job.final, out / "videos" / f"stage_{job.n}.mp4")
    rdest = _save_video(ctx, files[clips.RIFE_SAVE], ctx.d / rife.FOLDER / "videos" / f"stage_{job.n}.mp4")
    _note_rife_source(ctx, job.n)
    print(f"  [{job.tag}] -> {rdest} (RIFE x{rife.MULTIPLIER})", flush=True)
    return dest


STEPS = {s.name: s for s in [
    Step("main", "", lambda c, s, i, t: stage_graph(c.engine, s, c.common),
         lambda c: (c.engine.FINAL_IMAGE_NODE, c.engine.NODE_LABELS), targets=("cloud",)),
    Step("preview", "preview", lambda c, s, i, t: stage_graph(preview_engine(c.engine), s, c.common),
         lambda c: ("436", {"436": "final"}), suffix="krea2", takes=True),
    Step("il", "IL", lambda c, s, i, t: replace_krea(stage_graph(c.engine, s, c.common), i),
         lambda c: (c.engine.FINAL_IMAGE_NODE, c.engine.NODE_LABELS), needs=il_source, engines=("full", "bgswap"),
         needs_hint="make the Krea2 base first: project.py {name} generate {stages} --preview"),
    Step("face", "_debug/face", lambda c, s, i, t: stage_graph(face_engine, {**s, "_image": i}, c.common),
         lambda c: (face_engine.FINAL_IMAGE_NODE, face_engine.NODE_LABELS), needs="IL", suffix="face",
         needs_hint="run the IL + upscale pass first", after=_replace_il),
    Step("hires", "hires", _hires, lambda c: ("436", {"436": "final"}), needs="preview", targets=("local",), suffix="hires",
         engines=("full", "krea2", "bgswap"),  # SesquiLSR + Krea2 re-diffusion: Krea2 latents only
         needs_hint="make the Krea2 base first: project.py {name} generate {stages} --preview"),
    Step("finish", "finish", lambda c, s, i, t: seedvr_graph(i, s.get("finish_resolution", c.common.get("finish_resolution")) or 1440,
                                                          t.params.get("seedvr_model"), t.params.get("seedvr_color")),
         lambda c: ("436", {"436": "final"}), needs=finish_source, suffix="finish",
         needs_hint="run an upscale first (IL: project.py {name} generate {stages} --local; latent: ... --hires --local)"),
    Step("animate", _clip_from, _animate_graph, lambda c: (clips.engine_spec(clips.DEFAULT_ENGINE)["save"], {}), needs=_clip_from,
         targets=("cloud", "runpod"), final_for=lambda c, s: clips.engine_spec(c.video_engine(s))["save"], keep=(clips.RIFE_SAVE,),
         debug_prefix="i2v_", suffix="animate", save=_save_clip,
         needs_hint="no still to animate: make it first, or name the folder it is in"),
]}

# (step, target) -> (the web UI's action id, `project.py generate` flags); jobs.ACTIONS, projects.PREVIEW_GATED and project.estimate
# are generated from this
ACTION_IDS = {
    ("main", "cloud"): ("generate", []),
    ("preview", "cloud"): ("preview", ["--preview"]), ("preview", "local"): ("preview_local", ["--preview", "--local"]),
    ("il", "cloud"): ("il_cloud", ["--il"]), ("il", "local"): ("local", ["--local"]),
    ("face", "cloud"): ("face", ["--face"]), ("face", "local"): ("face_local", ["--face", "--local"]),
    ("hires", "local"): ("hires_local", ["--hires", "--local"]),
    ("finish", "cloud"): ("finish", ["--finish"]), ("finish", "local"): ("finish_local", ["--finish", "--local"]),
}
