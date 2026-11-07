"""The passes a project goes through, defined once. A step builds its graph in cloud-canonical form (what the engines emit) and
says where its input comes from and where its result goes; which target runs it (Comfy Cloud or a local ComfyUI) is the runner's
business (runner.py), not the step's.

    preview   Krea2 base render only                                  -> preview/
    il        IL + upscale + finish on a Krea2 preview                 -> IL/
    face      IL FaceDetailer on an IL image (replaces it)             -> IL/ (rendered in _debug/face first)
    hires     latent upscale + Krea2 re-diffusion of a preview (local) -> hires/
    finish    SeedVR2 upscale of an upscale result (hires/ or IL/)     -> finish/
    main      the full engine, start to finish                         -> project root
"""
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_face as face_engine  # noqa: E402
import graph_settings  # noqa: E402
import history  # noqa: E402
import image_loras  # noqa: E402
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
    """The full-engine graph with its Krea2 nodes (30:*, 49) replaced by LoadImage(image_name), so IL + upscale + finish
    start from an existing still."""
    g = {k: v for k, v in graph.items() if not k.startswith("30:") and k != "49"}
    g["29"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
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


STEPS = {s.name: s for s in [
    Step("main", "", lambda c, s, i, t: stage_graph(c.engine, s, c.common),
         lambda c: (c.engine.FINAL_IMAGE_NODE, c.engine.NODE_LABELS), targets=("cloud",)),
    Step("preview", "preview", lambda c, s, i, t: stage_graph(preview_engine(c.engine), s, c.common),
         lambda c: ("436", {"436": "final"}), suffix="krea2", takes=True),
    Step("il", "IL", lambda c, s, i, t: replace_krea(stage_graph(c.engine, s, c.common), i),
         lambda c: (c.engine.FINAL_IMAGE_NODE, c.engine.NODE_LABELS), needs="preview", engines=("full",),
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
