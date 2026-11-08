"""Comfy Studio API server (FastAPI). Run: webui/.venv/bin/uvicorn app:app --app-dir webui/server --port 8191"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import sys
import urllib.request
from pathlib import Path

from fastapi import Body, FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import loras
import voices
import projects as P
import comfy_watch
import config
import jobs
from jobs import ACTIONS, edit_timing, manager
import run_pipeline  # noqa: E402  # pipeline/ is on sys.path via projects
import civitai_meta  # noqa: E402
import runpod_pod  # noqa: E402
from targets.models import local_models  # noqa: E402

ROOT = P.ROOT
DIST = ROOT / "webui" / "ui" / "dist"
AUDIO_DIR = config.AUDIO_DIR
LOCAL_SERVER = config.LOCAL_COMFY_URL

app = FastAPI(title="Comfy Studio")


@app.on_event("startup")
def recover_cloud_clips():
    manager.recover_orphans()
    comfy_watch.start()


@app.exception_handler(P.ApiError)
async def api_error(_, e: P.ApiError):
    return JSONResponse({"error": str(e)}, status_code=e.status)


# --- registry / health -------------------------------------------------------------------------------
def template_defaults():
    """{engine: {key: value}}: what each tunable is in the engine's template, i.e. what a run uses while the key is unset."""
    out = {}
    for eng, (table, _) in P.graph_settings.ENGINE_KEYS.items():
        graph = run_pipeline.ENGINES[eng].load_template()
        vals = {}
        for key, targets in table.items():
            node, field = (targets[0] if isinstance(targets, list) else targets)
            if field in graph.get(node, {}).get("inputs", {}) and not isinstance(graph[node]["inputs"][field], list):
                vals[key] = graph[node]["inputs"][field]
        code = getattr(run_pipeline.ENGINES[eng], "DEFAULTS", {})  # full: generate.DEFAULTS is laid over the template at build time
        out[eng] = {k: round(v, 6) if isinstance(v, float) else v for k, v in {**vals, **{k: v for k, v in code.items() if k in table}}.items()}
    return out


@app.get("/api/registry")
def registry():
    keys = {eng: sorted(set(t) | set(sp)) for eng, (t, sp) in P.graph_settings.ENGINE_KEYS.items()}
    sys.path.insert(0, str(ROOT))
    import project as proj  # noqa: E402  # project.py; models + default settings per engine
    return {
        "settings_keys": keys,
        "default_settings": {e: proj.new_settings(e) for e in P.graph_settings.ENGINE_KEYS},
        "model_choices": proj.model_library()["choices"],
        "default_beats": config.DEFAULT_BEATS,
        "template_defaults": template_defaults(),
        "local_only_unets": sorted(local_models()["local_only_unets"] + local_models().get("local_only_vaes", [])),  # marks "(local only)" in the model selects
        "tiers": P.TIERS,
        "video_engines": {k: v["label"] for k, v in P.animate_pipeline.VIDEO_ENGINES.items()},
        "image_lora_fields": P.image_loras.ENGINE_FIELDS,
        "actions": {k: {"cost": v[1], "lane": v[2]} for k, v in ACTIONS.items()},
    }


_credits = {"t": 0.0, "v": None}


# --- LoRA library (loras.json) ----------------------------------------------------------------------------
@app.get("/api/loras")
def loras_library():
    return loras.library()


@app.put("/api/loras/{kind}/{name}")
def loras_put(kind: str, name: str, spec: dict):
    return loras.put(kind, name, spec)


@app.delete("/api/loras/{kind}/{name}")
def loras_delete(kind: str, name: str, strip: bool = False):
    return {"ok": True, "stripped": loras.delete(kind, name, strip)}


@app.put("/api/lora-combos/{name}")
def lora_combo_put(name: str, spec: dict):
    return loras.put_combo(name, spec)


@app.delete("/api/lora-combos/{name}")
def lora_combo_delete(name: str):
    loras.delete_combo(name)
    return {"ok": True}


# --- voice library (voices.json) ---------------------------------------------------------------------------
@app.get("/api/voices")
def voices_library():
    return voices.library()


@app.put("/api/voices/{name}")
def voices_put(name: str, spec: dict):
    return voices.put(name, spec)


@app.delete("/api/voices/{name}")
def voices_delete(name: str):
    voices.delete(name)
    return {"ok": True}


@app.post("/api/voices/{name}/sample")
def voices_make_sample(name: str):
    """Create / refresh the voice's stage in the samples project; the client then animates it (Animate smooth)."""
    return voices.make_sample(name)


@app.post("/api/voices/from-stage")
def voices_from_stage(body: dict = Body(...)):
    return voices.from_stage(body["project"], int(body["stage"]), body["name"], body.get("folder") or "")


@app.put("/api/projects/{name}/voice")
def project_voice(name: str, body: dict = Body(...)):
    P.put_voice(name, body.get("voice"))
    return {}


@app.post("/api/loras/ignore")
def loras_ignore(body: dict):
    return {"ignore": loras.ignore(body["file"], bool(body.get("on", True)))}


@app.get("/api/credits")
def credits(refresh: bool = False):
    """Comfy Cloud balance (GET /api/billing/balance with the account key), cached 20s. The API labels the fields *_micros but the
    values read as plain credit units, so they're passed through untouched."""
    import time
    import comfy_api
    if refresh or time.time() - _credits["t"] > 20 or _credits["v"] is None:
        try:
            _credits["v"] = comfy_api.get("/api/billing/balance")
            _credits["t"] = time.time()
        # no API key: the client's sys.exit becomes an HTTP error here -- re-raising SystemExit would kill the server
        except SystemExit as e:  # NOSONAR(S5754)
            raise P.ApiError(str(e), 503) from e
        except Exception as e:  # noqa: BLE001
            raise P.ApiError(f"Comfy Cloud balance unavailable: {e}", 502)
    return _credits["v"]


@app.get("/api/health")
def health():
    try:
        urllib.request.urlopen(LOCAL_SERVER + "/system_stats", timeout=1.5).read()
        local = True
    except Exception:  # noqa: BLE001
        local = False
    return {"local": local, "local_server": LOCAL_SERVER, "cloud_key": config.cloud_key_present()}


@app.post("/api/local/free")
def local_free(body: dict):
    """Manual memory control on the local ComfyUI: `unload_models` drops loaded models from VRAM, `free_memory` clears the cache."""
    flags = {k: True for k in ("unload_models", "free_memory") if body.get(k)}
    if not flags:
        raise P.ApiError("pick unload_models and/or free_memory")
    req = urllib.request.Request(LOCAL_SERVER + "/free", data=json.dumps(flags).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:  # noqa: BLE001
        raise P.ApiError(f"local ComfyUI not reachable: {e}", 502)
    return flags


def _local_post(path, body):
    req = urllib.request.Request(LOCAL_SERVER + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=10).read()


@app.post("/api/local/cancel_all")
def local_cancel_all():
    """Stop every local-lane job (running and waiting), then empty the local ComfyUI queue and interrupt what it is running."""
    jobs = [j for j in manager.jobs.values() if j.lane == "local" and j.status in ("queued", "running")]
    for j in jobs:
        manager.cancel(j.id)
    cleared = False
    try:
        _local_post("/queue", {"clear": True})
        _local_post("/interrupt", {})
        cleared = True
    except Exception:  # noqa: BLE001  # server down: nothing to clear
        pass
    return {"jobs": len(jobs), "comfy_queue_cleared": cleared}


@app.post("/api/local/restart")
def local_restart():
    """Restart the local ComfyUI user service (it crashes under VRAM pressure); local jobs in flight are cancelled first."""
    for j in [j for j in manager.jobs.values() if j.lane == "local" and j.status in ("queued", "running")]:
        manager.cancel(j.id)
    comfy_watch.expect_restart()
    r = subprocess.run(["systemctl", "--user", "restart", config.COMFYUI_SERVICE], capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise P.ApiError(f"restart failed: {r.stderr.strip() or r.returncode}", 500)
    return {"ok": True}


@app.post("/api/cloud/cancel_all")
def cloud_cancel_all():
    """Stop every cloud-lane job (running and waiting); cancel also pulls what each already submitted off the Comfy Cloud queue."""
    jobs = [j for j in manager.jobs.values() if j.lane == "cloud" and j.status in ("queued", "running")]
    for j in jobs:
        manager.cancel(j.id)
    return {"jobs": len(jobs)}


@app.get("/api/runpod")
def runpod_status():
    """The animate pod for the ⚙ page: up/down, uptime, $/hr (pipeline/runpod_pod.py)."""
    try:
        return runpod_pod.status()
    except runpod_pod.PodError as e:
        return {"up": False, "error": str(e)}


_runpod_balance = {"t": 0.0, "v": None}


@app.get("/api/runpod/balance")
def runpod_balance(refresh: bool = False):
    """The header chip's RunPod balance and current $/hr while the ⚙ animate target is runpod, cached 20s like /api/credits."""
    import time
    if refresh or time.time() - _runpod_balance["t"] > 20 or _runpod_balance["v"] is None:
        try:
            _runpod_balance["v"] = runpod_pod.balance()
            _runpod_balance["t"] = time.time()
        except runpod_pod.PodError as e:
            raise P.ApiError(str(e), 502)
    return _runpod_balance["v"]


@app.post("/api/runpod/stop")
def runpod_stop():
    """Stop pod now: cancel the runpod-lane jobs (running and waiting), then terminate the pod (the volume stays)."""
    jobs = [j for j in manager.jobs.values() if j.lane == "runpod" and j.status in ("queued", "running")]
    for j in jobs:
        manager.cancel(j.id)
    try:
        return {"jobs": len(jobs), "terminated": runpod_pod.terminate()}
    except runpod_pod.PodError as e:
        raise P.ApiError(str(e), 502)


# --- projects ----------------------------------------------------------------------------------------
@app.get("/api/projects")
def projects_list():
    return P.list_projects()


@app.post("/api/projects")
def projects_new(body: dict = Body(...)):
    name, engine = body.get("name", ""), body.get("engine", "full")
    if body.get("oneoff") and not name:
        name = time.strftime("%Y%m%d_%H%M%S") + "_oneoff"
    if not P.NAME_RE.match(name) or (P.PIPELINES / name).exists():
        raise P.ApiError("invalid or existing project name")
    cmd = [sys.executable, str(ROOT / "project.py"), name, "new", "--engine", engine]
    if body.get("oneoff"):
        cmd.append("--oneoff")
    if body.get("from_stage_set"):
        cmd += ["--from", str(config.STAGE_SETS_DIR / Path(body["from_stage_set"]).name)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise P.ApiError((r.stdout + r.stderr).strip() or "new failed")
    return P.get(name)


@app.post("/api/projects/{name}/remix")
def project_remix(name: str, body: dict = Body(...)):
    """Copy the picked stages and their renders into a new project <name>_remix_<suffix>."""
    return P.remix(name, str(body.get("suffix", "")), [int(i) for i in body.get("stages", [])])


@app.delete("/api/projects/{name}")
def project_delete(name: str):
    """Permanently delete the project folder; refused while a job for it is queued/running."""
    P.project_dir(name)
    if any(j.project == name and j.status in ("queued", "running") for j in manager.jobs.values()):
        raise P.ApiError("a job is still queued/running for this project", 409)
    shutil.rmtree(P.PIPELINES / name)
    return {"ok": True}


@app.get("/api/stage_sets")
def stage_sets():
    return sorted(p.name for p in config.STAGE_SETS_DIR.glob("*.json"))


@app.get("/api/projects/{name}")
def project_get(name: str):
    return P.get(name)


@app.put("/api/projects/{name}/settings")
def project_settings(name: str, body: dict = Body(...)):
    return P.put_settings(name, body)


@app.put("/api/projects/{name}/video-megapixels")
def project_video_megapixels(name: str, body: dict = Body(...)):
    P.put_video_megapixels(name, body.get("megapixels"))
    return {}


@app.put("/api/projects/{name}/video-engine")
def project_video_engine(name: str, body: dict = Body(...)):
    P.put_video_engine(name, body.get("engine"))
    return {}


@app.put("/api/projects/{name}/seed")
def project_seed(name: str, body: dict = Body(...)):
    P.put_seed(name, body.get("seed"))
    return {"ok": True}


@app.post("/api/projects/{name}/stages")
def stage_add(name: str):
    return P.add_stage(name)


@app.post("/api/projects/{name}/stages/{sid}/duplicate")
def stage_duplicate(name: str, sid: int):
    return P.duplicate_stage(name, sid)


@app.patch("/api/projects/{name}/stages/{sid}")
def stage_patch(name: str, sid: int, body: dict = Body(...)):
    return P.patch_stage(name, sid, body)


@app.delete("/api/projects/{name}/stages/{sid}")
def stage_delete(name: str, sid: int):
    P.delete_stage(name, sid)
    return {"ok": True}


@app.put("/api/projects/{name}/order")
def stage_order(name: str, body: dict = Body(...)):
    P.reorder_stages(name, [int(i) for i in body["order"]])
    return {"ok": True}


def _civitai_still(name, folder, stage, take):
    """(source png, Civitai parameters text or None, export file name) for one still."""
    d = P.project_dir(name)
    if folder not in ("preview", "IL", "hires", "finish"):
        raise P.ApiError("folder must be preview, IL, hires or finish")
    if take:
        P._take_path(d, stage, take)
    src = civitai_meta.takes.stage_dir(d, stage) / f"{take}.png" if take else d / folder / f"stage_{stage}.png"
    if not src.is_file():
        raise P.ApiError(f"no {folder}/stage_{stage}.png", 404)
    return src, civitai_meta.params(d, folder, stage, take), f"{name}_stage_{stage}_{folder}{'_' + take if take else ''}.png"


@app.post("/api/projects/{name}/civitai/export")
def civitai_export(name: str, body: dict = Body(...)):
    """Write the Civitai-tagged copy to config.CIVITAI_DIR and show it in the file manager, ready to drag onto Civitai's upload."""
    src, text, out = _civitai_still(name, body.get("folder", ""), int(body.get("stage", 0)), body.get("take"))
    config.CIVITAI_DIR.mkdir(parents=True, exist_ok=True)
    f = config.CIVITAI_DIR / out
    f.write_bytes(civitai_meta.tagged_png(src, text))
    subprocess.Popen([*config.FILE_MANAGER, str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"path": str(f), "params": bool(text)}


def _civitai_post_dir(name):
    """The project's Civitai post folder: CIVITAI_DIR/<project>/, or the newest older-style <project>_<time>/ one; None if never exported."""
    found = [p for p in config.CIVITAI_DIR.glob(f"{name}*") if p.is_dir() and (p.name == name or re.fullmatch(re.escape(name) + r"_\d{8}-\d{6}", p.name))]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


@app.get("/api/projects/{name}/civitai/post")
def civitai_post_info(name: str):
    """The existing export (folder, stages in it, when), or {} when the project was never exported."""
    out = _civitai_post_dir(name)
    if not out:
        return {}
    stages = [int(m.group(1)) for p in sorted(out.glob("*_stage_*.png")) if (m := re.fullmatch(r"\d+_stage_(\d+)\.png", p.name))]
    return {"path": str(out), "stages": stages, "mtime": out.stat().st_mtime}


@app.post("/api/projects/{name}/civitai/open")
def civitai_open(name: str):
    out = _civitai_post_dir(name)
    if not out:
        raise P.ApiError("not exported yet", 404)
    subprocess.Popen(["xdg-open", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"ok": True}


def _civitai_plan_item(name: str, d, it) -> tuple:
    """(stage id, still src, its Civitai text, its clip path or None, clip folder) for one post item; ApiError on a bad folder/clip."""
    sid, clip = int(it.get("stage", 0)), it.get("clip")  # clip "" = the project root's videos/; None = still only
    src, text, _ = _civitai_still(name, it.get("still", ""), sid, None)
    mp4 = None
    if clip is not None:
        if clip not in P.folders(d):
            raise P.ApiError(f"no clip folder {clip!r}")
        mp4 = (d / clip if clip else d) / "videos" / f"stage_{sid}.mp4"
        if not mp4.is_file():
            raise P.ApiError(f"no {clip or 'root'}/videos/stage_{sid}.mp4", 404)
    return sid, src, text, mp4, clip


def _civitai_reset(out) -> None:
    """Overwrite: delete only the files an export writes; anything else the user put there stays."""
    for p in out.iterdir():
        if p.is_file() and (p.name == "clips.txt" or re.fullmatch(r"\d+_stage_\d+\.(png|mp4)", p.name)):
            p.unlink()


def _civitai_export(out, plan, d) -> list:
    """The export files: tagged stills, copied clips, and clips.txt with each clip's paste-able parameters."""
    notes = []
    for n, (sid, src, text, mp4, clip) in enumerate(plan, 1):
        (out / f"{n:02d}_stage_{sid}.png").write_bytes(civitai_meta.tagged_png(src, text))
        if mp4:
            shutil.copy2(mp4, out / f"{n:02d}_stage_{sid}.mp4")
            notes.append(f"{n:02d} stage {sid}\n{civitai_meta.clip_params(d, clip, sid) or '(no parameters recorded in the clip)'}\n")
    if notes:
        (out / "clips.txt").write_text("\n".join(notes))
    return notes


@app.post("/api/projects/{name}/civitai/post")
def civitai_post(name: str, body: dict = Body(...)):
    """One Civitai post's files in CIVITAI_DIR/<project>/: per item (in post order) NN_stage_N.png with the Civitai metadata next
    to NN_stage_N.mp4 (an item without "clip" exports its still alone), plus clips.txt with each clip's parameters to paste
    (Civitai doesn't read them from video). A project that
    was already exported is refused (409) unless overwrite: true, which replaces that folder's export files. Opens the folder."""
    d = P.project_dir(name)
    items = body.get("items") or []
    if not items:
        raise P.ApiError("pick at least one stage")
    plan = [_civitai_plan_item(name, d, it) for it in items]
    out = _civitai_post_dir(name)
    if out and not body.get("overwrite"):
        raise P.ApiError(f"already exported to {out}", 409)
    if out:
        _civitai_reset(out)
    else:
        out = config.CIVITAI_DIR / name
        out.mkdir(parents=True)
    notes = _civitai_export(out, plan, d)
    subprocess.Popen([*config.FILE_MANAGER, str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"path": str(out), "pairs": len(plan), "clips": len(notes), "missing_params": [sid for sid, _, text, _, _ in plan if not text]}


@app.get("/api/projects/{name}/civitai")
def civitai_copy(name: str, folder: str, stage: int, take: str = None, format: str = "png", kind: str = "still"):
    """A still with Civitai's `parameters` metadata (prompt, sampler, seed, model + LoRA hashes); format=text: just that text.
    kind=clip: the clip's parameters as text only (Civitai doesn't read them from video files)."""
    d = P.project_dir(name)
    if kind == "clip":
        return {"params": civitai_meta.clip_params(d, folder, stage)}
    src, text, out = _civitai_still(name, folder, stage, take)
    if format == "text":
        return {"params": text}
    return Response(civitai_meta.tagged_png(src, text), media_type="image/png",
                    headers={"Content-Disposition": f'attachment; filename="{out}"', "X-Civitai-Params": "yes" if text else "no"})


@app.get("/api/projects/{name}/stages/{sid}/takes/{tid}")
def stage_take(name: str, sid: int, tid: str):
    return P.take_info(name, sid, tid)


@app.post("/api/projects/{name}/stages/{sid}/takes/{tid}/use")
def stage_take_use(name: str, sid: int, tid: str):
    return P.use_take(name, sid, tid)


@app.post("/api/projects/{name}/stages/{sid}/takes/clear")
def stage_takes_clear(name: str, sid: int):
    return P.clear_takes(name, sid)


@app.get("/api/projects/{name}/stages/{sid}/history")
def stage_history(name: str, sid: int):
    return P.history(name, sid)


@app.post("/api/projects/{name}/stages/{sid}/restore")
def stage_restore(name: str, sid: int, body: dict = Body(...)):
    return {"restored": P.restore(name, sid, body["file"])}


@app.get("/api/projects/{name}/cleanup")
def cleanup_size(name: str):
    """What "Clean up project" would delete: {files, bytes}."""
    return P.hist.cleanup(P.project_dir(name), dry_run=True)


@app.post("/api/projects/{name}/cleanup")
def cleanup(name: str):
    """Delete every earlier version, the preview takes but the current one and the edits' build folders (pipeline/history.py)."""
    if any(j.project == name and j.status in ("queued", "running") for j in manager.jobs.values()):
        raise P.ApiError("a job is running on this project -- clean up when it is done", 409)
    return P.hist.cleanup(P.project_dir(name))


@app.post("/api/projects/{name}/finished")
def project_finished(name: str, body: dict = Body(...)):
    """Mark the project done: clean it up first (as Clean up project: only the current results stay), then set `finished`.
    Unmarking only clears the flag (deleted versions don't come back)."""
    done = bool(body.get("finished"))
    freed = cleanup(name) if done else None
    P.put_finished(name, done)
    return {"finished": done, "cleanup": freed}


@app.post("/api/projects/{name}/stages/{sid}/remove")
def stage_remove(name: str, sid: int, body: dict = Body(...)):
    P.remove_result(name, sid, body.get("folder", ""), body.get("kind", ""))
    return {"ok": True}


@app.post("/api/projects/{name}/stages/{sid}/import")
def stage_import(name: str, sid: int, body: dict = Body(...)):
    return P.import_image(name, sid, body.get("url", ""))


@app.delete("/api/projects/{name}/files/{path:path}")
def file_delete(name: str, path: str):
    """Delete a finished edit mp4 (<folder>/edits/*.mp4, or a legacy one in the project root) -- the only thing the UI deletes."""
    d = P.project_dir(name).resolve()
    f = (d / path).resolve()
    ok = f.parent == d or (f.parent.name == "edits" and f.parent.parent.parent == d or f.parent == d / "edits")
    if not ok or f.suffix != ".mp4" or not f.is_file():
        raise P.ApiError("only edit mp4s can be deleted")
    f.unlink()
    return {"ok": True}


@app.post("/api/projects/{name}/reveal")
def reveal(name: str, body: dict = Body(...)):
    """Open the file manager with an edit (or the project folder) selected -- local desktop only."""
    d = P.project_dir(name).resolve()
    f = (d / body.get("path", "")).resolve() if body.get("path") else d
    if f != d and (d not in f.parents or not f.exists()):
        raise P.ApiError("not found", 404)
    cmd = [*config.FILE_MANAGER, str(f)] if f != d else ["xdg-open", str(d)]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return {"ok": True}


@app.get("/api/projects/{name}/files/{path:path}")
def project_file(name: str, path: str):
    d = P.project_dir(name).resolve()
    f = (d / path).resolve()
    if d not in f.parents or not f.is_file():
        raise P.ApiError("not found", 404)
    return FileResponse(f, headers={"Cache-Control": "no-cache"})


# --- jobs --------------------------------------------------------------------------------------------
@app.get("/api/projects/{name}/estimate")
def estimate(name: str, stages: str = ""):
    """Expected seconds per local action over the given stage ids (comma separated), from past runs' timings."""
    try:
        sys.path.insert(0, str(ROOT))
        import project as proj  # noqa: E402
        return proj.estimate(name, [int(x) for x in stages.split(",") if x.strip()])
    # an estimate must never break the menu: no baseline, bad stage, missing settings
    except Exception:  # noqa: BLE001
        return {}


@app.post("/api/projects/{name}/run")
def run(name: str, body: dict = Body(...)):
    job = manager.submit(name, body.get("action", ""), body)
    return job.info()


@app.post("/api/projects/{name}/animate-batch")
def animate_batch_add(name: str, body: dict = Body(...)):
    """RunPod "Start batch": queue animate stages; the server fires the batch (all projects) once the ⚙ batch size is reached."""
    return manager.add_to_batch(name, body.get("action", ""), body)


@app.get("/api/animate-batch")
def animate_batch_state():
    return {"size": config.animate_batch_size(), "pending": manager.batch_state()}


@app.post("/api/animate-batch/flush")
def animate_batch_flush():
    return manager.flush_batch()


@app.delete("/api/animate-batch")
def animate_batch_clear():
    manager.clear_batch()
    return {"cleared": True}


@app.get("/api/jobs")
def jobs_list():
    return manager.list()


@app.get("/api/notify")
def notify_get():
    """The 🔔: armed = a Pushover message goes out every time the queue empties (no job left queued or running)."""
    return {"armed": manager.notify_armed_at is not None, "configured": all(config.pushover())}


@app.post("/api/notify")
def notify_set(body: dict = Body(...)):
    if not all(config.pushover()):
        raise P.ApiError("Pushover isn't set up: add your user key and app token under ⚙ Settings", 400)
    manager.arm_notify(bool(body.get("armed")))
    return notify_get()


# --- app settings (the ⚙ page; config.APP_SETTINGS_FILE) ------------------------------------------------------------
def _pushover_from(s, user, token):
    if s.get("pushover_user") or s.get("pushover_token"):
        return "settings"
    return ".env" if user or token else None


def app_settings_info():
    """Never returns the token itself: only whether one is set, and where each value comes from."""
    s, (user, token) = config.app_settings(), config.pushover()
    return {"animate_target": config.animate_target(), "animate_batch_size": config.animate_batch_size(),
            "runpod_h3_unet": config.runpod_h3_unet(), "runpod_h3_choices": config.runpod_h3_choices(),
            "pushover_user": user, "pushover_token_set": bool(token),
            "pushover_from": _pushover_from(s, user, token)}


@app.get("/api/app-settings")
def app_settings_get():
    return app_settings_info()


def _apply_target(s: dict, body: dict) -> None:
    if "animate_target" not in body:
        return
    if body["animate_target"] not in ("cloud", "runpod"):
        raise P.ApiError("animate_target must be cloud or runpod")
    s["animate_target"] = body["animate_target"]


def _apply_batch_size(s: dict, body: dict) -> None:
    if "animate_batch_size" not in body:
        return
    try:
        n = int(body["animate_batch_size"])
    except (TypeError, ValueError):
        raise P.ApiError("animate_batch_size must be an integer") from None
    if not 1 <= n <= 100:
        raise P.ApiError("animate_batch_size must be 1-100")
    s["animate_batch_size"] = n


def _apply_h3_unet(s: dict, body: dict) -> None:
    if "runpod_h3_unet" not in body:
        return
    v = str(body["runpod_h3_unet"] or "")
    if v and v not in config.runpod_h3_choices():
        raise P.ApiError(f"runpod_h3_unet must be \"\" (stock) or one of the RunPod volume's H3 UNets: {config.runpod_h3_choices()}")
    s["runpod_h3_unet"] = v


def _apply_pushover(s: dict, body: dict) -> None:
    for k in ("pushover_user", "pushover_token"):
        if k in body:
            v = str(body[k] or "").strip()
            if v:
                s[k] = v
            else:
                s.pop(k, None)


@app.put("/api/app-settings")
def app_settings_put(body: dict = Body(...)):
    """{"pushover_user": str, "pushover_token": str, "animate_target": "cloud" | "runpod", "animate_batch_size": int,
    "runpod_h3_unet": "" (stock) | a runpod_h3_choices name}: a key absent leaves the saved value; "" clears the pushover keys."""
    s = config.app_settings()
    _apply_target(s, body)
    _apply_batch_size(s, body)
    _apply_h3_unet(s, body)
    _apply_pushover(s, body)
    config.APP_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.APP_SETTINGS_FILE.write_text(json.dumps(s, indent=2))
    config.APP_SETTINGS_FILE.chmod(0o600)
    return app_settings_info()


@app.post("/api/app-settings/test-push")
def app_settings_test_push():
    if not all(config.pushover()):
        raise P.ApiError("set the user key and app token first", 400)
    if not jobs.send_pushover("Comfy Studio", "Test notification ✓"):
        raise P.ApiError("Pushover rejected it: check the user key and app token", 400)
    return {"ok": True}


@app.get("/api/jobs/{jid}")
def job_get(jid: str):
    j = manager.jobs.get(jid)
    if not j:
        raise P.ApiError("no such job", 404)
    return j.info(with_lines=True)


@app.post("/api/jobs/{jid}/cancel")
def job_cancel(jid: str):
    return manager.cancel(jid).info()


@app.post("/api/jobs/{jid}/retry")
def job_retry(jid: str):
    """Submit a failed/cancelled job again (only its unfinished stages when it named some): the new job's info."""
    return manager.retry(jid).info()


@app.get("/api/jobs/{jid}/events")
async def job_events(jid: str, request: Request):
    """SSE: `data: {"i": n, "line": ...}` per log line, then `event: end`. Replays from the start."""
    j = manager.jobs.get(jid)
    if not j:
        raise P.ApiError("no such job", 404)

    async def gen():
        i = 0
        while True:
            while i < len(j.lines):
                yield f"data: {json.dumps({'i': i, 'line': j.lines[i], 'stages': j.stage_status, 'status': j.status})}\n\n"
                i += 1
            if j.status in ("done", "failed", "cancelled") and i >= len(j.lines):
                yield f"event: end\ndata: {json.dumps(j.info())}\n\n"
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# --- audio -------------------------------------------------------------------------------------------
@app.get("/api/durations")
def durations(beat_times_file: str = "", beats_per_cut: int = 2, engine: str = "minimax"):
    try:
        P.animate_pipeline.engine_spec(engine)
    # engine_spec's sys.exit (unknown engine) becomes the request's 400 -- re-raising it would kill the server
    except SystemExit as e:  # NOSONAR(S5754)
        raise P.ApiError(str(e)) from e
    return P.fitting_durations(beat_times_file, beats_per_cut, engine)


@app.get("/api/edit-timing")
def edit_timing_get(kind: str = "beat"):
    return edit_timing(kind) or {}


@app.put("/api/audio/start")
def audio_start(body: dict = Body(...)):
    return {"start_time": P.put_start(body.get("beat_times_file"), body["start_time"])}


@app.put("/api/projects/{name}/music")
def project_music(name: str, body: dict = Body(...)):
    return {"music": P.put_music(name, body)}


@app.get("/api/audio")
def audio_list():
    tracks = [str(p) for ext in ("mp3", "wav", "m4a", "flac") for p in AUDIO_DIR.glob(f"*.{ext}")]
    return {"tracks": sorted(tracks), "beat_files": sorted(str(p.relative_to(config.DATA_DIR)) for p in AUDIO_DIR.glob("*.json"))}


PREVIEW_SECONDS = 5


def _track(beat_times_file, audio):
    """The track an edit with these settings would use: --audio, else the .mp3 beside the beat file."""
    f = Path(audio).resolve() if audio else None
    if f and (f.parent != AUDIO_DIR.resolve() or not f.is_file()):
        raise P.ApiError("audio must be a file in the audio folder")
    f = f or P.beat_file(beat_times_file).with_suffix(".mp3")
    if not f.is_file():
        raise P.ApiError("audio file not found", 404)
    return f


@app.get("/api/audio/track")
def audio_track(beat_times_file: str = "", audio: str = ""):
    """The whole track, for the start-time slider to scrub through."""
    return FileResponse(_track(beat_times_file, audio))


@app.get("/api/audio/clip")
def audio_clip(beat_times_file: str = "", audio: str = ""):
    """PREVIEW_SECONDS of the track an edit with these settings would use, starting at the beat file's start time."""
    f = _track(beat_times_file, audio)
    start = P.make_tiktok.load_beats(P.beat_file(beat_times_file))[1] or 0
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(PREVIEW_SECONDS), "-i", str(f), "-vn",
                        "-c:a", "libmp3lame", "-q:a", "4", "-f", "mp3", "-"], capture_output=True)
    if r.returncode or not r.stdout:
        raise P.ApiError(f"ffmpeg failed: {r.stderr.decode()[-200:]}", 500)
    return Response(r.stdout, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/audio/upload")
async def audio_upload(file: UploadFile):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    dst = AUDIO_DIR / Path(file.filename).name
    dst.write_bytes(await file.read())
    return {"path": str(dst)}


if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
