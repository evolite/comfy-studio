"""Project data access for the web UI: project.json read/write, per-folder file state, stage history."""
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: config.py
import config  # noqa: E402

ROOT = config.ROOT
PIPELINES = config.PIPELINES_DIR
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "video"))
import animate_pipeline  # noqa: E402
import interpolate  # noqa: E402
import make_edit  # noqa: E402
import graph_settings  # noqa: E402
import steps  # noqa: E402
import takes  # noqa: E402
import history as hist  # noqa: E402
import image_loras  # noqa: E402
import voice_registry  # noqa: E402

TIERS = [1.0, 2.0, 2.9, 5.0]
STAGE_FIELDS = {"krea_scene", "il_scene", "il_background", "motion", "duration", "loras", "video_loras", "video_engine", "audio", "seed", "video_seed",
                "il_denoise", "face_detailer_enabled", "face_wildcard", "done", "voice", "voices"}
PREVIEW_GATED = {action for (step, _), (action, _) in steps.ACTION_IDS.items() if steps.STEPS[step].needs}  # steps that build on an approved preview (animate only needs its own source still)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PROJECT_JSON = "project.json"  # a project's config file, in the project dir
MP4_GLOB = "*.mp4"  # edit files


class ApiError(Exception):
    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


# every project.json edit is read-modify-write of the whole file: serialize them (a bulk ✓ sends one PATCH per stage at once)
_edit_lock = threading.RLock()


def locked(fn):
    def wrapper(*a, **kw):
        with _edit_lock:
            return fn(*a, **kw)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def project_dir(name):
    if not NAME_RE.match(name) or not (PIPELINES / name / PROJECT_JSON).exists():
        raise ApiError(f"no project {name!r}", 404)
    return PIPELINES / name


def _write_json(path, data):
    """Atomic write; the previous file is kept as _debug/project.json.bak (out of the project folder's top level)."""
    if path.exists():
        bak = path.parent / "_debug" / "project.json.bak"
        bak.parent.mkdir(exist_ok=True)
        shutil.copy2(path, bak)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def preview_approved(stage):
    """Is the stage's preview ✓'d in the UI?"""
    return "preview" in (stage.get("done") or [])


def load(name):
    return json.loads((project_dir(name) / PROJECT_JSON).read_text())


def _stage_ids(d):
    return {int(p.stem[6:]): int(p.stat().st_mtime) for p in d.glob("stage_*.png") if p.stem[6:].isdigit()}


def _clip_ids(d):
    return {int(p.stem[6:]): int(p.stat().st_mtime) for p in (d / "videos").glob("stage_*.mp4") if p.stem[6:].isdigit()}


def folders(d):
    """the project root (only if it holds files) + every non-underscore subfolder that holds stills or clips -> {folder: {stills: {id: mtime}, clips: {id: mtime}}}."""
    root = {"stills": _stage_ids(d), "clips": _clip_ids(d)}
    out = {"": root} if root["stills"] or root["clips"] else {}  # the root only counts while it still holds files
    for sub in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name not in ("videos", "edits")):
        f = {"stills": _stage_ids(sub), "clips": _clip_ids(sub)}
        if f["stills"] or f["clips"] or sub.name in ("preview", "IL", "hires", "finish"):
            out[sub.name] = f
    return out


def edits(d):
    """Edit mp4s: <folder>/edits/*.mp4 (the project root's own are <project>/edits/). `name` is the path relative to the
    project; `folder` the source folder."""
    out = []
    spots = [("", d / "edits")] + [(p.name, p / "edits") for p in d.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name not in ("videos", "edits")]
    for folder, e in spots:
        for p in e.glob(MP4_GLOB) if e.is_dir() else []:
            out.append({"name": str(p.relative_to(d)), "mtime": int(p.stat().st_mtime), "size": p.stat().st_size, "folder": folder})
    return sorted(out, key=lambda e: -e["mtime"])


def stage_ids(cfg):
    return [s.get("id", pos) for pos, s in enumerate(cfg["stages"], 1)]


def last_activity(d):
    """Newest mtime of any file or folder under a project (renders, edits, _debug and project.json alike)."""
    newest = 0
    for root, dirs, files in os.walk(d):
        for n in files + dirs:
            try:
                newest = max(newest, int(os.stat(os.path.join(root, n)).st_mtime))
            except OSError:
                pass
    return newest


def list_projects():
    rows = []
    for p in PIPELINES.iterdir() if PIPELINES.is_dir() else ():  # a fresh data dir has no pipelines/ yet
        pj = p / PROJECT_JSON
        if not pj.exists():
            continue
        try:
            cfg = json.loads(pj.read_text())
        except ValueError:
            continue
        ids = stage_ids(cfg)
        rows.append({"name": p.name, "engine": cfg.get("engine"), "oneoff": bool(cfg.get("oneoff")), "finished": bool(cfg.get("finished")), "stages": len(ids), "first_id": ids[0] if ids else None,
                     "stills": len(_stage_ids(p)), "clips": len(_clip_ids(p)), "edits": len(list(p.glob(MP4_GLOB))),
                     "mtime": max(last_activity(p), int(pj.stat().st_mtime)), "thumb": next(iter(sorted(_stage_ids(p))), None)})
    return sorted(rows, key=lambda r: (-r["mtime"], r["name"]))


def get(name):
    d = project_dir(name)
    cfg = load(name)
    for pos, s in enumerate(cfg["stages"], 1):
        s.setdefault("id", pos)
    fs = folders(d)
    if "preview" in fs:  # a new project has no preview/ folder yet
        fs["preview"]["takes"] = take_lists(d, [s["id"] for s in cfg["stages"]])
    for f in ("IL", "hires"):
        if f in fs:
            fs[f]["stale"] = stale_sources(d, f, fs[f]["stills"])
    for f in fs:
        fs[f]["stale_clips"] = stale_clips(d, f, fs)
    return {"name": name, "config": cfg, "folders": fs, "edits": edits(d)}


# --- preview takes (pipeline/takes.py) -----------------------------------------------------------------------------
@locked
def take_lists(d, ids):
    """{sid: [{id, mtime, current, source, delta}]} for stages that have takes (sync takes in external files first)."""
    out = {}
    for sid in ids:
        if (d / "preview" / f"stage_{sid}.png").exists() or takes.stage_dir(d, sid).is_dir():
            takes.sync(d, sid)
            rows = takes.listing(d, sid)
            if rows:
                out[sid] = rows
    return out


def stale_sources(d, folder, stills):
    """Stage ids whose <folder>/stage_N.png was made from a preview take that is no longer current."""
    out = []
    for sid in stills:
        try:
            src = json.loads((d / folder / "_debug" / f"stage_{sid}" / "source.json").read_text()).get("take")
        except (OSError, ValueError):
            continue
        cur = takes.current_id(d, sid)
        if src and cur and src != cur:
            out.append(sid)
    return out


def stale_clips(d, folder, fs):
    """Stage ids whose clip is older than what it was made from: the folder's own still (animate reads <folder>/stage_N.png),
    or for interp/ the source clip named in interp/_debug/stage_N/source.json."""
    out = []
    for sid, clip in fs[folder]["clips"].items():
        if folder == interpolate.FOLDER:
            try:
                src = json.loads((d / folder / "_debug" / f"stage_{sid}" / "source.json").read_text()).get("folder", "")
            except (OSError, ValueError):
                continue
            made_from = fs.get(src, {}).get("clips", {}).get(sid)
        else:
            made_from = fs[folder]["stills"].get(sid)
        if made_from and made_from > clip + 1:  # mtimes in whole seconds; a clip saved right after its still isn't stale
            out.append(sid)
    return out


def _take_path(d, sid, tid):
    if not re.fullmatch(r"\d{8}-\d{6}-\d{3}", tid or "") or not (takes.stage_dir(d, sid) / f"{tid}.png").is_file():
        raise ApiError(f"no take {tid} for stage {sid}", 404)


def take_info(name, sid, tid):
    """A take's record, the current take's record, and the differences between them."""
    d = project_dir(name)
    _take_path(d, sid, tid)
    cur = takes.current_id(d, sid)
    rec, cur_rec = takes.load_record(d, sid, tid), takes.load_record(d, sid, cur) if cur else None
    return {"id": tid, "current": cur, "record": rec, "current_record": cur_rec,
            "diff": takes.diff(rec, cur_rec) if tid != cur else None}


@locked
def use_take(name, sid, tid):
    """Make a take current and ✓ the preview. If the current take changed, everything made from the old one loses its ✓."""
    d = project_dir(name)
    _take_path(d, sid, tid)
    takes.sync(d, sid)
    changed = takes.set_current(d, sid, tid)
    cfg = load(name)
    for pos, s in enumerate(cfg["stages"], 1):
        if s.get("id", pos) == sid:
            done = s.get("done") if isinstance(s.get("done"), list) else []
            s["done"] = ["preview"] if changed else sorted(set(done) | {"preview"})
            _write_json(d / PROJECT_JSON, cfg)
            return {"changed": changed, "done": s["done"]}
    raise ApiError(f"no stage {sid}", 404)


@locked
def clear_takes(name, sid):
    """Move every take but the current one to _debug/removed/."""
    d = project_dir(name)
    return {"moved": takes.remove_others(d, sid)}


def validate_stage(patch, label):
    bad = set(patch) - STAGE_FIELDS - set(graph_settings_keys())
    if bad:
        raise ApiError(f"{label}: unknown fields {sorted(bad)}")
    d = patch.get("duration")
    if d is not None and ("duration" in patch) and (isinstance(d, bool) or not isinstance(d, (int, float)) or not 1.0 <= d <= 10.0):
        raise ApiError(f"{label}: duration must be between 1.0 and 10.0 seconds")
    if patch.get("voice") and patch["voice"] not in voice_registry.load()["voices"]:
        raise ApiError(f"{label}: no voice {patch['voice']!r} in the voice library")
    try:
        voice_registry.check_cast(patch.get("voices"), label)
    except ValueError as e:
        raise ApiError(str(e)) from e
    if patch.get("video_engine") and patch["video_engine"] not in animate_pipeline.VIDEO_ENGINES:
        raise ApiError(f"{label}: video_engine must be one of {', '.join(animate_pipeline.VIDEO_ENGINES)}")
    try:
        if "loras" in patch:
            image_loras.check_loras(patch["loras"], label)
        if "video_loras" in patch:
            animate_pipeline.check_loras(patch["video_loras"], label)
    # validation's sys.exit becomes the request's 400 -- re-raising it would kill the server
    except SystemExit as e:  # NOSONAR(S5754)
        raise ApiError(str(e)) from e


def graph_settings_keys():
    try:
        keys = set()
        for table, special in graph_settings.ENGINE_KEYS.values():
            keys |= set(table) | set(special)
        return keys
    except Exception:
        return set()


@locked
def patch_stage(name, sid, patch):
    cfg = load(name)
    for pos, s in enumerate(cfg["stages"], 1):
        if s.get("id", pos) == sid:
            validate_stage(patch, f"stage {sid}")
            for k, v in patch.items():
                if v is None or v is False or (k == "done" and not v) or v == "" and k not in ("krea_scene", "il_scene", "motion"):
                    s.pop(k, None)
                else:
                    s[k] = v
            _write_json(project_dir(name) / PROJECT_JSON, cfg)
            return s
    raise ApiError(f"no stage {sid}", 404)


@locked
def put_settings(name, settings):
    cfg = load(name)
    try:
        graph_settings.check_settings(cfg.get("engine"), settings)
    # validation's sys.exit becomes the request's 400 -- re-raising it would kill the server
    except SystemExit as e:  # NOSONAR(S5754)
        raise ApiError(str(e)) from e
    cfg["settings"] = settings
    _write_json(project_dir(name) / PROJECT_JSON, cfg)
    return cfg


@locked
def put_seed(name, seed):
    cfg = load(name)
    if seed in (None, ""):
        cfg.pop("seed", None)
    else:
        cfg["seed"] = int(seed)
    _write_json(project_dir(name) / PROJECT_JSON, cfg)


@locked
def put_finished(name, finished):
    """Top-level `finished`: the project is done (set after its cleanup; the web UI dims it in the project list)."""
    cfg = load(name)
    if finished:
        cfg["finished"] = True
    else:
        cfg.pop("finished", None)
    _write_json(project_dir(name) / PROJECT_JSON, cfg)


@locked
def put_video_megapixels(name, mp):
    cfg = load(name)
    if mp in (None, ""):
        cfg.pop("video_megapixels", None)
    else:
        mp = float(mp)
        if not 0.1 <= mp <= 2.0:
            raise ApiError("video_megapixels must be between 0.1 and 2.0")
        cfg["video_megapixels"] = mp
    _write_json(project_dir(name) / PROJECT_JSON, cfg)


@locked
def put_video_engine(name, engine):
    cfg = load(name)
    if engine in (None, "", animate_pipeline.DEFAULT_ENGINE):
        cfg.pop("video_engine", None)
    elif engine in animate_pipeline.VIDEO_ENGINES:
        cfg["video_engine"] = engine
    else:
        raise ApiError(f"video_engine must be one of {', '.join(animate_pipeline.VIDEO_ENGINES)}")
    _write_json(project_dir(name) / PROJECT_JSON, cfg)


@locked
def put_voice(name, voice):
    """The project's default voice (a stage's own `voice` wins); None / "" clears it."""
    cfg = load(name)
    if not voice:
        cfg.pop("voice", None)
    elif voice in voice_registry.load()["voices"]:
        cfg["voice"] = voice
    else:
        raise ApiError(f"no voice {voice!r} in the voice library")
    _write_json(project_dir(name) / PROJECT_JSON, cfg)


@locked
def add_stage(name):
    cfg = load(name)
    ids = stage_ids(cfg)
    new = {"id": max(ids, default=0) + 1, "krea_scene": "", "il_scene": "", "motion": "", "video_loras": {}, "loras": {}, "audio": True}
    cfg["stages"].append(new)
    _write_json(project_dir(name) / PROJECT_JSON, cfg)
    return new


@locked
def duplicate_stage(name, sid):
    """A copy of stage `sid` right after it, under a new id: its config and every folder's still plus the preview takes, so it
    can be re-animated with another motion. Clips stay behind (and so do their ✓s): the copy is there to make a new one."""
    cfg = load(name)
    pos = next((i for i, s in enumerate(cfg["stages"]) if s.get("id", i + 1) == sid), None)
    if pos is None:
        raise ApiError(f"no stage {sid}", 404)
    new = {**json.loads(json.dumps(cfg["stages"][pos])), "id": max(stage_ids(cfg), default=0) + 1}
    if isinstance(new.get("done"), list):
        new["done"] = [k for k in new["done"] if not k.endswith("/videos")]
    cfg["stages"].insert(pos + 1, new)
    d = project_dir(name)
    for f in [d] + [p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name not in ("videos", "edits")]:
        if (f / f"stage_{sid}.png").is_file():
            shutil.copy2(f / f"stage_{sid}.png", f / f"stage_{new['id']}.png")
    if takes.stage_dir(d, sid).is_dir():
        shutil.copytree(takes.stage_dir(d, sid), takes.stage_dir(d, new["id"]), copy_function=shutil.copy2, dirs_exist_ok=True)
    _write_json(d / PROJECT_JSON, cfg)
    return new


@locked
def delete_stage(name, sid):
    cfg = load(name)
    before = len(cfg["stages"])
    cfg["stages"] = [s for pos, s in enumerate(cfg["stages"], 1) if s.get("id", pos) != sid]
    if len(cfg["stages"]) == before:
        raise ApiError(f"no stage {sid}", 404)
    _write_json(project_dir(name) / PROJECT_JSON, cfg)


@locked
def remix(name, suffix, ids):
    """New project <name>_remix_<suffix> with the picked stages (source order, same ids) and their files: every folder's
    still/clip, its _debug/[*_]stage_N records and the preview takes. Edits and project-level history stay behind."""
    d = project_dir(name)
    new = f"{name}_remix_{suffix}"
    if not suffix or not NAME_RE.match(suffix) or not NAME_RE.match(new) or (PIPELINES / new).exists():
        raise ApiError(f"invalid or existing project name {new!r}")
    cfg = load(name)
    ids = set(ids)
    if not ids or not ids <= set(stage_ids(cfg)):
        raise ApiError("pick at least one of the project's stages")
    cfg["stages"] = [s for pos, s in enumerate(cfg["stages"], 1) if s.get("id", pos) in ids]
    out = PIPELINES / new
    out.mkdir()
    _write_json(out / PROJECT_JSON, cfg)

    def copy(src):
        dst = out / src.relative_to(d)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, copy_function=shutil.copy2)
        else:
            shutil.copy2(src, dst)
    spots = [d] + [p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name not in ("videos", "edits")]
    for f in spots:
        _remix_spot_files(f, d, ids, copy)
    return get(new)


def _remix_spot_files(f, d, ids, copy) -> None:
    """One folder's picked-stage files: its still/clip, the preview takes folder, and its _debug stage records."""
    for sid in ids:
        _remix_stage_files(f, d, sid, copy)
        _remix_debug_files(f, d, sid, copy)


def _remix_stage_files(f, d, sid, copy) -> None:
    """The stage's result file in this folder (plus its takes folder when this spot is the previews one)."""
    spots = (f / f"stage_{sid}.png", f / "videos" / f"stage_{sid}.mp4", takes.stage_dir(d, sid) if f.name == takes.FOLDER else None)
    for src in spots:
        if src and src.exists():
            copy(src)


def _remix_debug_files(f, d, sid, copy) -> None:
    """The folder's _debug records of this stage (only below the project root, which has none)."""
    if f == d or not (f / "_debug").is_dir():
        return
    for src in (f / "_debug").iterdir():
        if re.fullmatch(rf"(.+_)?stage_{sid}", src.name):
            copy(src)


@locked
def reorder_stages(name, order):
    cfg = load(name)
    by = {s.get("id", pos): s for pos, s in enumerate(cfg["stages"], 1)}
    if sorted(order) != sorted(by):
        raise ApiError("order must list every stage id exactly once")
    cfg["stages"] = [by[i] for i in order]
    _write_json(project_dir(name) / PROJECT_JSON, cfg)


# --- stage history: earlier versions of a stage's results (pipeline/history.py; previews are takes) ----------
def history(name, sid):
    return hist.listing(project_dir(name), sid)


def restore(name, sid, file):
    try:
        return hist.restore(project_dir(name), sid, file)
    except FileNotFoundError as e:
        raise ApiError(str(e), 404)


def remove_result(name, sid, folder, kind):
    """Take one still/clip out of a folder (into its history, not deleted)."""
    d = project_dir(name).resolve()
    base = (d / folder).resolve()
    if base != d and base.parent != d:
        raise ApiError("bad folder")
    f = base / ("videos" if kind == "clip" else "") / f"stage_{sid}.{'mp4' if kind == 'clip' else 'png'}"
    if kind not in ("still", "clip") or not f.is_file():
        raise ApiError("no such file", 404)
    hist.archive(d, f)


def import_image(name, sid, url):
    """Fetch an image from a civitai URL into import/stage_<sid>.png (a previous import goes into its history)."""
    import io
    import urllib.parse
    import urllib.request
    from PIL import Image
    u = urllib.parse.urlparse(url.strip())
    if u.scheme != "https" or not (u.hostname or "").endswith("civitai.com"):
        raise ApiError("only https civitai.com image links are supported")
    d = project_dir(name)
    req = urllib.request.Request(u.geturl(), headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read(40_000_001)
    except Exception as e:
        raise ApiError(f"download failed: {e}")
    if len(raw) > 40_000_000:
        raise ApiError("image is larger than 40 MB")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise ApiError("that link is not an image")
    dst = d / "import" / f"stage_{sid}.png"
    dst.parent.mkdir(exist_ok=True)
    hist.archive(d, dst)
    img.save(dst)
    return {"folder": "import", "size": list(img.size)}


# --- music: which track/beats the edit will use, and the clip lengths that fit it --------------------
DEFAULT_BEATS = config.DEFAULT_BEATS  # relative to config.DATA_DIR


def beat_file(rel):
    f = config.data_path(rel or DEFAULT_BEATS).resolve()
    if f.parent != config.AUDIO_DIR.resolve() or f.suffix != ".json" or not f.is_file():
        raise ApiError("beat file must be a .json in audio/")
    return f


def fitting_durations(beats, beats_per_cut=2, engine=animate_pipeline.DEFAULT_ENGINE):
    """Clip lengths on the i2v engine's frame grid (MiniMax 17k+5, LTX 8k+1; up to 8s) and how well each fills whole cuts of this music.
    A clip gets round(length / cut) cuts, capped at make_edit.MAX_SUBCUTS_PER_STAGE (what the default edit does);
    `coverage` = cuts' total / clip length (100% = no unused tail, no freeze). `request` is the value to send as the
    duration: floored to 0.01s so it snaps to exactly this grid point."""
    bt, start = make_edit.load_beats(beat_file(beats))
    beat = (bt[-1] - bt[0]) / (len(bt) - 1)
    cut = beats_per_cut * beat
    opts = []
    e = animate_pipeline.engine_spec(engine)
    for k in range(e["min_k"], 8 * e["fps"] // e["grid"] + 1):
        frames = e["grid"] * k + e["offset"]
        secs = frames / e["fps"]
        if secs < 1.0:  # below the shortest duration a stage takes
            continue
        raw = secs / cut
        n = max(1, min(make_edit.MAX_SUBCUTS_PER_STAGE, round(raw)))
        cov = n * cut / secs
        opts.append({"request": math.floor(secs * 100) / 100, "frames": frames, "seconds": round(secs, 2), "cuts": n,
                     "coverage": round(cov, 2), "recommended": n == round(raw) and 0.93 <= cov <= 1.06})
    return {"beat_times_file": str(beat_file(beats).relative_to(config.DATA_DIR)), "bpm": round(60 / beat, 1), "beat": round(beat, 3), "cut": round(cut, 3), "start_time": start, "options": opts}


def put_start(beats, start_time):
    """Save a track's start time (where in the song an edit starts) into its beat file (the one place it lives)."""
    f = beat_file(beats)
    data = json.loads(f.read_text())
    data.pop("drop_time", None)  # older files' name for it
    data["start_time"] = float(start_time)
    f.write_text(json.dumps(data))
    return data["start_time"]


# no-music ("scenario") edit settings, saved in project.json "music" beside the track: {"none": true, "transition": "fade", ...}
SCENARIO_NUMS = ("transition_seconds", "still_seconds", "fade_in", "fade_out")


def _validated_music(music) -> dict:
    """The scenario-edit settings, validated and coerced: numbers as floats, blanks dropped."""
    if music.get("beat_times_file"):
        beat_file(music["beat_times_file"])
    if music.get("audio") and not Path(music["audio"]).is_file():
        raise ApiError("audio file not found")
    if music.get("transition") and music["transition"] not in make_edit.TRANSITIONS:
        raise ApiError(f"transition must be one of {', '.join(make_edit.TRANSITIONS)}")
    for k in SCENARIO_NUMS:
        if music.get(k) not in (None, "") and not 0 <= float(music[k]) <= 30:
            raise ApiError(f"{k} must be 0-30 s")
    return {k: (float(v) if k in SCENARIO_NUMS else v) for k, v in music.items()
            if k in ("beat_times_file", "audio", "none", "transition", *SCENARIO_NUMS) and v not in (None, "", False)}


@locked
def put_music(name, music):
    cfg = load(name)
    music = _validated_music(music) if music else None
    if music:
        cfg["music"] = music
    else:
        cfg.pop("music", None)
    _write_json(project_dir(name) / PROJECT_JSON, cfg)
    return cfg.get("music")
