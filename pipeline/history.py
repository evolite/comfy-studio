"""Earlier versions of a stage's results: every still or clip a re-run, face pass, re-roll, remove or roll-back replaced, so the web
UI can bring one back. Preview renders are not here -- they are takes (takes.py).

    _debug/history/<slot>/stage_N/<YYYYMMDD-HHMMSS-mmm>.<png|mp4>     slot = the folder ("root" for the project root), "-videos" for clips

Files archived before this existed are listed too: _debug/prev_stills (root stills), prev_IL, prev_videos (<folder>_stage_N_<hex>.mp4)
and removed/ (<folder>_<unix time>_stage_N.<ext>).
"""
import os
import re
import shutil
import time
from pathlib import Path

import takes

DIR = "history"
SKIP = {"preview"}  # its own history is the takes


def _slot(d: Path, path: Path):
    """(folder, kind) of a result path in the project: IL/stage_3.png -> ("IL", "still"), hires/videos/stage_3.mp4 -> ("hires", "clip")."""
    rel = Path(path).resolve().relative_to(Path(d).resolve()).parts
    kind = "clip" if len(rel) >= 2 and rel[-2] == "videos" else "still"
    parts = rel[:-2] if kind == "clip" else rel[:-1]
    return (parts[0] if parts else ""), kind


def _dir(d: Path, folder: str, kind: str, sid) -> Path:
    return Path(d) / "_debug" / DIR / f"{folder or 'root'}{'-videos' if kind == 'clip' else ''}" / f"stage_{sid}"


def archive(d: Path, path: Path):
    """Move a result that is about to be replaced into its history (nothing if it doesn't exist). Returns the archived path."""
    path = Path(path)
    if not path.is_file():
        return None
    folder, kind = _slot(d, path)
    dst = _dir(d, folder, kind, path.stem[6:])
    dst.mkdir(parents=True, exist_ok=True)
    t = time.time()
    while True:
        out = dst / (time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + f"-{int(t * 1000) % 1000:03d}{path.suffix}")
        if not out.exists():
            break
        t += 0.001
    shutil.move(path, out)
    return out


def _legacy_stills(dbg: Path, sid):
    for p in (dbg / "prev_stills").glob(f"stage_{sid}_*.png"):
        yield p, "", "still"
    for p in (dbg / "prev_IL").glob(f"stage_{sid}_*.png"):
        yield p, "IL", "still"


def _prev_videos(dbg: Path, sid, project: str):
    """Archived clips from the pre-history folders: (file, folder label, kind)."""
    for p in (dbg / "prev_videos").glob(f"*stage_{sid}_*.mp4"):
        m = re.fullmatch(rf"(.*?)_?stage_{sid}_[0-9a-f]+", p.stem)
        if m:
            yield p, "" if m.group(1) in ("", project) else m.group(1), "clip"


def _removed(dbg: Path, sid):
    """Stage files swept into _debug/removed: (file, folder label, kind)."""
    for p in (dbg / "removed").glob(f"*_stage_{sid}.*"):
        m = re.fullmatch(rf"(.+?)_\d+_stage_{sid}", p.stem)
        if m and p.suffix in (".png", ".mp4"):
            label = "" if m.group(1) == "root" else m.group(1)
            yield p, label, "clip" if p.suffix == ".mp4" else "still"


def _legacy_clips(dbg: Path, sid, project: str):
    """(file under _debug, folder, kind) of the archived clips from before this module."""
    yield from _prev_videos(dbg, sid, project)
    yield from _removed(dbg, sid)


def _legacy(dbg: Path, sid, project: str):
    """(file under _debug, folder, kind) of the archives from before this module."""
    yield from _legacy_stills(dbg, sid)
    yield from _legacy_clips(dbg, sid, project)


def listing(d: Path, sid):
    """Every earlier version of stage sid, newest first: [{folder, kind, file (relative to _debug), mtime}]."""
    dbg = Path(d) / "_debug"
    rows = []
    for slot in (dbg / DIR).iterdir() if (dbg / DIR).is_dir() else ():
        folder, kind = (slot.name[:-7], "clip") if slot.name.endswith("-videos") else (slot.name, "still")
        for p in (slot / f"stage_{sid}").glob("*.*"):
            rows.append((p, "" if folder == "root" else folder, kind))
    rows += _legacy(dbg, sid, Path(d).name)
    out = [{"folder": f, "kind": k, "file": str(p.relative_to(dbg)), "mtime": int(p.stat().st_mtime)} for p, f, k in rows if f not in SKIP]
    return sorted(out, key=lambda r: -r["mtime"])


def restore(d: Path, sid, file: str) -> str:
    """Put an earlier version back as the stage's result; the one it replaces goes into history. Returns the result's path."""
    row = next((r for r in listing(d, sid) if r["file"] == file), None)
    if row is None:
        raise FileNotFoundError(f"no earlier version {file!r} of stage {sid}")
    base = Path(d) / row["folder"]
    dst = base / "videos" / f"stage_{sid}.mp4" if row["kind"] == "clip" else base / f"stage_{sid}.png"
    archive(d, dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(Path(d) / "_debug" / file, dst)
    os.utime(dst)  # it is the newest result now: thumbnails refresh, later steps see it as changed
    return str(dst.relative_to(d))


LEGACY = ("prev_stills", "prev_IL", "prev_videos", "prev_previews", "removed")


def cleanup(d: Path, dry_run: bool = False) -> dict:
    """Delete every earlier version (history and the older archives), every preview take but the current one, and the edits' build
    folders. Current results, approvals and finished edits stay. dry_run: only count. -> {"files", "bytes"}."""
    d = Path(d)
    doomed = [d / "_debug" / x for x in (DIR, *LEGACY)] + list(d.glob("*/_debug/tiktok_build_*")) + list(d.glob("_debug/tiktok_build_*"))
    for sd in (d / takes.FOLDER / "takes").glob("stage_*"):
        cur = takes.current_id(d, sd.name[6:])
        if cur:  # no current take recorded: leave that stage's takes alone
            doomed += [f for f in sd.iterdir() if f.name != "current" and f.name.split(".")[0] != cur]
    files = [f for p in doomed if p.exists() for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file()]
    size = sum(f.stat().st_size for f in files)
    if not dry_run:
        for p in doomed:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()
    return {"files": len(files), "bytes": size}
