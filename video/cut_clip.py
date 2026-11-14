"""Trim a finished clip's start and end (the board's Cut phase): <folder>/videos/stage_N.mp4 -> cut/videos/stage_N.mp4.

A frame-exact re-encode (libx264 CRF 16, the source's fps, its audio as AAC), so the cut lands where the handles were set. The source
clip stays; a re-cut moves the previous result into the stage's history (pipeline/history.py). cut/_debug/stage_N/source.json records
the source folder and the in/out seconds, so the editor reopens where it was left. Make_edit takes cut/ clips without its default
ramp-up trim (make_edit.trim_for): the in-point is already chosen.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import history  # noqa: E402
from interpolate import probe  # noqa: E402

FOLDER = "cut"
MIN_SECONDS = 0.3  # shortest cut kept


def source_file(d: Path, sid) -> Path:
    return Path(d) / FOLDER / "_debug" / f"stage_{sid}" / "source.json"


def info(d: Path, sid) -> dict | None:
    """The last cut's {folder, start, end}, or None."""
    try:
        return json.loads(source_file(d, sid).read_text())
    except (OSError, ValueError):
        return None


def cut(d: Path, sid, folder: str, start: float, end: float) -> Path:
    """Cut <folder>/videos/stage_<sid>.mp4 to [start, end] seconds into cut/videos/stage_<sid>.mp4; returns that path."""
    d = Path(d)
    src = (d / folder if folder else d) / "videos" / f"stage_{sid}.mp4"
    dst = d / FOLDER / "videos" / f"stage_{sid}.mp4"
    fps, audio = probe(src)
    with tempfile.TemporaryDirectory(dir=d / "_debug" if (d / "_debug").is_dir() else d) as tmp:
        out = Path(tmp) / dst.name
        subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.4f}", "-i", str(src), "-t", f"{end - start:.4f}", "-map", "0:v:0",
                        *(["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"] if audio else []),
                        "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", f"{fps:g}",
                        "-movflags", "+faststart", str(out), "-loglevel", "error"], check=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        history.archive(d, dst)
        shutil.move(out, dst)
    source_file(d, sid).parent.mkdir(parents=True, exist_ok=True)
    source_file(d, sid).write_text(json.dumps({"folder": folder, "start": round(start, 4), "end": round(end, 4)}))
    return dst
