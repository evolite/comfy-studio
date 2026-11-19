"""Seamless loop of a finished clip (the Cut phase's "Seamless loop"): <folder>/videos/stage_N.mp4 -> cut/videos/stage_N.mp4.

Within the in/out range the user set, find the loop [i, j) whose seam matches best over a window of frames around it (pose and
motion phase, not one frame; judged per region, so a head or hand that drifted counts even when the bodies match), preferring the longest loop that is nearly as good as the best. Then drop BRIDGE frames each side of
the seam and let RIFE (local ComfyUI, rife49) generate the in-between frames across the gap, so the seam is motion, not a dissolve.
The file is rotated so that bridge sits mid-file: its last frame is the real frame before its first, so the wrap is plain footage.
Audio: the same loop with a short crossfade at the seam, rotated the same way, padded to the exact video length.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import config  # noqa: E402
import cut_clip  # noqa: E402
import history  # noqa: E402
from interpolate import probe  # noqa: E402
from targets import local  # noqa: E402

BRIDGE = 3          # frames dropped each side of the seam; RIFE makes 2 * BRIDGE new ones
WINDOW = 5          # frames each side of the seam compared when scoring a loop
MIN_SECONDS = 1.0   # shortest loop searched
NEAR_BEST = 1.10    # a longer loop wins when its score is within this factor of the best
AUDIO_FADE = 6      # frames of audio crossfade at the seam
THUMB = (96, 64)
CELL = 16           # seam scoring compares 16x16-pixel cells of the thumbnail; the worst cell is the score


def gray_frames(src: Path) -> np.ndarray:
    w, h = THUMB
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h * w).astype(np.float32)


def find_loop(frames: np.ndarray, first: int, last: int, fps: float) -> tuple:
    """(i, j): the loop frames[i:j] inside [first, last) whose seam (frames around j vs around i) matches best."""
    pad = max(WINDOW, BRIDGE)
    last = min(last, len(frames))
    lo, hi = first + pad, last - pad - 1  # every compared frame and the audio past j stay inside [first, last)
    shortest = int(round(MIN_SECONDS * fps))
    if hi - lo <= shortest:
        raise ValueError(f"the range is too short for a loop: keep at least {(shortest + 2 * pad) / fps:.1f}s")
    w, h = THUMB
    span = frames[lo - WINDOW:hi + WINDOW + 1].reshape(-1, h // CELL, CELL, w // CELL, CELL)
    # dist[a, b, c]: mean difference of grid cell c between span frames a and b
    dist = np.stack([np.abs(span - f).mean(axis=(2, 4)).reshape(len(span), -1) for f in span])
    ks = np.arange(-WINDOW, WINDOW + 1)
    best = []
    for i in range(lo, hi - shortest):
        a = i - lo + WINDOW + ks
        for j in range(i + shortest, hi + 1):
            b = j - lo + WINDOW + ks
            best.append((float(dist[b, a].mean(axis=0).max()), i, j))  # the worst-matching cell decides: a head that moved counts
    floor = min(s for s, _, _ in best)
    good = [(j - i, -s, i, j) for s, i, j in best if s <= floor * NEAR_BEST]
    _, _, i, j = max(good)
    return i, j


def _rife_between(a: Path, b: Path, n: int, out_dir: Path) -> list:
    """n frames between images a and b, made by RIFE on the local ComfyUI (queued at the front)."""
    server = config.LOCAL_COMFY_URL.rstrip("/")
    na, nb = local.upload(server, a, "loop_seam_a.png"), local.upload(server, b, "loop_seam_b.png")
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": na}},
        "2": {"class_type": "LoadImage", "inputs": {"image": nb}},
        "3": {"class_type": "ImageBatch", "inputs": {"image1": ["1", 0], "image2": ["2", 0]}},
        "4": {"class_type": "RIFE VFI", "inputs": {"frames": ["3", 0], "ckpt_name": "rife49.pth", "clear_cache_after_n_frames": 10,
                                                   "multiplier": n + 1, "fast_mode": False, "ensemble": True, "scale_factor": 1.0,
                                                   "dtype": "float32", "torch_compile": False, "batch_size": 1}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": "loop_seam/f"}},
    }
    body = json.dumps({"prompt": graph, "client_id": local.CLIENT_ID, "front": True, "extra_data": {"label": "seamless loop seam"}})
    pid = json.loads(local._call(server, "/prompt", body.encode(), local.JSON_HDR))["prompt_id"]
    hist = local.wait(server, pid, 600)
    out = []
    for k, f in enumerate(hist["outputs"]["5"]["images"]):
        dest = out_dir / f"{k:04d}.png"
        local.download(server, f, dest)
        out.append(dest)
    if len(out) != n + 2:
        raise RuntimeError(f"RIFE returned {len(out)} frames, expected {n + 2}")
    return out[1:-1]


def loop(d: Path, sid, folder: str, start: float, end: float) -> dict:
    """Make cut/videos/stage_<sid>.mp4 a seamless loop found inside [start, end] s of <folder>/videos/stage_<sid>.mp4."""
    d = Path(d)
    src = (d / folder if folder else d) / "videos" / f"stage_{sid}.mp4"
    dst = d / cut_clip.FOLDER / "videos" / f"stage_{sid}.mp4"
    fps, audio = probe(src)
    frames = gray_frames(src)
    first, last = max(0, int(round(start * fps))), min(len(frames), int(round(end * fps)))
    i, j = find_loop(frames, first, last, fps)
    length, m = j - i, BRIDGE
    with tempfile.TemporaryDirectory(dir=d) as tmp:
        tmp = Path(tmp)
        full, seq, bridge = tmp / "src", tmp / "seq", tmp / "bridge"
        for p in (full, seq, bridge):
            p.mkdir()
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-start_number", "0", str(full / "%05d.png")], check=True)
        mids = _rife_between(full / f"{j - m - 1:05d}.png", full / f"{i + m:05d}.png", 2 * m, bridge)
        frames_out = [full / f"{n:05d}.png" for n in range(i + m, j - m)] + mids  # body, then the bridge back into its first frame
        h = (length - 2 * m) // 2 + m  # rotation: the bridge lands mid-file
        for k, p in enumerate(frames_out[h:] + frames_out[:h]):
            shutil.copyfile(p, seq / f"{k:05d}.png")
        cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:g}", "-i", str(seq / "%05d.png")]
        if audio:
            fade, total = AUDIO_FADE / fps, length / fps
            fc = (f"[0:a]asplit=2[a1][a2];[a1]atrim=start={(i + m) / fps}:end={min(j + m, len(frames)) / fps},asetpts=PTS-STARTPTS[b];"
                  f"[a2]atrim=start={i / fps}:end={(i + m) / fps + fade},asetpts=PTS-STARTPTS[c];"
                  f"[b][c]acrossfade=d={fade}:c1=tri:c2=tri,apad=whole_dur={total},atrim=end={total},asplit[x1][x2];"
                  f"[x1]atrim=start={h / fps},asetpts=PTS-STARTPTS[q2];[x2]atrim=end={h / fps},asetpts=PTS-STARTPTS[q1];"
                  f"[q2][q1]concat=n=2:v=0:a=1,apad=whole_dur={total},atrim=end={total}[a]")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex", fc, "-map", "[a]", str(tmp / "a.wav")], check=True)
            cmd += ["-i", str(tmp / "a.wav"), "-c:a", "aac", "-b:a", "192k"]
        out = tmp / dst.name
        cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
        subprocess.run(cmd, check=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        history.archive(d, dst)
        shutil.move(out, dst)
    info = {"folder": folder, "start": round(start, 4), "end": round(end, 4),
            "loop": {"in": round(i / fps, 4), "out": round(j / fps, 4), "seconds": round(length / fps, 3)}}
    cut_clip.source_file(d, sid).parent.mkdir(parents=True, exist_ok=True)
    cut_clip.source_file(d, sid).write_text(json.dumps(info))
    return info
