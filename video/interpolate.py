"""RIFE frame interpolation of finished i2v clips on the local ComfyUI (free): H3 renders at 24 fps, the edit at 30, so a 24 fps clip
repeats frames in the edit; interpolating it to 48 fps gives the edit real in-between frames. Same length, same audio.

Graph: LoadVideo -> GetVideoComponents -> RIFE VFI (rife49, x`MULTIPLIER`) -> CreateVideo (source fps x MULTIPLIER, source audio) ->
SaveVideo. RIFE VFI comes from the ComfyUI-Frame-Interpolation custom node (downloads rife49.pth on first use).
Normally run via `project.py <name> interpolate [folder] [N..]`: <folder>/videos/stage_N.mp4 -> interp/videos/stage_N.mp4.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from targets import local  # noqa: E402

MULTIPLIER = 2
FOLDER = "interp"  # results land in <project>/interp/videos/, a clip folder of its own (the board's Interpolate phase, Compose)


def probe(path: Path) -> tuple:
    """(fps, has an audio stream) of a clip."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    streams = json.loads(out.stdout)["streams"]
    num, den = next(s["r_frame_rate"] for s in streams if s["codec_type"] == "video").split("/")
    return int(num) / int(den), any(s["codec_type"] == "audio" for s in streams)


def rife_inputs(frames: list) -> dict:
    """The RIFE VFI node's inputs (also appended to the cloud i2v graph by animate_pipeline.add_rife)."""
    return {"frames": frames, "ckpt_name": "rife49.pth", "clear_cache_after_n_frames": 10, "multiplier": MULTIPLIER, "fast_mode": True,
            "ensemble": True, "scale_factor": 1.0, "dtype": "float32", "torch_compile": False, "batch_size": 1}


def video_inputs(images: list, fps: float, audio: list = None) -> dict:
    """CreateVideo inputs for the interpolated frames: fps x MULTIPLIER, the source audio when there is some."""
    return {"images": images, "fps": fps * MULTIPLIER, "bit_depth": 8, "color_space": "sRGB", "codec": "none",
            **({"audio": audio} if audio else {})}


def graph(name: str, fps: float, audio: bool) -> dict:
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": name}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "RIFE VFI", "inputs": rife_inputs(["2", 0])},
        "4": {"class_type": "CreateVideo", "inputs": video_inputs(["3", 0], fps, ["2", 1] if audio else None)},
        "5": {"class_type": "SaveVideo", "inputs": {"video": ["4", 0], "filename_prefix": "video/interpolated", "format": "auto",
                                                    "format.codec": "auto", "codec": "auto"}},
    }


def interpolate(clip: Path, dest: Path, server: str, label: str, timeout: float = 1800.0):
    """Render the interpolated clip to dest; returns ComfyUI's processing seconds (None if unknown)."""
    fps, audio = probe(clip)
    name = local.upload(server, clip, f"interp_{clip.parent.parent.name}_{clip.name}")
    pid = local.submit(server, graph(name, fps, audio), label)
    deadline = time.time() + timeout
    while time.time() < deadline:
        hist = json.loads(local._call(server, f"/history/{pid}")).get(pid)
        if hist and hist["status"].get("completed") is not None:
            if hist["status"]["status_str"] != "success":
                raise RuntimeError(f"local job failed: {json.dumps(hist['status'])[:800]}")
            out = hist["outputs"]["5"]
            f = next(iter(out.get("images") or out.get("videos") or out.get("gifs")))
            q = f"filename={f['filename']}&subfolder={f.get('subfolder', '')}&type={f.get('type', 'output')}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(local._call(server, f"/view?{q}"))
            return local.processing_seconds(hist)
        time.sleep(2)
    raise TimeoutError(f"local job {pid} did not finish in {timeout:.0f}s")
