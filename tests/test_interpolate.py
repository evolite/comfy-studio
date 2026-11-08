"""RIFE interpolation step: the graph (fps doubled, source audio carried over) and which clips `project.py interpolate` picks."""
import os
import subprocess
import sys

import config
import interpolate
from conftest import ROOT


def run(*args):
    return subprocess.run([sys.executable, str(ROOT / "project.py"), *args], capture_output=True, text=True, env=os.environ)


def clip(path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=64x96:rate={fps}:duration=0.5", str(path)], check=True)


def test_graph_doubles_the_fps_and_keeps_audio_only_when_there_is_some():
    g = interpolate.graph("in.mp4", 24.0, audio=True)
    assert g["3"]["class_type"] == "RIFE VFI" and g["3"]["inputs"]["multiplier"] == 2
    assert g["4"]["inputs"]["fps"] == 48.0 and g["4"]["inputs"]["audio"] == ["2", 1]
    assert "audio" not in interpolate.graph("in.mp4", 24.0, audio=False)["4"]["inputs"]


def test_cli_picks_clips_without_an_interp_result_and_rejects_missing_ones():
    name = "20990101_interp"
    assert run(name, "new").returncode == 0
    d = config.PIPELINES_DIR / name
    clip(d / "videos" / "stage_1.mp4", 24)
    clip(d / "interp" / "videos" / "stage_1.mp4", 48)
    assert interpolate.probe(d / "interp" / "videos" / "stage_1.mp4") == (48.0, False)
    r = run(name, "interpolate")  # stage 1 already has an interp/ result: nothing to do, nothing sent to ComfyUI
    assert r.returncode == 0, r.stderr
    r = run(name, "interpolate", "2")
    assert r.returncode != 0 and "stages [2]: no videos/stage_N.mp4" in r.stderr
    r = run(name, "interpolate", "interp")
    assert r.returncode != 0 and "interp/ holds the results" in r.stderr


def test_the_edit_runs_at_48_fps_only_when_every_clip_is_interpolated(tmp_path):
    import make_edit as mt
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    clip(a, 48)
    clip(b, 24)
    assert mt.edit_fps([a, a]) == mt.HI_FPS == 48
    assert mt.edit_fps([a, b]) == 30
    assert mt.edit_fps([a, tmp_path / "still.png"]) == 48  # stills loop at any rate: a mixed interp edit keeps 48
    assert mt.edit_fps([tmp_path / "still.png"]) == 30
