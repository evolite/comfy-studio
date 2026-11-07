import json
import os
import subprocess
import sys
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, str(ROOT / "project.py"), *args], capture_output=True, text=True, env=os.environ)


def test_new_project_is_created_in_the_data_dir_and_listed_by_status():
    r = run("t_roundtrip", "new")
    assert r.returncode == 0, r.stderr
    pj = config.PIPELINES_DIR / "t_roundtrip" / "project.json"
    assert pj.is_file()
    assert not (ROOT / "pipelines").exists() and not (ROOT / "data" / "pipelines" / "t_roundtrip").exists()  # nothing leaked into the real tree
    cfg = json.loads(pj.read_text())
    assert cfg["engine"] == "full" and len(cfg["stages"]) == 2
    s = run("t_roundtrip", "status")
    assert s.returncode == 0 and "t_roundtrip" in s.stdout


def test_show_dumps_the_stage_its_outputs_and_rejects_unknown_ids():
    assert run("t_show", "new").returncode == 0
    d = config.PIPELINES_DIR / "t_show"
    (d / "stage_2.png").write_bytes(b"png")
    r = run("t_show", "show", "2")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["ref"] == "@t_show" and out["stage_ids"] == [1, 2] and list(out["stages"]) == ["2"]
    s = out["stages"]["2"]
    assert s["stage"]["id"] == 2 and s["outputs"] == {"root": {"still": {"path": "stage_2.png", "mtime": s["outputs"]["root"]["still"]["mtime"]}}}
    assert s["video_engine"] == "minimax"
    assert set(json.loads(run("t_show", "show").stdout)["stages"]) == {"1", "2"}
    bad = run("t_show", "show", "9")
    assert bad.returncode != 0 and "no stage 9" in bad.stderr
