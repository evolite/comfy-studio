"""Preview takes: several previews per stage kept with a record of what made each, one of them current (pipeline/takes.py)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import config
import project as P
import runner
import steps
import takes
import targets
from fake_comfy import PNG, FakeComfy

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fake():
    f = FakeComfy()
    yield f
    f.close()


@pytest.fixture
def proj(fake):
    """A full-engine project with two filled-in stages and no previews yet."""
    name = f"t_takes_{os.urandom(3).hex()}"
    assert subprocess.run([sys.executable, str(ROOT / "project.py"), name, "new"], capture_output=True, env=os.environ).returncode == 0
    pj = config.PIPELINES_DIR / name / "project.json"
    cfg = json.loads(pj.read_text())
    cfg["settings"]["krea_unet"] = "krea2_turbo_bf16.safetensors"
    for s in cfg["stages"]:
        s.update(krea_scene="a woman on a bed", il_scene="1girl, solo")
    pj.write_text(json.dumps(cfg))
    ctx = P.make_ctx(name)
    t = targets.get("local")
    for step in steps.STEPS.values():
        if step.supports("local", ctx.engine_name):
            fake.classes |= {n["class_type"] for n in t.prepare(step.graph(ctx, ctx.stages[1], "x", t)).values()}
    return ctx


def preview(fake, ctx, numbers, k=1, **extra):
    vars(ctx).update(extra)
    return runner.run_step(steps.STEPS["preview"], targets.get("local", server=fake.url), ctx, numbers, takes=k)


def test_takes_are_kept_round_robin_with_fresh_seeds_and_the_first_is_current(fake, proj, capsys):
    assert preview(fake, proj, [1, 2], k=3) == []
    out = capsys.readouterr().out
    assert out.index("[stage_1.t1] done") < out.index("[stage_2.t1] done") < out.index("[stage_1.t2] done")  # round-robin
    assert "stage 1: 3/3 takes" in out
    for n in (1, 2):
        rows = takes.listing(proj.d, n)
        assert len(rows) == 3 and sum(r["current"] for r in rows) == 1
        cur = takes.current_id(proj.d, n)
        assert (proj.d / "preview" / f"stage_{n}.png").read_bytes() == (takes.stage_dir(proj.d, n) / f"{cur}.png").read_bytes()
        seeds = {takes.load_record(proj.d, n, r["id"])["seed"] for r in rows}
        assert len(seeds) == 3
        assert {r["delta"] for r in rows if not r["current"]} == {"seed only"}  # same prompt and settings, only the noise differs


def test_every_new_take_becomes_current_and_displacing_one_resets_its_done(fake, proj):
    preview(fake, proj, [1])
    first = takes.current_id(proj.d, 1)
    proj.stages[1]["done"] = ["preview", "il"]
    preview(fake, proj, [1], k=2)
    rows = takes.listing(proj.d, 1)
    assert len(rows) == 3
    assert takes.current_id(proj.d, 1) != first  # every new render becomes current, even with the preview ✓'d
    done = json.loads((proj.d / "project.json").read_text())["stages"][0].get("done")
    assert done == ["preview"]  # a different take took over: everything made from the old one lost its ✓


def test_one_failed_take_does_not_fail_the_stage(fake, proj, monkeypatch, capsys):
    real = targets.local.LocalTarget.collect
    def collect(self, jobs):
        for j in real(self, jobs):
            if j.tag == "stage_1.t2":
                j.final, j.error = None, "boom"
            yield j
    monkeypatch.setattr(targets.local.LocalTarget, "collect", collect)
    assert preview(fake, proj, [1], k=3) == []
    assert "stage 1: 2/3 takes" in capsys.readouterr().out and len(takes.listing(proj.d, 1)) == 2


def test_external_and_legacy_files_become_takes(proj):
    d = proj.d
    (d / "preview").mkdir(exist_ok=True)
    (d / "preview" / "stage_1.png").write_bytes(PNG)
    takes.sync(d, 1)
    rows = takes.listing(d, 1)
    assert len(rows) == 1 and rows[0]["source"] == "legacy" and rows[0]["current"]
    time.sleep(0.01)
    (d / "preview" / "stage_1.png").write_bytes(PNG + b"x")  # replaced by hand: the canonical file wins
    takes.sync(d, 1)
    rows = takes.listing(d, 1)
    assert len(rows) == 2 and rows[0]["current"] and rows[0]["source"] == "external"
    takes.sync(d, 1)
    assert len(takes.listing(d, 1)) == 2  # stable once in sync


def test_diff_and_delta():
    base = {"seed": 1, "unet": "a", "sampler": "euler", "scheduler": "simple", "steps": 8, "cfg": 1, "aspect_ratio": "2:3",
            "megapixels": 1, "target": "local", "prompt": "a woman on a bed", "loras": {"style_a": 1.0}, "settings": {}}
    other = {**base, "seed": 2, "scheduler": "beta", "prompt": "a woman on a red bed", "loras": {"style_a": 1.0, "detail": 1.0}}
    d = takes.diff(other, base)
    assert [f["key"] for f in d["fields"]] == ["seed", "scheduler"]
    assert d["loras"] == [{"name": "detail", "this": 1.0, "current": None}]
    assert ["+", "red "] in d["prompt"]
    assert takes.delta(other, base) == "scheduler beta · +detail 1 · prompt edited"
    assert takes.delta({**base, "seed": 9}, base) == "seed only" and takes.delta(None, base) == ""
    stronger = {**base, "loras": {"style_a": 0.6}, "cfg": 1.5, "settings": {"krea_cfg": 1.5}, "unet": "mymodelV5_x.safetensors"}
    assert takes.delta(stronger, base) == "UNet mymodelV5 · cfg 1.5 · style_a 1→0.6"  # krea_cfg is not listed twice
    assert takes.delta({**base, "target": None}, base) == "same"  # unrecorded (backfilled) fields don't count as changes


def test_a_legacy_preview_gets_its_record_from_the_last_run(fake, proj):
    preview(fake, proj, [1, 2])
    for n in (1, 2):  # as before takes: no takes folder, just the canonical file and the run's _debug
        import shutil
        shutil.rmtree(takes.stage_dir(proj.d, n))
    (proj.d / "preview" / "stage_2.png").write_bytes(PNG + b"other")  # not the image that run made
    for n in (1, 2):
        takes.sync(proj.d, n)
    rec = takes.load_record(proj.d, 1, takes.current_id(proj.d, 1))
    assert rec and rec["prompt"] and rec["loras"] == {} and rec["seed"]  # no LoRA is on by default
    assert takes.load_record(proj.d, 2, takes.current_id(proj.d, 2)) is None
    preview(fake, proj, [1])
    assert {r["delta"] for r in takes.listing(proj.d, 1) if not r["current"]} <= {"seed only", "same"}  # the same ctx keeps its seed; no false LoRA or setting changes


def test_use_take_api_and_downstream_checks(fake, proj):
    from fastapi.testclient import TestClient
    from app import app
    import projects as WP
    c = TestClient(app)
    preview(fake, proj, [1], k=2)
    rows = c.get(f"/api/projects/{proj.name}").json()["folders"]["preview"]["takes"]["1"]
    other = next(r for r in rows if not r["current"])
    WP.patch_stage(proj.name, 1, {"done": ["preview", "hires", "finish"]})
    info = c.get(f"/api/projects/{proj.name}/stages/1/takes/{other['id']}").json()
    assert info["diff"]["fields"][0]["key"] == "seed" and info["record"]["prompt"]
    r = c.post(f"/api/projects/{proj.name}/stages/1/takes/{other['id']}/use").json()
    assert r == {"changed": True, "done": ["preview"]}  # what was made from the old take is no longer approved
    assert takes.current_id(proj.d, 1) == other["id"]
    assert c.post(f"/api/projects/{proj.name}/stages/1/takes/clear").json() == {"moved": 1}
    assert len(takes.listing(proj.d, 1)) == 1
    assert c.get(f"/api/projects/{proj.name}/stages/1/takes/nope").status_code == 404


def test_hires_records_its_source_take_and_is_flagged_when_it_changes(fake, proj):
    import projects as WP
    preview(fake, proj, [1], k=2)
    assert runner.run_step(steps.STEPS["hires"], targets.get("local", server=fake.url), proj, [1]) == []
    src = json.loads((proj.d / "hires" / "_debug" / "stage_1" / "source.json").read_text())["take"]
    assert src == takes.current_id(proj.d, 1)
    assert WP.get(proj.name)["folders"]["hires"]["stale"] == []
    other = next(r["id"] for r in takes.listing(proj.d, 1) if not r["current"])
    WP.use_take(proj.name, 1, other)
    assert WP.get(proj.name)["folders"]["hires"]["stale"] == [1]


def test_build_argv_passes_takes():
    import jobs as J
    import projects as WP
    name = f"t_takes_argv_{os.urandom(3).hex()}"
    assert subprocess.run([sys.executable, str(ROOT / "project.py"), name, "new"], capture_output=True, env=os.environ).returncode == 0
    argv = J.Manager().build_argv(name, "preview_local", {"stages": [1], "takes": 4})
    assert argv[-3:] == ["--takes", "4", "1"]
    assert "--takes" not in J.Manager().build_argv(name, "preview_local", {"stages": [1]})
    assert WP.history(name, 1) == []


def test_a_project_without_a_preview_folder_loads():
    import projects as WP
    name = f"t_takes_new_{os.urandom(3).hex()}"
    assert subprocess.run([sys.executable, str(ROOT / "project.py"), name, "new"], capture_output=True, env=os.environ).returncode == 0
    assert WP.get(name)["folders"]["preview"]["takes"] == {}  # loads, with no takes yet
