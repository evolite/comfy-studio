"""Steps run through the runner on both targets: local against the fake ComfyUI server, cloud against a monkeypatched client."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import config
import project as P
import runner
import steps
import targets
from fake_comfy import PNG, FakeComfy
from targets import cloud, local

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fake():
    f = FakeComfy()
    yield f
    f.close()


@pytest.fixture
def proj(fake):
    """A fresh full-engine project with two filled-in stages and a preview for stage 1."""
    name = f"t_targets_{os.urandom(3).hex()}"
    assert subprocess.run([sys.executable, str(ROOT / "project.py"), name, "new"], capture_output=True, env=os.environ).returncode == 0
    pj = config.PIPELINES_DIR / name / "project.json"
    cfg = json.loads(pj.read_text())
    cfg["settings"]["krea_unet"] = "krea2_turbo_bf16.safetensors"
    for s in cfg["stages"]:
        s.update(krea_scene="a woman on a bed", il_scene="1girl, solo")
    pj.write_text(json.dumps(cfg))
    d = pj.parent
    (d / "preview").mkdir(exist_ok=True)
    for n in (1, 2):
        (d / "preview" / f"stage_{n}.png").write_bytes(PNG)
    (d / "IL").mkdir()
    (d / "IL" / "stage_1.png").write_bytes(PNG)
    ctx = P.make_ctx(name)
    t = targets.get("local")
    for step in steps.STEPS.values():  # the fake server knows every node class these projects' graphs use
        if step.supports("local", ctx.engine_name):
            fake.classes |= {n["class_type"] for n in t.prepare(step.graph(ctx, ctx.stages[1], "x", t)).values()}
    return ctx


@pytest.mark.parametrize("step,folder,needs_files", [("preview", "preview", False), ("il", "IL", True), ("hires", "hires", True)])
def test_local_target_runs_each_step_into_its_folder(fake, proj, step, folder, needs_files):
    target = targets.get("local", server=fake.url)
    failed = runner.run_step(steps.STEPS[step], target, proj, [1, 2])
    assert failed == []
    assert (proj.d / folder / "stage_1.png").read_bytes() == PNG and (proj.d / folder / "stage_2.png").is_file()
    assert (proj.d / folder / "_debug" / "stage_1" / "graph.json").is_file()
    sent = fake.prompts[-1]
    assert all(n["class_type"] != "Film Grain" for n in sent.values())      # localised: no cloud-only node left
    if needs_files:
        assert f"{step}_{proj.name}_2.png" in fake.uploads                    # the input still was uploaded under a per-stage name


def test_finish_starts_from_the_chosen_upscale(fake, proj):
    (proj.d / "hires").mkdir()
    (proj.d / "hires" / "stage_2.png").write_bytes(PNG)
    assert [steps.finish_source(proj, n) for n in (1, 2)] == ["IL", "hires"]  # only IL / hires preferred over IL
    (proj.d / "IL" / "stage_2.png").write_bytes(PNG)
    proj.stages[2]["done"] = ["preview", "IL"]
    assert steps.finish_source(proj, 2) == "IL"                                 # the ✓'d upscale wins
    proj.finish_from = "hires"
    assert steps.finish_source(proj, 2) == "hires"                              # --from wins over both
    proj.finish_from = None
    assert runner.run_step(steps.STEPS["finish"], targets.get("local", server=fake.url), proj, [1, 2]) == []
    assert (proj.d / "finish" / "stage_2.png").read_bytes() == PNG


def test_local_face_replaces_the_il_image_and_keeps_the_old_one(fake, proj):
    target = targets.get("local", server=fake.url)
    (proj.d / "IL" / "stage_1.png").write_bytes(b"old")
    assert runner.run_step(steps.STEPS["face"], target, proj, [1]) == []
    assert (proj.d / "IL" / "stage_1.png").read_bytes() == PNG
    assert [p.read_bytes() for p in (proj.d / "_debug" / "history" / "IL" / "stage_1").iterdir()] == [b"old"]


def test_local_reports_the_failed_stage_and_still_finishes_the_others(fake, proj):
    fake.fail_next = True
    failed = runner.run_step(steps.STEPS["preview"], targets.get("local", server=fake.url), proj, [1, 2])
    assert failed == [1] and not (proj.d / "preview" / "stage_1.png").read_bytes() == b"" and (proj.d / "preview" / "stage_2.png").is_file()


def test_local_dry_run_queues_nothing(fake, proj):
    n = len(fake.prompts)
    runner.run_step(steps.STEPS["il"], targets.get("local", server=fake.url), proj, [1], dry_run=True)
    assert len(fake.prompts) == n


def test_unreachable_local_exits_with_a_hint(proj):
    t = targets.get("local", server="http://127.0.0.1:9")
    with pytest.raises(SystemExit, match="can't reach the local ComfyUI"):
        runner.run_step(steps.STEPS["preview"], t, proj, [1])


def test_local_prepare_maps_model_names_and_the_krea_stand_in(proj):
    t = targets.get("local")
    g = t.prepare(steps.STEPS["preview"].graph(proj, proj.stages[1], None, t))
    assert g["30:10"]["class_type"] == "CheckpointLoaderSimple" and g["30:10"]["inputs"]["ckpt_name"] == local.local_models()["krea_checkpoint"]
    il = t.prepare(steps.STEPS["il"].graph(proj, proj.stages[1], "x", t))
    assert any(str(n["inputs"].get("ckpt_name", "")).startswith("IL/") for n in il.values())


def test_step_support_matrix():
    assert not steps.STEPS["hires"].supports("cloud", "full") and steps.STEPS["hires"].supports("local", "full")
    assert not steps.STEPS["il"].supports("local", "krea2") and not steps.STEPS["main"].supports("local", "full")


# --- cloud, with the HTTP client replaced --------------------------------------------------------------------------
@pytest.fixture
def fake_cloud(monkeypatch):
    sent = []

    def submit(graph):
        sent.append(graph)
        return f"c{len(sent)}", {"prompt_id": f"c{len(sent)}"}
    monkeypatch.setattr(cloud, "submit", submit)
    monkeypatch.setattr(cloud, "job_state", lambda pid: ("success", {"status": "success"}))
    monkeypatch.setattr(cloud, "fetch_job", lambda pid: {"outputs": {"436": {"images": [{"filename": f"{pid}.png", "subfolder": "", "type": "output"}]}}})
    monkeypatch.setattr(cloud, "_request", lambda *a, **k: PNG)
    monkeypatch.setattr(cloud.timings, "record_cloud", lambda *a: None)
    monkeypatch.setattr(cloud, "upload_image", lambda path: f"cloud_{path.name}")
    return sent


def test_cloud_target_runs_a_step_and_downloads_each_stage(fake_cloud, proj):
    assert runner.run_step(steps.STEPS["preview"], targets.get("cloud"), proj, [1, 2]) == []
    assert (proj.d / "preview" / "stage_1.png").read_bytes() == PNG and len(fake_cloud) == 2


def test_cloud_il_uploads_the_preview_and_replaces_krea(fake_cloud, proj):
    runner.run_step(steps.STEPS["il"], targets.get("cloud"), proj, [1])
    graph = fake_cloud[-1]
    assert graph["29"]["inputs"]["image"] == "cloud_il_" + proj.name + "_1.png" or graph["29"]["inputs"]["image"].startswith("cloud_")
    assert not any(k.startswith("30:") for k in graph)


def test_cloud_refuses_a_local_only_unet_before_queuing(fake_cloud, proj):
    proj.stages[1]["krea_unet"] = local.local_models()["local_only_unets"][0]
    cloud = targets.get("cloud")
    with pytest.raises(SystemExit, match="exists only on this machine"):
        runner.run_step(steps.STEPS["preview"], cloud, proj, [1])
    assert fake_cloud == []


def test_same_models_still_free_when_memory_is_tight(monkeypatch):
    from targets import local
    monkeypatch.setattr(local, "same_as_last", lambda server, graph: True)
    freed = []
    monkeypatch.setattr(local, "free", lambda server: freed.append(server))
    stats = {"system": {"ram_free": 8 * 2 ** 30}}
    monkeypatch.setattr(local, "_call", lambda server, path, *a: json.dumps(stats).encode())
    monkeypatch.setattr(local, "swap_used_pct", lambda: 10.0)
    assert local.free_if_changed("http://127.0.0.1:8188", {}) is False and not freed
    monkeypatch.setattr(local, "swap_used_pct", lambda: 60.0)
    assert local.free_if_changed("http://127.0.0.1:8188", {}) is True
    assert local.free_if_changed("http://gpu-box:8188", {}) is False  # swap is only measured for this machine
    stats["system"]["ram_free"] = 1 * 2 ** 30
    assert local.free_if_changed("http://gpu-box:8188", {}) is True and len(freed) == 2
