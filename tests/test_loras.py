"""The LoRA library (loras.json): registry reads/writes, local path mapping, the cloud refusal of local-only LoRAs, the API."""
import json
import shutil

import pytest
from fastapi.testclient import TestClient

import config
import lora_registry


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """A scratch copy of the test library (tests/fixtures/library), so tests can edit it."""
    f = tmp_path / "loras.json"
    shutil.copy(config.LORAS_FILE, f)
    monkeypatch.setattr(config, "LORAS_FILE", f)
    return f


def test_test_and_starter_libraries_are_valid(tmp_path, monkeypatch):
    lora_registry.validate(lora_registry.load())
    monkeypatch.setattr(config, "LORAS_FILE", tmp_path / "loras.json")  # not there yet: the defaults/ starter is read
    starter = lora_registry.load()
    lora_registry.validate(starter)
    assert "background_detail" in starter["image"]


def test_save_writes_the_library_copy_and_keeps_backups(tmp_path, monkeypatch):
    f = tmp_path / "library" / "loras.json"
    stamps = iter(range(10))
    monkeypatch.setattr(config, "LORAS_FILE", f)
    monkeypatch.setattr(config, "LIBRARY_BACKUPS", 2)
    monkeypatch.setattr(config.time, "strftime", lambda fmt: f"t{next(stamps)}")
    data = lora_registry.load()  # nothing there yet: the defaults/ starter
    for i in range(4):
        data["image"][f"n{i}"] = {"krea": f"n{i}.safetensors"}
        lora_registry.save(data)
    assert "n3" in lora_registry.load()["image"]
    assert sorted(p.name for p in (f.parent / "_bak").iterdir()) == ["loras.json.t1", "loras.json.t2"]  # the newest 2 kept


def test_save_keeps_entry_order_and_the_pipeline_sees_edits(lib):
    import animate_pipeline
    import image_loras
    data = lora_registry.load()
    order = list(data["image"])
    data["image"]["zz_new"] = {"krea": "zz.safetensors", "cloud": False}
    data["video"]["vv"] = {"file": "vv.safetensors"}
    lora_registry.save(data)
    assert list(lora_registry.load()["image"]) == order + ["zz_new"]
    assert image_loras.IMAGE_LORAS["zz_new"]["krea"] == "zz.safetensors"
    assert animate_pipeline.LORA_FILES["vv"] == "vv.safetensors"
    assert image_loras.check_loras({"zz_new": 0.5}, "t") == {"zz_new": 0.5}


@pytest.mark.parametrize("kind,name,spec", [
    ("image", "Bad-Name", {"krea": "a.safetensors"}),
    ("image", "nofile", {"note": "x"}),
    ("image", "ext", {"krea": "a.ckpt"}),
    ("video", "typo", {"file": "a.safetensors", "sugested": 1}),
    ("video", "num", {"file": "a.safetensors", "suggested": "1"}),
    ("image", "cat", {"krea": "a.safetensors", "category": "pose"}),
    ("video", "halfslider", {"file": "a.safetensors", "min": -4}),
    ("video", "backslider", {"file": "a.safetensors", "min": 4, "max": -4}),
])
def test_validate_rejects(lib, kind, name, spec):
    data = lora_registry.load()
    data[kind][name] = spec
    with pytest.raises(ValueError):
        lora_registry.save(data)
    assert name not in lora_registry.load()[kind]  # nothing written


def test_slider_range_is_saved_and_a_negative_strength_passes(lib):
    import animate_pipeline
    data = lora_registry.load()
    data["video"]["sl"] = {"file": "sl.safetensors", "min": -4, "max": 4, "suggested": -1.2}
    lora_registry.save(data)
    assert lora_registry.load()["video"]["sl"]["min"] == -4
    assert animate_pipeline.check_loras({"sl": -1.2}, "t", "minimax") == {"sl": -1.2}


def test_localize_maps_krea_and_il_loras_to_their_own_folders():
    from targets import local
    lora = lambda f: {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": f, "strength_model": 1, "model": ["x", 0]}}  # noqa: E731
    g = {"30:10": {"class_type": "UNETLoader", "inputs": {"unet_name": "krea.safetensors"}},
         "30:12": {"class_type": "VAELoader", "inputs": {"vae_name": "v.safetensors"}},
         "30:300": lora("k.safetensors"), "433:300": lora("i.safetensors"), "face:300": lora("sub/f.safetensors")}
    out = local.localize(g)
    m = local.local_models()
    assert out["30:300"]["inputs"]["lora_name"] == f"{m['krea_lora_folder']}k.safetensors"
    assert out["433:300"]["inputs"]["lora_name"] == f"{m['il_lora_folder']}i.safetensors"
    assert out["face:300"]["inputs"]["lora_name"] == "sub/f.safetensors"


def test_cloud_refuses_a_local_only_lora(lib):
    from targets import cloud
    data = lora_registry.load()
    data["image"]["mine"] = {"krea": "mine.safetensors", "cloud": False}
    lora_registry.save(data)
    g = {"30:300": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "mine.safetensors"}, "_meta": {"title": "Load LoRA (mine)"}},
         "30:301": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "detail_k.safetensors"}, "_meta": {"title": "Load LoRA (detail)"}}}
    problems = cloud.CloudTarget().check(g)
    assert len(problems) == 1 and "mine" in problems[0] and "local-only" in problems[0]


def test_api_library_discovery_edit_and_delete(lib, monkeypatch):
    import loras
    from app import app
    c = TestClient(app)
    monkeypatch.setattr(loras, "local_files", lambda: ["krea/detail_k.safetensors", "krea/new_style-trigger-abc.safetensors",
                                                       "IL/other.safetensors", "root.safetensors"])
    d = c.get("/api/loras").json()
    assert d["local_up"] and d["image"]["detail"]["local"] == {"krea": True, "il": False}
    found = {f["path"]: f for f in d["discovered"]}
    assert set(found) == {"krea/new_style-trigger-abc.safetensors", "IL/other.safetensors", "root.safetensors"}
    new = found["krea/new_style-trigger-abc.safetensors"]
    assert new["pass"] == "krea" and new["trigger"] == "abc" and new["files"]["krea"] == "new_style-trigger-abc.safetensors"
    assert found["IL/other.safetensors"]["pass"] == "il" and found["root.safetensors"]["pass"] is None

    assert c.put("/api/loras/image/new_style", json={"krea": new["files"]["krea"], "cloud": False, "note": ""}).status_code == 200
    assert "note" not in lora_registry.load()["image"]["new_style"]  # blank fields are dropped
    assert c.put("/api/loras/image/bad", json={"krea": "x.bin"}).status_code == 400
    assert c.post("/api/loras/ignore", json={"file": "root.safetensors"}).json()["ignore"] == ["root.safetensors"]
    paths = {f["path"] for f in c.get("/api/loras").json()["discovered"]}
    assert paths == {"IL/other.safetensors"}

    p = config.PIPELINES_DIR / "lora_use"
    p.mkdir(parents=True, exist_ok=True)
    (p / "project.json").write_text(json.dumps({"engine": "full", "stages": [{"id": 3, "loras": {"new_style": 0.7}}]}))
    try:
        assert c.get("/api/loras").json()["image"]["new_style"]["used"] == [{"project": "lora_use", "stage": 3}]
        assert c.delete("/api/loras/image/new_style").status_code == 409
        r = c.delete("/api/loras/image/new_style?strip=true").json()
        assert r["stripped"] == [{"project": "lora_use", "stage": 3}]
        assert json.loads((p / "project.json").read_text())["stages"][0]["loras"] == {}
    finally:
        shutil.rmtree(p)
    assert "new_style" not in lora_registry.load()["image"]
    assert c.delete("/api/loras/image/new_style").status_code == 404


def test_video_engines_match_the_animate_table():
    import animate_pipeline
    assert set(lora_registry.VIDEO_ENGINES) == set(animate_pipeline.VIDEO_ENGINES)


def test_image_lora_fields_cover_every_image_engine():
    import image_loras
    import run_pipeline
    assert set(image_loras.ENGINE_FIELDS) == set(run_pipeline.ENGINES)


def test_video_lora_engine_is_checked_when_animating(lib):
    import animate_pipeline
    data = lora_registry.load()
    data["video"]["ltx_only"] = {"file": "l.safetensors", "engine": "ltx"}
    lora_registry.save(data)
    assert animate_pipeline.check_loras({"ltx_only": 1.0}, "t", "ltx")
    assert animate_pipeline.check_loras({"ltx_only": 0}, "t", "minimax")  # off: fine
    with pytest.raises(SystemExit, match="for the ltx model"):
        animate_pipeline.check_loras({"ltx_only": 1.0}, "t", "minimax")
    data["video"]["bad"] = {"file": "b.safetensors", "engine": "nope"}
    with pytest.raises(ValueError, match="engine"):
        lora_registry.validate(data)


def test_video_trigger_is_put_in_front_of_the_motion_once(lib):
    data = lora_registry.load()
    data["video"]["tv"] = {"file": "tv.safetensors", "trigger": "trg"}
    lora_registry.save(data)
    assert lora_registry.with_video_triggers("she moves", {"tv": 1}) == "trg\n\nshe moves"
    assert lora_registry.with_video_triggers("trg\n\nshe moves", {"tv": 1}) == "trg\n\nshe moves"
    assert lora_registry.with_video_triggers("she moves", {"tv": 0}) == "she moves"


def test_recover_saves_an_orphaned_clip_and_skips_live_runs(tmp_path, monkeypatch):
    import json
    import os
    import animate_pipeline as a
    pend = tmp_path / "_debug" / a.PENDING
    pend.mkdir(parents=True)
    (pend / "stage_1.json").write_text(json.dumps({"prompt_id": "p1", "stage": 1, "engine": "minimax", "loras": {}, "seed": 5, "audio": False, "rife_dir": None, "pid": 0}))
    (pend / "stage_2.json").write_text(json.dumps({"prompt_id": "p2", "stage": 2, "engine": "minimax", "loras": {}, "seed": 5, "audio": False, "rife_dir": None, "pid": os.getpid()}))
    monkeypatch.setattr(a, "_alive", lambda pid: pid == os.getpid())
    monkeypatch.setattr(a.api, "job_state", lambda pid: ("success", None))
    monkeypatch.setattr(a.api, "fetch_job", lambda pid: {"id": pid})
    saved = []
    monkeypatch.setattr(a, "save_clip", lambda job, n, *rest: saved.append((job["id"], n)))
    assert a.recover(tmp_path) == 0
    assert saved == [("p1", 1)]
    assert not (pend / "stage_1.json").exists() and (pend / "stage_2.json").exists()  # the live run saves its own


def test_each_engine_chains_only_its_own_files():
    import generate_krea2
    import generate_moss
    def chained(graph):
        return sorted(n["inputs"]["lora_name"] for k, n in graph.items() if k.startswith("30:3") and n["class_type"] == "LoraLoaderModelOnly")
    loras = {"realistic_snapshot": 1.5, "background_detail": 0.5}  # background_detail has no MOSS file
    assert chained(generate_moss.build_graph("a photo", seed=1, loras=loras)) == ["moss/MOSSRealisticSnapshotV1.safetensors"]
    assert chained(generate_krea2.build_graph("a photo", seed=1, loras=loras)) == ["RealisticSnapshotKrea2V2.safetensors", "zy_BackgroundDetail_K2.safetensors"]


def test_combos_validate_and_follow_lora_deletes(lib):
    import loras as L
    data = lora_registry.load()
    assert data["combos"]["finish"]["loras"] == {"unlock": 1.0, "burst": 0.7, "motion_plus": 0.7}
    for bad in ({"kind": "video", "loras": {}}, {"kind": "video", "loras": {"nope": 1}}, {"kind": "image", "loras": {"burst": 1}},
                {"kind": "video", "loras": {"burst": "1"}}, {"kind": "video", "loras": {"burst": 1}, "extra": 1}):
        data = lora_registry.load()
        data["combos"]["bad"] = bad
        with pytest.raises(ValueError):
            lora_registry.validate(data)
    data = lora_registry.load()
    data["video"]["solo"] = {"file": "solo.safetensors"}
    data["combos"]["solo_only"] = {"kind": "video", "loras": {"solo": 1}}
    data["combos"]["pair"] = {"kind": "video", "loras": {"solo": 1, "burst": 0.7}}
    lora_registry.save(data)
    L.delete("video", "solo")
    combos = lora_registry.load()["combos"]
    assert "solo_only" not in combos and combos["pair"]["loras"] == {"burst": 0.7}
