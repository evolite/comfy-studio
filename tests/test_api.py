from fastapi.testclient import TestClient

import config


def client():
    from app import app  # imported late: config must already point at the temp data dir
    return TestClient(app)


def test_health_and_empty_project_list():
    c = client()
    h = c.get("/api/health").json()
    assert h["local_server"] == config.LOCAL_COMFY_URL and "cloud_key" in h
    assert c.get("/api/projects").json() == []


def test_edit_timing_starts_empty():
    assert client().get("/api/edit-timing").json() == {}


def test_audio_list_reads_only_the_audio_dir():
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    (config.AUDIO_DIR / "x.mp3").write_bytes(b"")
    (config.AUDIO_DIR / "x.json").write_text("{}")
    a = client().get("/api/audio").json()
    assert a["tracks"] == [str(config.AUDIO_DIR / "x.mp3")] and a["beat_files"] == ["audio/x.json"]


def test_edit_args_mix_stills_into_a_clip_edit(tmp_path):
    import pytest
    from jobs import Manager
    from projects import ApiError
    (tmp_path / "finish").mkdir()
    argv = Manager.edit_args(tmp_path, "p", {"mode": "animate", "still_stages": [2, 4], "stills_from": "finish"}, "interp", [1, 2, 3, 4])
    i = argv.index("--as-still")
    assert argv[i:i + 5] == ["--as-still", "2", "4", "--stills-from", str(tmp_path / "finish")]
    assert "--as-still" not in Manager.edit_args(tmp_path, "p", {"mode": "stills", "still_stages": [2]}, "", [1, 2])
    for bad in ("nope", "../finish"):
        with pytest.raises(ApiError):
            Manager.edit_args(tmp_path, "p", {"mode": "animate", "still_stages": [2], "stills_from": bad}, "", [1, 2])


def test_no_music_settings_are_saved_and_become_edit_flags(tmp_path):
    import json
    from jobs import Manager
    d = config.PIPELINES_DIR / "20261002_scen"
    d.mkdir(parents=True)
    (d / "project.json").write_text(json.dumps({"engine": "full", "settings": {}, "stages": [{"id": 1, "krea_scene": "s"}]}))
    c = client()
    r = c.put("/api/projects/20261002_scen/music", json={"none": True, "transition": "fadeblack", "transition_seconds": "0.8", "fade_in": 0.3})
    assert r.status_code == 200, r.text
    assert json.loads((d / "project.json").read_text())["music"] == {"none": True, "transition": "fadeblack", "transition_seconds": 0.8, "fade_in": 0.3}
    assert c.put("/api/projects/20261002_scen/music", json={"none": True, "transition": "swirl"}).status_code == 400
    assert c.put("/api/projects/20261002_scen/music", json={"none": False}).json() == {"music": None}  # off = back to the default track
    argv = Manager.edit_args(tmp_path, "p", {"mode": "animate", "no_music": True, "transition": "dissolve", "transition_seconds": 1,
                                             "clip_audio": True, "mute_clip_audio": [2]}, "interp", [1, 2])
    assert "--no-music" in argv and "--clip-audio" not in argv
    assert argv[argv.index("--transition"):argv.index("--transition") + 4] == ["--transition", "dissolve", "--transition-seconds", "1"]
    assert argv[argv.index("--mute-clip-audio") + 1] == "2"
    argv = Manager.edit_args(tmp_path, "p", {"mode": "animate", "trims_end": {"3": "0.4", "4": "", "5": "0"}}, "interp", [3, 4, 5])
    assert argv[argv.index("--trim-end"):] == ["--trim-end", "3=0.4"]


def test_remix_copies_picked_stages():
    import json
    src = config.PIPELINES_DIR / "20261001_src"
    src.mkdir(parents=True)
    (src / "project.json").write_text(json.dumps({"engine": "full", "settings": {"x": 1}, "stages": [{"id": i, "krea_scene": f"s{i}"} for i in (1, 2, 3)]}))
    for i in (1, 2):
        for rel in (f"preview/stage_{i}.png", f"preview/takes/stage_{i}/t.png", f"preview/takes/stage_{i}/current",
                    f"hires/videos/stage_{i}.mp4", f"hires/_debug/stage_{i}/source.json", f"hires/_debug/i2v_stage_{i}/g.json"):
            (src / rel).parent.mkdir(parents=True, exist_ok=True)
            (src / rel).write_text(rel)
    c = client()
    r = c.post("/api/projects/20261001_src/remix", json={"suffix": "x", "stages": [2]})
    assert r.status_code == 200, r.text
    new = config.PIPELINES_DIR / "20261001_src_remix_x"
    cfg = json.loads((new / "project.json").read_text())
    assert [s["id"] for s in cfg["stages"]] == [2] and cfg["settings"] == {"x": 1}
    for rel in ("preview/stage_2.png", "preview/takes/stage_2/t.png", "hires/videos/stage_2.mp4",
                "hires/_debug/stage_2/source.json", "hires/_debug/i2v_stage_2/g.json"):
        assert (new / rel).read_text() == rel
    assert (new / "preview/takes/stage_2/current").is_file()  # rewritten by the take sync in get()
    assert not list(new.rglob("*stage_1*"))
    for body in ({"suffix": "../x", "stages": [2]}, {"suffix": "x", "stages": [2]}, {"suffix": "y", "stages": []}, {"suffix": "y", "stages": [9]}):
        assert c.post("/api/projects/20261001_src/remix", json=body).status_code == 400


def test_job_flags_a_memory_check_free():
    from jobs import Job
    j = Job("j1", "p", "preview_local", ["py", "project.py"], {})
    j.emit("  same models as the last run -- keeping them loaded")
    assert j.info()["mem_freed"] is None
    j.emit("  same models as the last run, but swap 52% used -- freeing memory first")
    assert j.info()["mem_freed"] == "swap 52% used"


def test_stale_clips_flag_clips_older_than_their_source():
    import json
    import os
    d = config.PIPELINES_DIR / "20261001_stale"
    (d / "hires" / "videos").mkdir(parents=True)
    (d / "interp" / "videos").mkdir(parents=True)
    (d / "interp" / "_debug" / "stage_1").mkdir(parents=True)
    (d / "project.json").write_text(json.dumps({"engine": "full", "stages": [{"id": 1}, {"id": 2}]}))
    (d / "interp" / "_debug" / "stage_1" / "source.json").write_text(json.dumps({"folder": "hires"}))
    files = {"hires/videos/stage_1.mp4": 100, "hires/stage_1.png": 200,  # still re-rendered after its clip -> stale
             "hires/videos/stage_2.mp4": 300, "hires/stage_2.png": 200,  # clip made from the current still
             "interp/videos/stage_1.mp4": 50}  # older than its source clip -> stale
    for rel, t in files.items():
        (d / rel).write_bytes(b"x")
        os.utime(d / rel, (t, t))
    fs = client().get("/api/projects/20261001_stale").json()["folders"]
    assert fs["hires"]["stale_clips"] == [1] and fs["interp"]["stale_clips"] == [1]


def test_recover_job_is_queued_for_orphaned_cloud_clips():
    from jobs import Manager
    d = config.PIPELINES_DIR / "20261001_orphan"
    (d / "hires" / "_debug" / "i2v_pending").mkdir(parents=True)
    (d / "project.json").write_text('{"engine": "full", "stages": [{"id": 1}]}')
    (d / "hires" / "_debug" / "i2v_pending" / "stage_1.json").write_text('{"prompt_id": "x", "stage": 1, "pid": 0}')
    m = Manager()
    m._run = lambda job: None  # don't start project.py
    m.recover_orphans()
    assert [(j.project, j.action, j.argv[-1]) for j in m.jobs.values()] == [("20261001_orphan", "recover", "recover")]


def test_civitai_post_exports_still_and_clip_pairs(monkeypatch):
    import json
    from PIL import Image
    d = config.PIPELINES_DIR / "20261001_cv"
    (d / "hires").mkdir(parents=True)
    (d / "interp" / "videos").mkdir(parents=True)
    (d / "project.json").write_text(json.dumps({"engine": "full", "stages": [{"id": 1}, {"id": 2}]}))
    for i in (1, 2):
        Image.new("RGB", (8, 12)).save(d / "hires" / f"stage_{i}.png")
        (d / "interp" / "videos" / f"stage_{i}.mp4").write_bytes(b"clip")
    monkeypatch.setattr(config, "FILE_MANAGER", ["true"])
    c = client()
    r = c.post("/api/projects/20261001_cv/civitai/post", json={"items": [{"stage": 2, "still": "hires", "clip": "interp"}, {"stage": 1, "still": "hires", "clip": "interp"}]})
    assert r.status_code == 200, r.text
    out = config.CIVITAI_DIR / r.json()["path"].split("/")[-1]
    assert sorted(p.name for p in out.iterdir()) == ["01_stage_2.mp4", "01_stage_2.png", "02_stage_1.mp4", "02_stage_1.png", "clips.txt"]
    assert (out / "01_stage_2.mp4").read_bytes() == b"clip" and r.json()["pairs"] == 2
    assert c.get("/api/projects/20261001_cv/civitai/post").json()["stages"] == [2, 1]
    (out / "notes.md").write_text("mine")
    one = {"items": [{"stage": 1, "still": "hires", "clip": "interp"}]}
    assert c.post("/api/projects/20261001_cv/civitai/post", json=one).status_code == 409  # already exported: open it or overwrite
    r = c.post("/api/projects/20261001_cv/civitai/post", json={**one, "overwrite": True})
    assert r.status_code == 200 and r.json()["path"] == str(out)
    assert sorted(p.name for p in out.iterdir()) == ["01_stage_1.mp4", "01_stage_1.png", "clips.txt", "notes.md"]  # other files stay
    assert c.post("/api/projects/20261001_cv/civitai/post", json={"items": []}).status_code == 400
    assert c.post("/api/projects/20261001_cv/civitai/post", json={"items": [{"stage": 1, "still": "hires", "clip": "hires"}]}).status_code == 404


def test_beat_and_no_music_edits_are_separate_jobs_files_and_timings(tmp_path):
    import jobs
    from jobs import Job, Manager
    beat, nm = {"folder": "interp", "stages": [1, 2]}, {"folder": "interp", "stages": [1, 2], "no_music": True}
    assert Job("j1", "p", "edit", [], beat).key() != Job("j2", "p", "edit", [], nm).key()
    name = lambda req: (a := Manager.edit_args(tmp_path, "p", req, "interp", [1, 2]))[a.index("--out-name") + 1]
    assert name(beat).startswith("p_edit_") and name(nm).startswith("p_nomusic_")
    jobs.save_edit_timing("nomusic", 60, 1500)
    jobs.save_edit_timing("beat", 40, 1400)
    assert jobs.edit_timing("nomusic") == {"seconds": 60, "frames": 1500} and jobs.edit_timing("beat")["seconds"] == 40


def test_marking_a_project_finished_cleans_it_up_and_flags_it():
    import json
    d = config.PIPELINES_DIR / "20261001_fin"
    d.mkdir(parents=True)
    (d / "project.json").write_text(json.dumps({"engine": "full", "stages": [{"id": 1}]}))
    c = client()
    r = c.post("/api/projects/20261001_fin/finished", json={"finished": True})
    assert r.status_code == 200, r.text
    assert "files" in r.json()["cleanup"] and json.loads((d / "project.json").read_text())["finished"] is True
    assert next(p for p in c.get("/api/projects").json() if p["name"] == "20261001_fin")["finished"] is True
    c.post("/api/projects/20261001_fin/finished", json={"finished": False})
    assert "finished" not in json.loads((d / "project.json").read_text())


def test_civitai_post_can_export_stills_only(monkeypatch):
    import json
    from PIL import Image
    d = config.PIPELINES_DIR / "20261001_cvs"
    (d / "finish").mkdir(parents=True)
    (d / "project.json").write_text(json.dumps({"engine": "full", "stages": [{"id": 3}]}))
    Image.new("RGB", (8, 12)).save(d / "finish" / "stage_3.png")
    monkeypatch.setattr(config, "FILE_MANAGER", ["true"])
    r = client().post("/api/projects/20261001_cvs/civitai/post", json={"items": [{"stage": 3, "still": "finish"}]})
    assert r.status_code == 200, r.text
    out = config.CIVITAI_DIR / r.json()["path"].split("/")[-1]
    assert sorted(p.name for p in out.iterdir()) == ["01_stage_3.png"] and r.json()["clips"] == 0


def test_notify_pushes_each_time_the_queue_empties(monkeypatch):
    import time
    import jobs
    from jobs import Job, Manager
    sent = []
    monkeypatch.setattr(jobs, "send_pushover", lambda title, msg: sent.append(msg))

    def wait(n):
        for _ in range(100):
            if len(sent) >= n:
                return
            time.sleep(0.01)

    m = Manager()
    old = Job("j0", "p", "preview_local", [], {})
    old.status, old.ended = "done", time.time() - 60
    m.jobs = {"j0": old}
    m.arm_notify(True)  # idle: stays armed until something has run
    time.sleep(0.05)
    assert sent == [] and m.notify_armed_at is not None
    a, b = Job("j1", "p", "preview_local", [], {}), Job("j2", "q", "animate", ["py", "project.py", "animate", "--target", "cloud"], {})
    a.status, b.status = "running", "queued"
    m.jobs.update({"j1": a, "j2": b})
    a.status, a.ended = "done", time.time()
    m._maybe_notify()
    time.sleep(0.05)
    assert sent == [] and m.notify_armed_at is not None  # j2 still queued
    b.status, b.ended = "failed", time.time()
    m._maybe_notify()
    wait(1)
    assert sent[0] == "Queue empty — 1 done, 1 failed\n✗ j2 animate" and m.notify_armed_at is not None  # job id, never the project name
    m._maybe_notify()
    time.sleep(0.05)
    assert len(sent) == 1  # nothing new ran: one push per emptying
    c = Job("j3", "p", "preview_local", [], {})
    c.status = "cancelled"  # cancelled while queued: `ended` is set later, by its worker
    m.jobs["j3"] = c
    m._maybe_notify()
    wait(2)
    assert sent[1] == "Queue empty — 1 cancelled"
    c.ended = time.time()
    m._maybe_notify()
    time.sleep(0.05)
    assert len(sent) == 2  # the same job doesn't push again
    m.arm_notify(False)
    d = Job("j4", "p", "preview_local", [], {})
    d.status, d.ended = "done", time.time()
    m.jobs["j4"] = d
    m._maybe_notify()
    time.sleep(0.05)
    assert len(sent) == 2  # turned off


def test_notify_needs_pushover_config(monkeypatch):
    c = client()
    assert c.get("/api/notify").json() == {"armed": False, "configured": True}
    monkeypatch.setattr(config, "PUSHOVER_USER", "")
    assert c.post("/api/notify", json={"armed": True}).status_code == 400


def test_app_settings_pushover_keys(monkeypatch):
    c = client()
    monkeypatch.setattr(config, "PUSHOVER_USER", "")
    monkeypatch.setattr(config, "PUSHOVER_TOKEN", "")
    assert c.get("/api/notify").json()["configured"] is False
    r = c.put("/api/app-settings", json={"pushover_user": "u123", "pushover_token": "secret"}).json()
    assert r == {"pushover_user": "u123", "pushover_token_set": True, "pushover_from": "settings",
                 "animate_target": "cloud", "animate_batch_size": 4}  # the token is never sent back
    assert config.pushover() == ("u123", "secret") and c.get("/api/notify").json()["configured"] is True
    c.put("/api/app-settings", json={"pushover_user": "u456"})  # an absent key keeps its saved value
    assert config.pushover() == ("u456", "secret")
    c.put("/api/app-settings", json={"pushover_user": "", "pushover_token": ""})
    assert config.pushover() == ("", "") and c.get("/api/app-settings").json()["pushover_from"] is None


def test_comfy_watch_messages():
    from comfy_watch import change
    up, up2 = ("active", "100", "success"), ("active", "200", "success")
    assert change(up, up) is None and change(None, up) is None
    assert change(up, ("failed", "0", "oom-kill")) == "Local ComfyUI is down: failed (oom-kill)"
    assert change(("activating", "0", "oom-kill"), up2) == "Local ComfyUI is back up"
    assert change(up, up2) == "Local ComfyUI restarted"


def _batch_project(name="20261010_batch"):
    import json
    d = config.PIPELINES_DIR / name
    (d / "hires").mkdir(parents=True, exist_ok=True)
    (d / "project.json").write_text(json.dumps({"engine": "full", "stages": [{"id": i, "motion": "test motion"} for i in (1, 2, 3)]}))
    return d


def test_animate_batch_accumulates_dedups_and_fires_at_size(monkeypatch):
    import pytest
    from jobs import Manager
    from projects import ApiError
    _batch_project()
    monkeypatch.setattr(config, "animate_batch_size", lambda: 2)
    m = Manager()
    m._run = lambda job: None  # don't start project.py
    r1 = m.add_to_batch("20261010_batch", "animate", {"stages": [1]})
    assert r1["job"] is None and r1["pending"] == 1 and not m.jobs
    again = m.add_to_batch("20261010_batch", "animate", {"stages": [1]})  # a repeat click doesn't double-queue
    assert again["pending"] == 1 and not m.jobs
    r2 = m.add_to_batch("20261010_batch", "animate", {"stages": [2]})
    assert r2["job"] is not None and len(m.jobs) == 1
    job = next(iter(m.jobs.values()))
    assert job.req["stages"] == [1, 2] and m.batch_state() == []
    with pytest.raises(ApiError):  # only animate actions batch
        m.add_to_batch("20261010_batch", "preview", {"stages": [1]})
    with pytest.raises(ApiError):
        m.add_to_batch("20261010_batch", "animate", {"stages": []})


def test_animate_batch_keys_flush_and_clear(monkeypatch):
    from jobs import Manager
    _batch_project()
    monkeypatch.setattr(config, "animate_batch_size", lambda: 5)
    m = Manager()
    m._run = lambda job: None
    m.add_to_batch("20261010_batch", "animate", {"stages": [1], "folder": "hires"})
    m.add_to_batch("20261010_batch", "animate", {"stages": [2], "folder": ""})
    m.add_to_batch("20261010_batch", "animate_rife", {"stages": [3], "folder": "hires"})
    state = {(b["action"], b["folder"]): b["stages"] for b in m.batch_state()}
    assert state == {("animate", "hires"): [1], ("animate", ""): [2], ("animate_rife", "hires"): [3]}
    r = m.flush_batch()  # "Send now": everything fires even below the size
    assert len(r["sent"]) == 3 and r["pending"] == [] and len(m.jobs) == 3
    assert {(j.action, j.req.get("folder") or "") for j in m.jobs.values()} == {("animate", "hires"), ("animate", ""), ("animate_rife", "hires")}
    m.add_to_batch("20261010_batch", "animate", {"stages": [1]})
    m.clear_batch()
    assert m.batch_state() == []


def test_animate_batch_spans_projects(monkeypatch):
    import pytest
    from jobs import Manager
    from projects import ApiError
    _batch_project("20261010_batch_a")
    _batch_project("20261010_batch_b")
    monkeypatch.setattr(config, "animate_batch_size", lambda: 3)
    m = Manager()
    m._run = lambda job: None
    assert m.add_to_batch("20261010_batch_a", "animate", {"stages": [1, 2]})["pending"] == 2
    assert {b["project"] for b in m.batch_state()} == {"20261010_batch_a"}
    r = m.add_to_batch("20261010_batch_b", "animate", {"stages": [1]})  # the total reaches the size: both projects fire
    assert r["job"]["project"] == "20261010_batch_b" and r["pending"] == 0 and m.batch_state() == []
    assert {j.project: j.req["stages"] for j in m.jobs.values()} == {"20261010_batch_a": [1, 2], "20261010_batch_b": [1]}
    m.add_to_batch("gone_project", "animate", {"stages": [1]})
    m.add_to_batch("20261010_batch_a", "animate", {"stages": [3]})
    with pytest.raises(ApiError):  # a batch that can't be submitted stays queued, the others still go out
        m.flush_batch()
    assert [b["project"] for b in m.batch_state()] == ["gone_project"]
    assert any(j.req["stages"] == [3] for j in m.jobs.values())


def test_animate_batch_endpoints(monkeypatch):
    import app as appmod
    _batch_project("20261010_batchhttp")
    monkeypatch.setattr(config, "animate_target", lambda: "runpod")
    monkeypatch.setattr(config, "animate_batch_size", lambda: 2)
    monkeypatch.setattr(appmod.manager, "_run", lambda job: None)
    c = client()
    r = c.post("/api/projects/20261010_batchhttp/animate-batch", json={"action": "animate", "stages": [1]}).json()
    assert r["job"] is None and r["pending"] == 1
    st = c.get("/api/animate-batch").json()
    assert st["size"] == 2 and st["pending"][0]["stages"] == [1]
    r = c.post("/api/projects/20261010_batchhttp/animate-batch", json={"action": "animate", "stages": [2]}).json()
    job = r["job"]
    assert job and job["lane"] == "runpod" and job["req"]["stages"] == [1, 2] and "--target runpod" in job["argv"]
    assert c.get("/api/animate-batch").json()["pending"] == []
    c.post("/api/projects/20261010_batchhttp/animate-batch", json={"action": "animate", "stages": [3]})  # below the size
    assert c.post("/api/animate-batch/flush").json()["sent"]
    c.post("/api/projects/20261010_batchhttp/animate-batch", json={"action": "animate", "stages": [1]})
    assert c.delete("/api/animate-batch").json() == {"cleared": True}
    assert c.get("/api/animate-batch").json()["pending"] == []
    assert c.post("/api/projects/20261010_batchhttp/animate-batch", json={"action": "preview", "stages": [1]}).status_code == 400


def test_animate_refused_without_motion(monkeypatch):
    """Animate is the paid step: a submit naming a stage with no motion prompt is refused up front, before any job exists."""
    import json
    import pytest
    from jobs import Manager
    from projects import ApiError
    d = _batch_project()
    cfg = json.loads((d / "project.json").read_text())
    cfg["stages"][1]["motion"] = ""
    (d / "project.json").write_text(json.dumps(cfg))
    m = Manager()
    with pytest.raises(ApiError, match=r"stages \[2\].*motion"):
        m.submit("20261010_batch", "animate", {"stages": [1, 2]})
    assert not m.jobs
