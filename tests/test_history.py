import history


def test_archive_list_and_restore_swap_versions(tmp_path):
    d = tmp_path / "proj"
    (d / "hires" / "videos").mkdir(parents=True)
    (d / "hires" / "stage_2.png").write_bytes(b"v1")
    history.archive(d, d / "hires" / "stage_2.png")
    (d / "hires" / "stage_2.png").write_bytes(b"v2")
    (d / "hires" / "videos" / "stage_2.mp4").write_bytes(b"clip")
    history.archive(d, d / "hires" / "videos" / "stage_2.mp4")
    assert history.archive(d, d / "hires" / "stage_9.png") is None  # nothing to archive
    rows = history.listing(d, 2)
    assert sorted((r["folder"], r["kind"]) for r in rows) == [("hires", "clip"), ("hires", "still")]
    still = next(r for r in rows if r["kind"] == "still")
    assert history.restore(d, 2, still["file"]) == "hires/stage_2.png"
    assert (d / "hires" / "stage_2.png").read_bytes() == b"v1"
    back = [r for r in history.listing(d, 2) if r["kind"] == "still"]
    assert [(d / "_debug" / r["file"]).read_bytes() for r in back] == [b"v2"]  # the replaced one is restorable in turn
    clip = next(r for r in history.listing(d, 2) if r["kind"] == "clip")
    history.restore(d, 2, clip["file"])
    assert (d / "hires" / "videos" / "stage_2.mp4").read_bytes() == b"clip"


def test_older_archives_are_listed_by_folder(tmp_path):
    d = tmp_path / "proj"
    for sub, name in [("prev_videos", "proj_stage_3_ab12cd.mp4"), ("prev_videos", "interp_stage_3_ab12cd.mp4"),
                      ("removed", "finish_1700000000_stage_3.png"), ("removed", "take_stage_3_20260101-000000-000.png"),
                      ("prev_videos", "proj_stage_33_ab12cd.mp4")]:
        (d / "_debug" / sub).mkdir(parents=True, exist_ok=True)
        (d / "_debug" / sub / name).write_bytes(b"x")
    got = sorted((r["folder"], r["kind"]) for r in history.listing(d, 3))
    assert got == [("", "clip"), ("finish", "still"), ("interp", "clip")]
    history.restore(d, 3, "prev_videos/interp_stage_3_ab12cd.mp4")
    assert (d / "interp" / "videos" / "stage_3.mp4").is_file()


def test_cleanup_keeps_current_results_and_takes_only(tmp_path):
    import json
    d = tmp_path / "proj"
    (d / "hires").mkdir(parents=True)
    (d / "hires" / "stage_1.png").write_bytes(b"old")
    history.archive(d, d / "hires" / "stage_1.png")
    (d / "hires" / "stage_1.png").write_bytes(b"cur")
    sd = d / "preview" / "takes" / "stage_1"
    sd.mkdir(parents=True)
    for tid in ("20260101-000000-000", "20260101-000001-000"):
        (sd / f"{tid}.png").write_bytes(b"t")
        (sd / f"{tid}.json").write_text("{}")
    (sd / "current").write_text(json.dumps({"id": "20260101-000001-000"}))
    (d / "interp" / "_debug" / "edit_build_x").mkdir(parents=True)
    (d / "interp" / "_debug" / "edit_build_x" / "cut.mp4").write_bytes(b"c")
    (d / "interp" / "edits").mkdir()
    (d / "interp" / "edits" / "e.mp4").write_bytes(b"e")
    assert history.cleanup(d, dry_run=True) == {"files": 4, "bytes": 3 + 1 + 2 + 1}
    assert history.listing(d, 1)  # a dry run deletes nothing
    history.cleanup(d)
    assert history.listing(d, 1) == []
    assert (d / "hires" / "stage_1.png").read_bytes() == b"cur" and (d / "interp" / "edits" / "e.mp4").is_file()
    assert sorted(f.name for f in sd.iterdir()) == ["20260101-000001-000.json", "20260101-000001-000.png", "current"]
