"""The voice library (voices.json): slot expansion, its refusals, usage, and the API (library, edits, the 400/409s)."""
import json
import shutil

import pytest
from fastapi.testclient import TestClient

import config
import voice_registry as vr

VOICES = {"fr": {"label": "French", "voice": "a low French voice", "delivery": "purrs (S) slowly", "moans": "sighs (S) softly",
                 "sound": "soft French sighs"},
          "bare": {"voice": "a plain voice"}}


@pytest.fixture
def lib(tmp_path, monkeypatch):
    f = tmp_path / "voices.json"
    f.write_text(json.dumps({"voices": VOICES}))
    monkeypatch.setattr(config, "VOICES_FILE", f)
    return f


def test_real_library_is_valid():
    vr.validate(vr.load())


def test_expand_fills_every_slot_and_the_speaker_tag(lib):
    motion = "Her voice (S1) is {voice}. She {voice.delivery}, <d>hi</d> She {voice.moans}. Sound: her {voice.sound}. {voice.delivery:S2}"
    out = vr.expand(motion, "fr")
    assert out == ("Her voice (S1) is a low French voice. She purrs (S1) slowly, <d>hi</d> She sighs (S1) softly. "
                   "Sound: her soft French sighs. purrs (S2) slowly")


def test_expand_leaves_a_plain_motion_alone(lib):
    assert vr.expand("no slots here", None) == "no slots here"
    assert vr.expand("no slots here", "fr") == "no slots here"


@pytest.mark.parametrize(("motion", "name", "msg"), [
    ("{voice}", None, "no voice is set"),
    ("{voice}", "nope", "no voice 'nope'"),
    ("{voice.moans}", "bare", "has no `moans`"),
])
def test_expand_refuses_what_it_cant_fill(lib, motion, name, msg):
    with pytest.raises(SystemExit, match=msg):
        vr.expand(motion, name, "stage 3")


def test_stage_voice_wins_over_the_project():
    assert vr.voice_of({"voice": "a"}, {"voice": "b"}) == "a"
    assert vr.voice_of({}, {"voice": "b"}) == "b"
    assert vr.voice_of({}, {}) is None


@pytest.mark.parametrize("spec", [{"label": "no voice"}, {"voice": "x", "colour": "red"}, {"voice": "x", "tags": "a"}, {"voice": "x", "disabled": "yes"},
                                  {"voice": "x", "sample": {"project": "p"}}])
def test_validate_rejects_bad_entries(spec):
    with pytest.raises(ValueError):
        vr.validate({"voices": {"v": spec}})


def _project(name, stages, **top):
    d = config.PIPELINES_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "project.json").write_text(json.dumps({"engine": "full", "stages": stages, **top}))
    return d


def test_usage_counts_stage_and_project_voices(lib):
    _project("20261102_voices_a", [{"id": 1, "motion": "m", "voice": "fr"}, {"id": 2, "motion": "m"}], voice="bare")
    used = vr.usage()
    assert {"project": "20261102_voices_a", "stage": 1} in used["fr"]
    assert {"project": "20261102_voices_a", "stage": 2} in used["bare"]


def test_api_library_edit_and_refusals(lib):
    from app import app
    c = TestClient(app)
    _project("20261102_voices_b", [{"id": 1, "motion": "Her voice (S1) is {voice}.", "voice": "fr"}])
    got = c.get("/api/voices").json()
    assert set(got["voices"]) == {"fr", "bare"} and got["slots"] == list(vr.SLOTS)
    assert c.put("/api/voices/new_one", json={"voice": "a new voice", "tags": ["x"]}).status_code == 200
    assert c.put("/api/voices/Bad-Name", json={"voice": "x"}).status_code == 400
    assert c.delete("/api/voices/fr").status_code == 409  # in use
    assert c.delete("/api/voices/new_one").status_code == 200
    assert c.patch("/api/projects/20261102_voices_b/stages/1", json={"voice": "nope"}).status_code == 400
    assert c.put("/api/projects/20261102_voices_b/voice", json={"voice": "bare"}).status_code == 200
    assert json.loads((config.PIPELINES_DIR / "20261102_voices_b" / "project.json").read_text())["voice"] == "bare"


def test_animate_submit_refuses_an_unfilled_slot(lib):
    from jobs import Manager
    from projects import ApiError
    cfg = {"stages": [{"id": 1, "motion": "x {voice}"}, {"id": 2, "motion": "x {voice}", "voice": "fr"}]}
    with pytest.raises(ApiError, match="no voice is set"):
        Manager._check_motion(cfg, [1])
    Manager._check_motion(cfg, [2])


def test_real_library_copy_round_trips(tmp_path, monkeypatch):
    """Every starter entry fills a full four-slot template without errors."""
    f = tmp_path / "voices.json"
    shutil.copy(config.DEFAULTS_DIR / "voices.json", f)
    monkeypatch.setattr(config, "VOICES_FILE", f)
    for name, spec in vr.load()["voices"].items():
        slots = " ".join("{voice}" if k == "voice" else "{voice.%s}" % k for k in vr.SLOTS if spec.get(k))
        assert "{voice" not in vr.expand(slots, name)


def test_make_sample_creates_one_stage_per_voice_from_the_first(lib, monkeypatch):
    import voices
    from projects import ApiError
    monkeypatch.setattr(config, "VOICE_SAMPLES_PROJECT", "20261102_voice_samples_t")
    d = _project("20261102_voice_samples_t", [{"id": 1, "krea_scene": "k", "motion": "Her voice (S1) is {voice}.", "duration": 7, "voice": "fr"}])
    with pytest.raises(ApiError, match="no preview"):
        voices.make_sample("bare")
    (d / "preview").mkdir()
    (d / "preview" / "stage_1.png").write_bytes(b"png")
    assert voices.make_sample("fr")["stage"] == 1  # the first stage is its own voice's sample
    t = voices.make_sample("bare")
    again = voices.make_sample("bare")
    assert t == again == {"project": "20261102_voice_samples_t", "stage": 2, "folder": "preview"}
    s2 = json.loads((d / "project.json").read_text())["stages"][1]
    assert s2["voice"] == "bare" and s2["motion"] == "Her voice (S1) is {voice}." and s2["duration"] == 7 and s2["done"] == ["preview"]
    assert (d / "preview" / "stage_2.png").read_bytes() == b"png"
    assert vr.load()["voices"]["bare"]["sample"] == {"project": "20261102_voice_samples_t", "stage": 2, "folder": "interp"}


def test_disabled_voice_validates_and_still_expands(lib):
    data = vr.load()
    data["voices"]["fr"]["disabled"] = True
    vr.save(data)
    assert vr.expand("{voice}", "fr") == "a low French voice"  # off = not offered; stages using it keep working
