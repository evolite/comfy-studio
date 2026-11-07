"""The voice library, `voices.json` (data/library, else the defaults/ starter; config.library_file): named voices a stage's `motion` takes its voice text from, so one voice
reads the same in every stage and swapping it is one field. Read fresh on every call, so a web UI edit applies to the next job.

    {"voices": {name: {"label": str, "tags": [str], "voice": str, "delivery": str, "moans": str, "sound": str,
                       "note": str, "tips": str, "sample": {"project": str, "stage": int, "folder": str}}}}

Slots (prompt fragments, written to read in place):
  `voice` (required)  the voice itself:            "Her voice (S1) is {voice}."
  `delivery`          the verb + how a line is said: "... and {voice.delivery}, <d>[English] ...</d>"  e.g. "purrs (S) in her heavy French accent, slow and teasing"
  `moans`             the verb + her moans:          "She {voice.moans}, at the same pace ..."           e.g. "moans (S) in soft, purring French sighs ..."
  `sound`             the moans in overall_soundscape: "... and her {voice.sound} in time with the motion"
`(S)` in a slot is the speaker tag right after the verb (H3 wants it there): filled with (S1), or the speaker a placeholder names
({voice.delivery:S2}). `sample` = a clip of the voice to listen to
(<project>/<folder>/videos/stage_<stage>.mp4; folder "" = the project root).

A motion uses the slots as placeholders: {voice}, {voice.delivery}, {voice.moans}, {voice.sound} (optionally :Sn). The voice is the stage's
`voice`, else the project's top-level `voice`; a second speaker gets its own with `voices` ({"S2": name}, stage or project level, the
stage's wins), which fills that speaker's :S2 slots (a speaker without an entry uses the S1 voice). expand() fills them in when the clip is animated (project.py writes the expanded
text to _debug/motion_prompts.json); a motion without placeholders is left as it is.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: config.py
import config  # noqa: E402

NAME = re.compile(r"^[a-z0-9_]+$")
SLOTS = ("voice", "delivery", "moans", "sound")
TEXT = ("label", "note", "tips") + SLOTS
FIELDS = set(TEXT) | {"tags", "sample", "disabled"}  # disabled: not offered in the pickers; stages already using it keep it
PLACEHOLDER = re.compile(r"\{voice(?:\.(delivery|moans|sound))?(?::(S\d+))?\}")
SPEAKER = "(S)"  # in a slot: the speaker tag, filled per placeholder


def load() -> dict:
    f = config.VOICES_FILE if config.VOICES_FILE.is_file() else config.DEFAULTS_DIR / config.VOICES_FILE.name
    data = json.loads(f.read_text()) if f.exists() else {}
    return {"voices": data.get("voices", {})}


def _validate_sample(label: str, sample) -> None:
    if sample is None:
        return
    if not isinstance(sample, dict) or set(sample) - {"project", "stage", "folder"} or not isinstance(sample.get("project"), str) \
            or isinstance(sample.get("stage"), bool) or not isinstance(sample.get("stage"), int) or not isinstance(sample.get("folder", ""), str):
        raise ValueError(f"{label}: sample must be {{\"project\": str, \"stage\": int, \"folder\": str}}")


def validate_entry(name, spec) -> None:
    label = f"voice {name!r}"
    if not isinstance(name, str) or not NAME.match(name):
        raise ValueError(f"{label}: names are lowercase letters, digits and _")
    if not isinstance(spec, dict):
        raise ValueError(f"{label}: must be an object")
    extra = set(spec) - FIELDS
    if extra:
        raise ValueError(f"{label}: unknown field(s) {', '.join(sorted(extra))}")
    if not isinstance(spec.get("voice"), str) or not spec["voice"].strip():
        raise ValueError(f"{label}: needs a `voice` description")
    for k in TEXT:
        if not isinstance(spec.get(k, ""), str):
            raise ValueError(f"{label}: {k} must be text")
    if not isinstance(spec.get("disabled", False), bool):
        raise ValueError(f"{label}: disabled must be true or false")
    if not isinstance(spec.get("tags", []), list) or not all(isinstance(t, str) for t in spec.get("tags", [])):
        raise ValueError(f"{label}: tags must be a list of words")
    _validate_sample(label, spec.get("sample"))


def validate(data: dict) -> None:
    """Raises ValueError naming the first bad entry."""
    for name, spec in data["voices"].items():
        validate_entry(name, spec)


def save(data: dict) -> None:
    validate(data)
    config.save_library(config.VOICES_FILE, data)


def voice_of(stage: dict, cfg: dict):
    """The voice a stage uses: its own `voice`, else the project's."""
    return stage.get("voice") or cfg.get("voice") or None


SPEAKER_KEY = re.compile(r"^S([2-9]|[1-9]\d+)$")


def cast_of(stage: dict, cfg: dict) -> dict:
    """{speaker: voice name} for a stage: S1 = voice_of(); other speakers from `voices` ({"S2": name}), the stage's entry winning
    over the project's. A {voice...:Sn} slot whose speaker has no entry uses S1's voice."""
    cast = {k: v for k, v in {**(cfg.get("voices") or {}), **(stage.get("voices") or {})}.items() if v}
    if name := voice_of(stage, cfg):
        cast["S1"] = name
    return cast


def check_cast(value, label: str, voices: dict = None) -> None:
    """Raises ValueError unless `value` is a valid `voices` field ({"S2": name, ...}; S1 is `voice`)."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f'{label}: voices must be an object like {{"S2": "<voice>"}}')
    voices = load()["voices"] if voices is None else voices
    for sp, name in value.items():
        if not isinstance(sp, str) or not SPEAKER_KEY.match(sp):
            raise ValueError(f"{label}: voices keys are speakers S2, S3, ... (S1 is `voice`), not {sp!r}")
        if name and name not in voices:
            raise ValueError(f"{label}: no voice {name!r} in the voice library")


def _cast(cast) -> dict:
    return {"S1": cast} if isinstance(cast, str) else {k: v for k, v in (cast or {}).items() if v}


def _pick(cast: dict, speaker):
    return cast.get(speaker or "S1") or cast.get("S1")


def problems(motion: str, cast, voices: dict = None) -> list:
    """Why expand() would refuse this motion with this cast ([] = it expands). `cast` = a voice name (every speaker) or
    {speaker: name} (cast_of())."""
    cast = _cast(cast)
    voices = load()["voices"] if voices is None else voices
    unknown = [f"no voice {n!r} in the voice library" for n in dict.fromkeys(cast.values()) if n not in voices]
    if unknown:
        return unknown
    used = [(m.group(1) or "voice", m.group(2)) for m in PLACEHOLDER.finditer(motion or "")]
    unset = sorted({s for s, sp in used if not _pick(cast, sp)}, key=SLOTS.index)
    if unset:
        return [f"the motion uses {', '.join('{voice}' if s == 'voice' else '{voice.' + s + '}' for s in unset)} but no voice is set "
                f"(pick one for the stage or the project)"]
    gaps = {(_pick(cast, sp), s) for s, sp in used if not (voices[_pick(cast, sp)].get(s) or "").strip()}
    return [f"voice {n!r} has no `{s}` for {{voice.{s}}}" for n, s in sorted(gaps, key=lambda g: (g[0], SLOTS.index(g[1])))]


def expand(motion: str, cast, label: str = "", voices: dict = None) -> str:
    """The motion with its {voice...} placeholders filled from the cast (a voice name, or {speaker: name}: a :Sn slot takes
    that speaker's voice, else S1's); exits (before anything is spent) when one can't be filled."""
    voices = load()["voices"] if voices is None else voices
    bad = problems(motion, cast, voices)
    if bad:
        sys.exit(f"{label + ': ' if label else ''}{'; '.join(bad)}")
    cast = _cast(cast)
    return PLACEHOLDER.sub(lambda m: voices[_pick(cast, m.group(2))][m.group(1) or "voice"].strip().replace(SPEAKER, f"({m.group(2) or 'S1'})"),
                           motion)


def usage() -> dict:
    """{name: [{"project", "stage"}]}: which stages use each voice (a project-level voice counts for each stage without its own)."""
    out = {}
    for pj in sorted(config.PIPELINES_DIR.glob("*/project.json")):
        try:
            cfg = json.loads(pj.read_text())
        except (OSError, ValueError):
            continue
        for pos, s in enumerate(cfg.get("stages", []), 1):
            for name in dict.fromkeys(cast_of(s, cfg).values()):
                out.setdefault(name, []).append({"project": pj.parent.name, "stage": s.get("id", pos)})
    return out
