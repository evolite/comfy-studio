"""Preview takes: every preview render of a stage is kept, with a record of what made it, and one of them is "current".

    preview/takes/stage_N/<id>.png          id = YYYYMMDD-HHMMSS-<ms> (sorts by time)
    preview/takes/stage_N/<id>.json         record: seed, prompt, LoRAs, sampler, unet, settings (diffable)
    preview/takes/stage_N/<id>.graph.json   the exact graph (redo)
    preview/takes/stage_N/current           {"id", "size", "mtime_ns"} of the take copied to preview/stage_N.png

preview/stage_N.png stays the one file every later step reads; it is a copy of the current take. Every new render
becomes the current take as it lands (the previous current stays in the takes folder for comparison) and ✓s the
preview; if that displaced a different take, everything made from the old one loses its ✓. If the file no longer
matches `current` (restored, imported) it is taken in as an `external` take; an untracked preview/stage_N.png (no takes
yet) as a `legacy` take (no record).
"""
import json
import re
import shutil
import time
from pathlib import Path

FOLDER = "preview"
PNG_GLOB = "*.png"
FINAL_PNG = "final.png"
PREVIEW_SETTINGS = ("krea_unet", "krea_vae", "krea_steps", "krea_cfg", "krea_sampler", "krea_scheduler", "aspect_ratio", "megapixels")


def stage_dir(d: Path, n) -> Path:
    return d / FOLDER / "takes" / f"stage_{n}"


def canonical(d: Path, n) -> Path:
    return d / FOLDER / f"stage_{n}.png"


def new_id(t: float = None, taken=()) -> str:
    """YYYYMMDD-HHMMSS-mmm (milliseconds), so ids sort in render order; bumped past any id in `taken`."""
    t = t or time.time()
    while True:
        tid = time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + f"-{int(t * 1000) % 1000:03d}"
        if tid not in taken:
            return tid
        t += 0.001


def _stamp(p: Path) -> dict:
    st = p.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def current_id(d: Path, n):
    try:
        return json.loads((stage_dir(d, n) / "current").read_text()).get("id")
    except (OSError, ValueError):
        return None


def record(graph: dict, stage: dict, common: dict, target: str | None, loras: dict) -> dict:
    """What made a preview, read from its prepared graph (Krea2 KSampler 30:3, UNet 30:10, size 49, prompt 30:6) plus the
    stage's resolved image LoRAs and the effective preview settings."""
    node = lambda nid: graph.get(nid, {}).get("inputs", {})  # noqa: E731
    ks = node("30:3")
    settings = {k: v for k, v in common.items() if k in PREVIEW_SETTINGS}
    settings.update({k: v for k, v in stage.items() if k in PREVIEW_SETTINGS})
    return {"target": target, "seed": ks.get("seed"), "unet": node("30:10").get("unet_name"),
            "sampler": ks.get("sampler_name"), "scheduler": ks.get("scheduler"), "steps": ks.get("steps"), "cfg": ks.get("cfg"),
            "aspect_ratio": node("49").get("aspect_ratio"), "megapixels": node("49").get("megapixels"),
            "prompt": node("30:6").get("text", ""), "krea_scene": stage.get("krea_scene", ""),
            "loras": {k: v for k, v in (loras or {}).items() if v}, "settings": settings}


def add(d: Path, n, png: Path, graph: dict = None, rec: dict | None = None, make_current: bool = False, source: str = "run",
        move: bool = False, t: float = None) -> str:
    """Store a render as a take of stage n; with make_current it also becomes preview/stage_N.png."""
    sd = stage_dir(d, n)
    sd.mkdir(parents=True, exist_ok=True)
    tid = new_id(t, {p.stem for p in sd.glob(PNG_GLOB)})
    (shutil.move if move else shutil.copyfile)(png, sd / f"{tid}.png")
    if rec is not None:
        (sd / f"{tid}.json").write_text(json.dumps({"id": tid, "created": time.time(), "source": source, **rec}, indent=2))
    if graph is not None:
        (sd / f"{tid}.graph.json").write_text(json.dumps(graph, indent=2))
    if make_current:
        set_current(d, n, tid)
    return tid


def set_current(d: Path, n, tid: str) -> bool:
    """Copy take `tid` onto preview/stage_N.png; returns whether the current take changed."""
    sd = stage_dir(d, n)
    src = sd / f"{tid}.png"
    if not src.is_file():
        raise FileNotFoundError(f"no take {tid} for stage {n}")
    changed = current_id(d, n) != tid
    dst = canonical(d, n)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    (sd / "current").write_text(json.dumps({"id": tid, **_stamp(tmp)}))  # pointer first: a sync in between sees an older canonical
    tmp.replace(dst)                                                     # file than the pointer and leaves it alone (see sync)
    return changed


def sync(d: Path, n) -> None:
    """Take in a canonical file that isn't the current take."""
    sd = stage_dir(d, n)
    sd.mkdir(parents=True, exist_ok=True)
    canon = canonical(d, n)
    if canon.is_file():
        _sync_canonical(d, n, sd, canon)
    _backfill(d, n, sd)


def _backfill(d: Path, n, sd: Path) -> None:
    """A current take without a record (made before takes were kept) gets one from the last preview run's graph, if that run's
    final.png is this very image."""
    tid = current_id(d, n)
    dbg = d / FOLDER / "_debug" / f"stage_{n}"
    final = dbg / FINAL_PNG
    if not tid or (sd / f"{tid}.json").exists() or not (dbg / "graph.json").is_file() or not final.is_file():
        return  # a record (or a stub: external / already tried) exists, or there is no run to read
    png = sd / f"{tid}.png"
    if not png.is_file() or png.read_bytes() != final.read_bytes():
        (sd / f"{tid}.json").write_text(json.dumps({"id": tid, "source": "legacy"}))  # not that run's image: don't compare again
        return
    graph = json.loads((dbg / "graph.json").read_text())
    loras = {re.sub(r"^Load LoRA \((.*)\)$", r"\1", v.get("_meta", {}).get("title", "")): v["inputs"].get("strength_model")
             for v in graph.values() if v.get("class_type") == "LoraLoaderModelOnly"}
    rec = record(graph, {}, {}, None, loras)
    (sd / f"{tid}.json").write_text(json.dumps({"id": tid, "created": final.stat().st_mtime, "source": "legacy",
                                                **rec}, indent=2))


def _sync_canonical(d: Path, n, sd: Path, canon: Path) -> None:
    try:
        cur = json.loads((sd / "current").read_text())
    except (OSError, ValueError):
        cur = {}
    st = canon.stat()
    if cur.get("id") and (sd / f"{cur['id']}.png").is_file() and cur.get("size") == st.st_size and cur.get("mtime_ns") == st.st_mtime_ns:
        return
    if cur.get("id") and (sd / f"{cur['id']}.png").is_file() and st.st_mtime_ns <= (sd / "current").stat().st_mtime_ns:
        return  # set_current is between writing the pointer and moving the file into place
    source = "external" if cur else "legacy"
    tid = add(d, n, canon, rec=None, source=source, t=canon.stat().st_mtime)
    if source == "external":
        (sd / f"{tid}.json").write_text(json.dumps({"id": tid, "created": time.time(), "source": "external"}))
    (sd / "current").write_text(json.dumps({"id": tid, **_stamp(canon)}))


def load_record(d: Path, n, tid: str):
    try:
        rec = json.loads((stage_dir(d, n) / f"{tid}.json").read_text())
    except (OSError, ValueError):
        return None
    return rec if "prompt" in rec else None  # external takes have only a stub


def listing(d: Path, n) -> list:
    """[{id, mtime, current, source, delta}] newest first (call sync first)."""
    sd = stage_dir(d, n)
    if not sd.is_dir():
        return []
    cur = current_id(d, n)
    cur_rec = load_record(d, n, cur) if cur else None
    out = []
    for p in sorted(sd.glob(PNG_GLOB), reverse=True):
        tid = p.stem
        rec = load_record(d, n, tid)
        stub = {}
        try:
            stub = json.loads((sd / f"{tid}.json").read_text())
        except (OSError, ValueError):
            pass
        out.append({"id": tid, "mtime": int(p.stat().st_mtime), "current": tid == cur,
                    "source": (rec or stub).get("source", "legacy"),
                    "delta": "" if tid == cur else delta(rec, cur_rec)})
    return out


def approve(d: Path, n, changed: bool) -> None:
    """A new take just became current: ✓ the preview in project.json. If it displaced a different take, the ✓s of
    everything made from the old one (IL, hires, finish…) no longer apply — done collapses back to the preview."""
    p = d / "project.json"
    cfg = json.loads(p.read_text())
    for pos, s in enumerate(cfg.get("stages", []), 1):
        if s.get("id", pos) == n:
            done = s.get("done") if isinstance(s.get("done"), list) else []
            s["done"] = ["preview"] if changed else sorted(set(done) | {"preview"})
            p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
            return


def remove_others(d: Path, n) -> int:
    """Move every take except the current one to _debug/removed/ (never deleted)."""
    sd = stage_dir(d, n)
    cur = current_id(d, n)
    dst = d / "_debug" / "removed"
    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in sorted(sd.glob(PNG_GLOB)):
        if p.stem == cur:
            continue
        for f in sd.glob(f"{p.stem}*"):
            shutil.move(f, dst / f"take_stage_{n}_{f.name}")
        moved += 1
    return moved


# --- what changed --------------------------------------------------------------------------------------------------
FIELDS = (("seed", "seed"), ("unet", "UNet"), ("sampler", "sampler"), ("scheduler", "scheduler"), ("steps", "steps"),
          ("cfg", "cfg"), ("aspect_ratio", "aspect"), ("megapixels", "megapixels"), ("target", "ran on"))


def diff(rec, cur) -> dict:
    """Field-by-field differences of take `rec` against the current take `cur`, plus a word diff of the prompt.
    {"fields": [{key, label, this, current}], "loras": [{name, this, current}], "prompt": [[op, text]...] | None}"""
    if not rec or not cur:
        return {"unknown": True}
    fields = [{"key": k, "label": lab, "this": rec.get(k), "current": cur.get(k)} for k, lab in FIELDS
              if rec.get(k) != cur.get(k) and rec.get(k) is not None and cur.get(k) is not None]  # skips fields a backfilled record lacks
    a, b = rec.get("loras") or {}, cur.get("loras") or {}
    loras = [{"name": k, "this": a.get(k), "current": b.get(k)} for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    sa, sb = rec.get("settings") or {}, cur.get("settings") or {}
    settings = [{"key": k, "this": sa.get(k), "current": sb.get(k)} for k in sorted(set(sa) | set(sb))
                if sa.get(k) != sb.get(k) and k not in PREVIEW_SETTINGS]  # those show as the graph fields above
    prompt = None
    if rec.get("prompt") != cur.get("prompt"):
        prompt = word_diff(cur.get("prompt", ""), rec.get("prompt", ""))
    return {"fields": fields, "loras": loras, "settings": settings, "prompt": prompt}


def word_diff(old: str, new: str) -> list:
    """[[op, text]] with op "=", "+" (only in new), "-" (only in old); words keep their trailing whitespace."""
    import difflib
    tok = lambda s: re.findall(r"\S+\s*", s)  # noqa: E731
    a, b = tok(old), tok(new)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(["=", "".join(a[i1:i2])])
        else:
            if i2 > i1:
                out.append(["-", "".join(a[i1:i2])])
            if j2 > j1:
                out.append(["+", "".join(b[j1:j2])])
    return out


def _short(k, v) -> str:
    if k == "unet":
        return Path(str(v)).stem.split("_")[0]
    if k == "aspect_ratio":
        return str(v).split(" ")[0]
    return _num(v)


def _lora_delta(x) -> str:
    if x["this"] is None:
        return f"-{x['name']}"
    if x["current"] is None:
        return f"+{x['name']} {_num(x['this'])}"
    return f"{x['name']} {_num(x['current'])}→{_num(x['this'])}"


def delta(rec, cur) -> str:
    """One-line caption of what differs from current: "seed", "prompt · +1 LoRA · cfg", "" when unknown."""
    if not rec or not cur:
        return ""
    dd = diff(rec, cur)
    parts = [f"{f['label']} {_short(f['key'], f['this'])}" for f in dd["fields"] if f["key"] != "seed"]
    for x in dd["loras"]:
        parts.append(_lora_delta(x))
    parts += [f"{s['key']} {_num(s['this'])}" for s in dd["settings"]]
    if dd["prompt"]:
        parts.append("prompt edited")
    if not parts:
        return "seed only" if any(f["key"] == "seed" for f in dd["fields"]) else "same"
    return " · ".join(parts)


def _num(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)
