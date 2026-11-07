#!/usr/bin/env python3
"""
Beat-synced TikTok edit from a pipeline's i2v clips (pipeline_dir/videos/stage_N.mp4,
built via video/animate_pipeline.py -- see AGENTS.md), or its stage_N.png stills when
it has no clips (--stills forces stills).

Structure: a calm lead-in on stage 1 (grade + vignette + grain only), then the
punch-cut sequence starts on the first beat after it. Each --beats-per-cut beats long
(default 2). Every later stage's clip gets cut count scaled to its own length
(usable clip length / (beats_per_cut x avg beat spacing), capped at
MAX_SUBCUTS_PER_STAGE) so a longer i2v render actually gets shown instead of
mostly sitting unused -- a ~1s clip gets 1 cut, a ~2s clip gets ~2, etc. Stills
have no inherent duration, so they use the flat --subcuts-per-image count
instead. Each cut is a subtle zoom + Ken-Burns drift, cycling crops (wide ->
hips -> face -> chest) with alternating zoom in/out. With --zoom (1.5x if given
without a value), each stage's first cut shows the full image and the rest
punch in to that zoom on a body part (hips -> face -> chest, cycling across
stages) and STAY at that zoom while the pan keeps drifting, until the next
shot. Warm grade, vignette/grain, and a motion-blur + chromatic-aberration
burst on every cut. The final cut holds --last-weight x.

Defaults are the house track (config.DEFAULT_BEATS, e.g. data/audio/default.json + default.mp3; its start time is in its beat
file), so the normal call is just:

    python3 video/make_tiktok.py data/pipelines/<run> [--zoom] [--stills]

Another track: --beat-times-file audio/<Song>.json -- the file holds the beats and the
start time (where in the song the video starts), and audio/<Song>.mp3 beside it is used as the audio.
--start-time/--audio override. The music plays from the start time under the --lead-in-seconds hold
on stage 1; the cuts start on the first beat at or after start + lead-in.

--clip-audio layers the clips' own generated audio (dialogue, breathing, ambience) over the music: each clip's
sound plays once at its stage's start (trimmed like the picture, atempo'd with a retimed one-cut stage),
the music ducks under it (--duck, --music-volume, --clip-volume, --mute-clip-audio IDs). Video is unchanged.

--as-still IDs mixes the two: those stages show their still (Ken-Burns crops, --subcuts-per-image of them), every other stage
its clip; --stills-from DIR takes the stills from another folder (interp/ has clips only). The edit stays at 48 fps when every
clip is interpolated -- stills don't count.

Stages play in stage-number order (first = lead-in); pass --order to play a
different sequence (repeats and drops allowed) without renaming files --
renaming breaks the stage-number keys in motion_prompts.json/stage_durations.json.
No music (--no-music, "scenario" edit): no beats, no punch cuts. Every stage plays once in full (a clip at its own speed minus
its trim, a still for --still-seconds with a slow drift) and flows into the next through an ffmpeg xfade --transition
(fade, fadeblack, dissolve, ...; "cut" = hard cuts) lasting --transition-seconds. The soundtrack is the clips' own audio,
crossfaded with the picture (--mute-clip-audio drops a stage's); --fade-in / --fade-out fade picture and sound from/to black.

Polaroids (video/polaroid.py): --polaroid shows every still as an Instax-style card (film grade, paper frame, handwritten
caption) dropped onto a blurred, darkened copy of itself, one cut per still. --collage IDS ends the edit on those stages' cards
piling up (one landing every --collage-beats beats, impact on the beat, then a hold; a no-music edit lands one every
COLLAGE_EVERY s), the long final hold giving way to it (last cut x1); its stills come from --collage-from (default
--stills-from, else pipeline_dir). --card-sounds mixes one of the user's camera sounds (config.POLAROID_SFX_DIR, dealt out
shuffled, levels varied) in at every card impact. --card-size sets the card width in px of the 1080 frame (default: by card count).

Long-hold look: --subcuts-per-image 1 = exactly one cut per stage, on a fixed beat grid
(period/phase fitted to the detected beats from the drop on, so detector jitter doesn't
move cuts). Each cut is an even number of beats (2-8) needing the least retime of the clip's length, and
the clip is retimed (about 0.8-1.4x) to fill it exactly, so cuts land on half-bars and no
clip is cut off or frozen (the final hold is capped at what the clip can be slowed to fill).
"""
import argparse
import bisect
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo root: config.py
import config  # noqa: E402
import polaroid  # noqa: E402  (video/, beside this file)
FPS = 30  # the edit's frame rate; make_tiktok() raises it to HI_FPS when every clip is interpolated (edit_fps)
HI_FPS = 48  # RIFE x2 of the 24 fps i2v clips (video/interpolate.py): render at their own rate, no frame dropping
FADE_OUT = 0.5  # seconds the music fades out over at the end of the edit (--fade-out; 0 = hard stop)
TRIM_START = 0.2  # seconds cut from the start of every video clip (i2v ramp-up); stills are untouched
WIDTH, HEIGHT = 1080, 1920
# zoompan snaps its crop window to whole pixels each frame, so zooming/panning a
# 1080-wide frame makes edges crawl/shimmer and punch-ins soft. Zoom on a frame
# SUPERSAMPLE x larger and let zoompan scale down to the output size instead.
SUPERSAMPLE = 2
# Cap on how many cuts one stage's clip can be sliced into, however long it runs
# -- sized so the longest documented i2v duration tier (5.17s) is fully shown at the
# default ~0.94s/cut (2 beats); a lower cap leaves the tail of long clips unplayed.
MAX_SUBCUTS_PER_STAGE = 6

DEFAULT_BEAT_TIMES = config.data_path(config.DEFAULT_BEATS)


def load_beats(path):
    """(beat_times, start_time) from an audio/*.json beat file: {"start_time": s, "beats": [s, ...]} (older files: "drop_time")."""
    data = json.loads(Path(path).read_text())
    return data["beats"], data.get("start_time", data.get("drop_time"))

# Crop/pan targets within the portrait frame, cycled across cuts so one clip
# yields several distinct-feeling shots. y_center is where the zoom settles
# (0=top, 1=bottom); x_start -> x_end is the diagonal Ken-Burns drift.
FOCUS_PRESETS = {
    "wide":  {"y_center": 0.50, "x_start": 0.485, "x_end": 0.515, "zoom_start": 1.02, "zoom_settle": 1.0, "zoom_max": 1.02, "zoom_out": 1.0},
    "hips":  {"y_center": 0.78, "x_start": 0.53, "x_end": 0.47, "zoom_start": 1.05, "zoom_settle": 1.02, "zoom_max": 1.05, "zoom_out": 1.02},
    "face":  {"y_center": 0.20, "x_start": 0.47, "x_end": 0.53, "zoom_start": 1.05, "zoom_settle": 1.02, "zoom_max": 1.05, "zoom_out": 1.02},
    "chest": {"y_center": 0.45, "x_start": 0.53, "x_end": 0.47, "zoom_start": 1.05, "zoom_settle": 1.02, "zoom_max": 1.05, "zoom_out": 1.02},
}
FOCUS_SEQUENCE = ["wide", "hips", "face", "chest"]  # plain mode: cycled across all cuts
BODY_PARTS = ["hips", "face", "chest"]  # --zoom mode: subcut targets, cycled across stages


@dataclass
class Cut:
    clip: Path
    frames: int
    focus: str = "wide"
    direction: str = "in"
    offset: float = 0.0   # seconds into the source clip this cut starts at
    lead_in: bool = False
    subcut: bool = False  # 2nd+ cut of the same stage clip: punched in to --zoom on a body part
    stretch: float = 1.0  # clip retime factor (>1 = slower) to fill the cut
    seg: int = 0          # which appearance of a stage clip this cut belongs to (cuts of one appearance share it)
    flare: bool = True    # motion-blur + chromatic-aberration burst at the cut (off in a no-music edit: transitions do that job)


@dataclass
class EditOptions:
    """The knobs make_tiktok/make_scenario take besides their beat/layout inputs, bundled so the signatures stay short."""
    trims: dict | None = None     # stage id -> seconds cut from that clip's start (others: TRIM_START)
    trims_end: dict | None = None # stage id -> seconds cut from the clip's end (others: 0)
    clip_audio: bool = False      # layer the clips' own audio over the music
    clip_volume: float = 1.0
    music_volume: float = 1.0
    duck: float = 0.0             # how hard the music ducks under the clip audio
    mute_clip_audio: tuple = ()   # stage ids whose clip audio is left out
    fade_out: float = FADE_OUT
    as_still: tuple = ()          # stages that show their stage_N.png still instead of the clip
    stills_from: Path | None = None  # folder the --as-still stills come from
    polaroid: bool = False        # stills show as polaroid cards
    collage: tuple = ()           # stages whose cards pile up in the outro
    collage_from: Path | None = None  # folder the collage stills come from (default stills_from, else pipeline_dir)
    card_size: int | None = None  # polaroid card width in px of the 1080 frame (None = by count)
    collage_beats: int = 2        # beats between collage landings (beat-synced edit)
    card_sounds: bool = False     # a sound from config.POLAROID_SFX_DIR on every card landing
    card_sound_volume: float = 0.8


def durations_to_frame_counts(durations):
    """Integer frame counts from CUMULATIVE frame boundaries, not per-clip
    rounding: independent rounding lets half-frame errors stack in one
    direction over dozens of cuts (audible drift), and handing ffmpeg a float
    duration (-t 0.9333...) can floor a frame short. Integer counts fed to
    -frames:v avoid both."""
    counts, total, prev = [], 0.0, 0
    for d in durations:
        total += d
        boundary = max(round(total * FPS), prev + 1)  # never a 0-frame clip
        counts.append(boundary - prev)
        prev = boundary
    return counts


def beat_cut_durations(beat_times, drop_time, n_cuts, beats_per_cut, last_weight, fit_grid=False):
    """(punch_start, durations): each cut spans beats_per_cut (an int, or one
    count per cut) REAL detected beats starting at the first beat >= drop_time (a synthetic BPM grid
    drifts). Past the last detected beat, extrapolate at the average spacing.
    fit_grid: instead use a fixed grid (robust line fit through the detected beats
    the cuts span, outliers > 60 ms dropped) so a mis-tracked beat can't shift a cut."""
    start = bisect.bisect_left(beat_times, drop_time)
    if start >= len(beat_times):
        raise ValueError(f"drop_time {drop_time}s is past the last detected beat ({beat_times[-1]}s)")
    last = len(beat_times) - 1
    avg = (beat_times[-1] - beat_times[0]) / last

    def t(idx):
        return beat_times[idx] if idx <= last else beat_times[last] + avg * (idx - last)

    counts = [beats_per_cut] * n_cuts if isinstance(beats_per_cut, int) else beats_per_cut
    edges = [start]
    for c in counts:
        edges.append(edges[-1] + c)
    if fit_grid:
        idx = list(range(start, min(edges[-1], last) + 1))
        if len(idx) < 3:
            raise ValueError("--subcuts-per-image 1 needs at least 3 detected beats after the drop to fit a grid")
        for _ in range(3):
            m, c = np.polyfit([i - start for i in idx], [beat_times[i] for i in idx], 1)
            keep = [i for i in idx if abs(beat_times[i] - (c + m * (i - start))) < 0.06]
            idx = keep if len(keep) >= 3 else idx
        t = lambda i: c + m * (i - start)
        punch = t(start)
    else:
        punch = beat_times[start]
    durations = [t(edges[i + 1]) - t(edges[i]) for i in range(n_cuts)]
    durations[-1] *= last_weight
    return punch, durations


def _plain_cut(clip, frames, cut_i, offset) -> Cut:
    """A plain-mode cut: cycles the focus crop, alternating zoom in/out."""
    return Cut(clip, frames, FOCUS_SEQUENCE[cut_i % len(FOCUS_SEQUENCE)], "in" if cut_i % 2 == 0 else "out", offset)


def _punch_cut(clip, frames, stage_i, sub_i, offset, body_part) -> tuple:
    """A --zoom-mode cut: a stage's first cut wide, later ones punched in on a body part (cycled across stages)."""
    if sub_i == 0:
        return Cut(clip, frames, "wide", "in" if stage_i % 2 == 0 else "out", offset), body_part
    return Cut(clip, frames, BODY_PARTS[body_part % len(BODY_PARTS)], offset=offset, subcut=True), body_part + 1


def plan_cuts(clips, lead_in_dur, cut_durations, subcut_counts, punch_in=False):
    """Lead-in cut on stage 1 (if lead_in_dur > 0 -- stage 1 is then reserved
    for it, otherwise it'd appear twice), then subcut_counts[stage_i] punch cuts
    for that stage's clip. Sub-cuts of one stage seek further into the SAME clip
    so they show continued motion instead of replaying its first frames."""
    lead = [lead_in_dur] if lead_in_dur > 0 else []
    frames = durations_to_frame_counts(lead + cut_durations)
    cuts = [Cut(clips[0], frames.pop(0), lead_in=True)] if lead else []
    body_part = 0
    i = 0
    for stage_i, clip in enumerate(clips[1:] if lead else clips):
        offset = 0.0
        for sub_i in range(subcut_counts[stage_i]):
            if punch_in:
                cut, body_part = _punch_cut(clip, frames[i], stage_i, sub_i, offset, body_part)
            else:
                cut = _plain_cut(clip, frames[i], i, offset)
            cut.seg = stage_i + 1
            cuts.append(cut)
            offset += frames[i] / FPS
            i += 1
    return cuts


def _pick_sources(clips, pngs, root_pngs, order, as_still, stills) -> tuple:
    """({stage: file}, stage order) for the requested sources: as_still mixes stills and clips, else one kind for all."""
    if as_still:
        ids = order if order is not None else sorted(set(clips) | set(as_still))
        return {n: (pngs if n in as_still else clips).get(n) for n in ids}, ids
    files = clips if clips and not stills else root_pngs  # stills (without as_still) always come from the root
    ids = order if order is not None else sorted(files)
    return {n: files.get(n) for n in ids}, ids


def stage_sources(pipeline_dir, stills=False, order=None, as_still=(), stills_dir=None):
    """videos/stage_N.mp4 clips in stage-number order, or stage_N.png stills if the
    run has no clips (or stills=True). Only stage_<int> names count, so extras
    like stage_21a_facial_before.png are ignored. --order overrides the sequence
    (repeats/drops allowed) without renaming files. as_still mixes the two: those
    stages use stills_dir/stage_N.png (default pipeline_dir), every other one its clip."""
    def found(folder, ext):
        return {int(p.stem[6:]): p for p in folder.glob(f"stage_*.{ext}") if p.stem[6:].isdigit()}

    clips, pngs = found(pipeline_dir / "videos", "mp4"), found(Path(stills_dir or pipeline_dir), "png")
    pick, ids = _pick_sources(clips, pngs, found(pipeline_dir, "png"), order, as_still, stills)
    missing = [n for n in ids if pick[n] is None]
    if missing:
        raise FileNotFoundError(f"stage(s) {missing} have no {'still/clip' if as_still else 'file'} in {pipeline_dir}"
                                + (f" (stills from {stills_dir})" if as_still and stills_dir else ""))
    return [pick[n] for n in ids]


MIN_STRETCH, MAX_STRETCH = 0.8, 1.4  # clip retime limits (>1 = slower) when filling a cut


def fit_beats(clip_len, beat):
    """Even beat count (2-8) whose cut needs the least retime of a clip_len-second
    clip, preferring counts inside MIN/MAX_STRETCH so the clip neither freezes at
    its end nor gets cut off."""
    def stretch(n):
        return n * beat / clip_len
    ok = [n for n in (2, 4, 6, 8) if MIN_STRETCH <= stretch(n) <= MAX_STRETCH]
    return min(ok or (2, 4, 6, 8), key=lambda n: abs(math.log(stretch(n))))


CSV_DUR = "csv=p=0"  # ffprobe "-of" format: bare value, no header (shared by every probe)


@lru_cache(maxsize=None)
def clip_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", CSV_DUR, str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


TRIMS = {}  # stage id -> seconds cut from that clip's start (--trim ID=SECONDS); other stages use TRIM_START


TRIMS_END = {}  # stage id -> seconds cut from the end of that clip (--trim-end ID=SECONDS); other stages keep their end
POLAROID = {}  # {"size": px or None} while --polaroid is on: build_clip renders stills as cards
SFX_EXT = (".wav", ".mp3", ".ogg", ".flac", ".m4a")
SFX_LEVEL_DB = -14.0  # every card sound is levelled to this mean volume before --card-sound-volume (the music sits near -8)
SFX_LEAD = 0.03  # a card sound starts this much before the impact (the attack of a shutter/thud reads as on time)
COLLAGE_HOLD_BEATS = 4  # beats the finished pile holds after the last card lands
COLLAGE_EVERY, COLLAGE_HOLD = 0.9, 2.5  # no-music edit: seconds between landings, and the hold after the last one


def _stage_id(path) -> int:
    stem = Path(path).stem
    return int(stem[6:]) if stem[6:].isdigit() else 0


def polaroid_card(still: Path):
    return polaroid.make_card(Path(still), polaroid.caption_for(_stage_id(still)), _stage_id(still))


def build_polaroid_still(cut, out_path):
    """A still as one card dropping onto its own blurred copy: the impact lands on the cut's first frame, no developing."""
    polaroid.render_collage([polaroid_card(cut.clip)], polaroid.blurred_background(cut.clip), cut.frames, FPS,
                            [-round(polaroid.LAND * 0.6)], out_path, push=0.04, develop_s=0, seed=_stage_id(cut.clip),
                            post_filters=polaroid.POST, card_w=POLAROID.get("size") or 900)


def collage_stills(pipeline_dir, opts) -> list:
    """The collage's stage_N.png files, in the given order."""
    folder = Path(opts.collage_from or opts.stills_from or pipeline_dir)
    files = [folder / f"stage_{n}.png" for n in opts.collage]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"collage: no {', '.join(missing)} in {folder}")
    return files


def build_collage(stills, frames, land_at, out_path, card_size):
    """The outro: the stills' cards pile up over a blurred copy of the last one, exactly `frames` long."""
    cards = [polaroid_card(f) for f in stills]
    return polaroid.render_collage(cards, polaroid.blurred_background(stills[-1]), frames, FPS, land_at, out_path,
                                   post_filters=polaroid.POST, card_w=card_size, seed=len(stills))


def trim_for(path):
    stem = Path(path).stem
    return TRIMS.get(int(stem[6:]), TRIM_START) if stem[6:].isdigit() else TRIM_START


def trim_end_for(path):
    stem = Path(path).stem
    return TRIMS_END.get(int(stem[6:]), 0.0) if stem[6:].isdigit() else 0.0


def usable_duration(path):
    """Seconds of a video clip the edit uses: i2v clips ramp up slowly, so the first trim_for() seconds are cut off, and
    trim_end_for() seconds come off the end."""
    return max(0.1, clip_duration(path) - trim_for(path) - trim_end_for(path))


def build_clip(cut, out_path, punch_zoom=1.5):
    if POLAROID and cut.clip.suffix == ".png":
        return build_polaroid_still(cut, out_path)
    source, filters = cut_filters(cut, punch_zoom)
    subprocess.run(["ffmpeg", "-y", *source, "-vf", ",".join(filters),
                    "-frames:v", str(cut.frames), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", "-threads", "2", str(out_path), "-loglevel", "error"], check=True)


def _scale_filters(cut, zoom, x, y) -> list:
    """The scale/crop stage: a plain fill-crop for a no-music clip (no flare), else the supersampled zoompan chain
    (zoompan is single-threaded, the slow part)."""
    if not cut.flare and cut.clip.suffix == ".mp4":
        return [f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={WIDTH}:{HEIGHT}"]
    return [f"scale={WIDTH * SUPERSAMPLE}:{HEIGHT * SUPERSAMPLE}:force_original_aspect_ratio=increase:flags=lanczos",
            f"crop={WIDTH * SUPERSAMPLE}:{HEIGHT * SUPERSAMPLE}",
            f"zoompan=z='{zoom}':d=1:x='{x}':y='{y}':s={WIDTH}x{HEIGHT}:fps={FPS}"]


def cut_filters(cut, punch_zoom=1.5):
    """(ffmpeg input args, video filter list) that turn the cut's source into its graded 1080x1920 frames."""
    n = cut.frames
    if cut.lead_in:
        zoom = "1.0"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    else:
        p = FOCUS_PRESETS[cut.focus]
        if cut.subcut:
            # Punch in: snap 10% past punch_zoom, settle onto it by frame 3,
            # then HOLD it -- only the pan keeps moving until the next shot.
            zs = punch_zoom * 1.1
            zoom = f"if(lte(on\\,3),{zs}-{zs - punch_zoom}*on/3,{punch_zoom})"
        else:
            # Full-image shot: hard snap past zset (frames 0-1), spring back
            # (1-4), then a straight ramp to the target over the REST of the
            # clip so motion never stops before the cut; gentle pulse on top.
            zs, zset = p["zoom_start"], p["zoom_settle"]
            spring_end = zset + 0.03
            target = p["zoom_max"] if cut.direction == "in" else p["zoom_out"]
            drift = f"({spring_end}+({target}-{spring_end})*((on-4)/{max(n - 5, 1)}))"
            base = f"if(lte(on\\,1),{zs}-{zs - zset}*on,if(lte(on\\,4),{zset}+0.01*(on-1),{drift}))"
            zoom = f"(({base})+0.009*sin(2*PI*on/{max(n, 1)}))"
        x_center = f"({p['x_start']}+({p['x_end']}-{p['x_start']})*(on/{max(n - 1, 1)}))"
        y = f"min(max(({p['y_center']}*ih)-(ih/zoom/2)\\,0)\\,ih-ih/zoom)"
        x = f"min(max(({x_center}*iw)-(iw/zoom/2)\\,0)\\,iw-iw/zoom)"

    filters = [
        *([f"setpts=PTS*{cut.stretch:.5f}"] if abs(cut.stretch - 1.0) > 1e-6 else []),
        f"fps={FPS}",  # MiniMax clips are 24fps; resample so zoompan's "on" matches our frame math
        # If the clip runs out, hold its last frame: -frames:v is only a max,
        # and a short cut would push every later cut off the beat.
        "tpad=stop_mode=clone:stop_duration=10",
        *_scale_filters(cut, zoom, x, y),
        "setsar=1",
        "eq=saturation=1.08:contrast=1.04:brightness=0.035:gamma=1.08",  # warm, rich grade; gamma/brightness lift keep it from reading dark
        "colorbalance=rm=0.025:gm=0.0:bm=-0.03:rs=0.015:bs=-0.015",
        "vignette=angle=PI/10",  # SMALLER angle = weaker vignette (default PI/5, max PI/2 = near-black corners)
        "noise=alls=10:allf=t+u",
    ]
    if not cut.lead_in and cut.flare:  # motion-blur streak + chromatic-aberration burst at the cut
        filters.append("tmix=frames=3:enable='lte(t\\,0.1)'")
        # Kept subtle: at the old 12px the fringe read as rainbow edges on fine hair.
        filters.append("rgbashift=rh=3:rv=1:bh=-3:bv=-1:enable='lte(t\\,0.18)'")

    # If the clip is too short for this sub-cut's running offset (e.g. 2 x 0.93s
    # from a 1.625s clip), start earlier so it ends on the clip's last frame --
    # a little motion repeats under a different crop instead of freezing.
    if cut.clip.suffix == ".png":
        source = ["-loop", "1", "-framerate", str(FPS), "-i", str(cut.clip)]  # still: loop it for the cut's length
    else:
        offset = trim_for(cut.clip) + (max(0.0, min(cut.offset, usable_duration(cut.clip) - n / FPS)) if cut.offset > 0 else 0.0)
        source = ["-ss", f"{offset:.4f}", "-i", str(cut.clip)]
    return source, filters


@lru_cache(maxsize=None)
def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", CSV_DUR, str(path)],
                         capture_output=True, text=True, check=True).stdout
    return bool(out.strip())


def voice_segments(cuts, muted=()):
    """One entry per appearance of a clip with its own audio: (clip, edit start s, edit length s, source start s,
    source length s, atempo). The clip's audio plays ONCE at its stage's start in the edit, however many sub-cuts
    the picture is sliced into (per-cut audio would repeat/cut off the dialogue). Stretched clips (one cut per
    stage) get atempo = 1/stretch to stay in step with the retimed picture."""
    segs, t = [], 0
    for _, group in itertools.groupby(cuts, key=lambda c: c.seg):
        group = list(group)
        frames = sum(c.frames for c in group)
        c0 = group[0]
        if c0.clip.suffix != ".mp4" or not has_audio(c0.clip):
            t += frames
            continue
        stem = c0.clip.stem
        if stem[6:].isdigit() and int(stem[6:]) in muted:
            t += frames
            continue
        edit_len = frames / FPS
        src_len = min(edit_len / c0.stretch, usable_duration(c0.clip))
        segs.append((c0.clip, t / FPS, edit_len, trim_for(c0.clip), src_len, 1 / c0.stretch))
        t += frames
    return segs


def mux(clip_paths, build_dir, audio, audio_start, total_dur, out_path, voices=(), clip_volume=1.0, music_volume=1.0, duck=0.0, fade_out=FADE_OUT):
    fade = min(max(0.0, fade_out), total_dur)
    fade_f = f"afade=t=out:st={total_dur - fade:.3f}:d={fade:.3f}" if fade > 0 else ""  # applied to the final mix
    concat_list = build_dir / "concat_list.txt"
    concat_list.write_text("".join(f"file '{p.name}'\n" for p in clip_paths))
    silent = build_dir / "_silent_concat.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                    "-c", "copy", str(silent), "-loglevel", "error"], check=True)
    if not voices:
        subprocess.run(["ffmpeg", "-y", "-i", str(silent), "-ss", str(audio_start), "-i", str(audio),
                        "-map", "0:v", "-map", "1:a", "-t", str(total_dur), "-c:v", "copy", *(["-af", fade_f] if fade_f else []),
                        "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path), "-loglevel", "error"], check=True)
        return
    # inputs: 0 = picture, 1 = music, 2.. = one per voiced appearance
    inputs = ["-i", str(silent), "-ss", str(audio_start), "-i", str(audio)]
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
    chains = []
    for k, (clip, start, edit_len, src_start, src_len, tempo) in enumerate(voices):
        inputs += ["-i", str(clip)]
        tempo_f = f",atempo={tempo:.5f}" if abs(tempo - 1) > 1e-3 else ""
        chains.append(f"[{k + 2}:a]atrim=start={src_start:.4f}:duration={src_len:.4f},asetpts=PTS-STARTPTS{tempo_f},{fmt},"
                      f"volume={clip_volume},adelay={round(start * 1000)}:all=1[v{k}]")
    n = len(voices)
    voice_bus = f"{''.join(f'[v{k}]' for k in range(n))}amix=inputs={n}:normalize=0:duration=longest,apad[voice]" if n > 1 else "[v0]apad[voice]"
    chains.append(voice_bus)
    chains.append(f"[1:a]{fmt},volume={music_volume}[music]")
    if duck > 0:
        chains.append("[voice]asplit=2[vkey][vmix]")
        chains.extend([
            f"[music][vkey]sidechaincompress=threshold=0.02:ratio={2 + 10 * duck:.2f}:attack=20:release=350[ducked]",
            f"[ducked][vmix]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95{',' + fade_f if fade_f else ''}[aout]",
        ])
    else:
        chains.append(f"[music][voice]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95{',' + fade_f if fade_f else ''}[aout]")
    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(chains), "-map", "0:v", "-map", "[aout]",
                    "-t", str(total_dur), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out_path), "-loglevel", "error"], check=True)


EDIT_NICE = 15  # scheduling priority of the render (19 = lowest)
CPU_SHARE = 0.375  # share of the physical cores the edit renders on (with their SMT siblings); the rest stay free for other work


def edit_cpus():
    """Logical CPUs to render on: CPU_SHARE of the physical cores, each with all its hardware threads, so the cores left out stay
    completely idle (pinning half the logical CPUs would still load every core through its sibling)."""
    cpus = sorted(os.sched_getaffinity(0))
    cores = {}
    for c in cpus:
        try:
            key = Path(f"/sys/devices/system/cpu/cpu{c}/topology/core_id").read_text().strip(), \
                  Path(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id").read_text().strip()
        except OSError:
            key = (str(c), "0")
        cores.setdefault(key, []).append(c)
    groups = list(cores.values())
    keep = groups[:max(1, int(len(groups) * CPU_SHARE))]
    return [c for g in keep for c in g]


def _render_pool() -> list:
    """Pin this process (the ffmpeg children inherit it) to an idle share of the cores, at low priority; returns the CPUs."""
    cpus = edit_cpus()
    os.sched_setaffinity(0, cpus)
    os.nice(EDIT_NICE)  # a render yields to anything interactive
    return cpus


def edit_fps(clips):
    """HI_FPS when every video source is a clip at HI_FPS or more (an interp/ folder), else the default FPS: 24 fps clips keep the
    30 fps edit. Stills loop at any rate, so they don't count (an all-stills edit stays at 30)."""
    videos = [c for c in clips if c.suffix == ".mp4"]
    if videos and all(clip_fps(c) >= HI_FPS for c in videos):
        return HI_FPS
    return 30


def clip_fps(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", CSV_DUR,
                          str(path)], capture_output=True, text=True, check=True).stdout.strip()
    num, den = out.split("/")
    return int(num) / int(den)


def _out_path(pipeline_dir, out_name) -> Path:
    """<folder>/edits/<out_name or the folder's name>.mp4; an absolute out_name is used as is."""
    return Path(out_name) if out_name and Path(out_name).is_absolute() else Path(pipeline_dir) / "edits" / (out_name or f"{Path(pipeline_dir).resolve().name}.mp4")


def _cut_names(cuts) -> list:
    """cut_000_stage_1_wide-in-off0.00-style file names (one per cut)."""
    return [f"cut_{i:03d}_{c.clip.stem}_" + ("leadin" if c.lead_in else f"{c.focus}{'-punch' if c.subcut else '-' + c.direction}-off{c.offset:.2f}")
            for i, c in enumerate(cuts)]


def _set_polaroid(opts):
    POLAROID.clear()
    if opts.polaroid:
        POLAROID["size"] = opts.card_size


def card_sound_files() -> list:
    files = sorted(f for f in config.POLAROID_SFX_DIR.glob("*") if f.suffix.lower() in SFX_EXT)
    if not files:
        raise FileNotFoundError(f"card sounds: no sound files in {config.POLAROID_SFX_DIR} -- put a few camera sounds there "
                                f"(shutter, flash, film eject...; {', '.join(SFX_EXT)})")
    return files


def card_times(cuts, overlap=0, collage_land=(), collage_start=None) -> list:
    """Edit seconds of every card impact: each polaroid still's first frame (its card hits on it) and each collage landing.
    overlap = frames each later cut overlaps the one before (a no-music edit's transitions)."""
    if not POLAROID and not collage_land:
        return []
    times, start, drop = [], 0, round(polaroid.LAND * 0.6)
    for k, c in enumerate(cuts):
        begin = start - k * overlap
        if POLAROID and c.clip.suffix == ".png":
            times.append(begin / FPS)
        if c.clip.name == "_collage":
            collage_start = begin
        start += c.frames
    if collage_land:
        times += [(collage_start + la + drop) / FPS for la in collage_land]
    return sorted(times)


@lru_cache(maxsize=None)
def sfx_gain_db(path: Path) -> float:
    """dB that brings the file's sound (leading silence cut) to SFX_LEVEL_DB mean volume, so files of any loudness sit alike."""
    r = subprocess.run(["ffmpeg", "-i", str(path), "-af", "silenceremove=start_periods=1:start_threshold=-40dB,volumedetect",
                        "-f", "null", "-"], capture_output=True, text=True)
    mean = next((float(l.split("mean_volume:")[1].split()[0]) for l in r.stderr.splitlines() if "mean_volume:" in l), SFX_LEVEL_DB)
    return SFX_LEVEL_DB - mean


def add_card_sounds(out_path: Path, times, volume, seed=0):
    """Mix a camera sound into the finished edit at each time: the files are dealt out shuffled so neighbours differ, and every
    play is varied (pitch/speed +-8%, a random high- or low-cut, level) so even one file never repeats exactly. Each file's
    leading silence is cut, so the sound starts on its hit. The picture is copied as is."""
    files, rng = card_sound_files(), np.random.default_rng(seed)
    deck = []
    inputs, chains = ["-i", str(out_path)], []
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
    for k, t in enumerate(times):
        if not deck:
            deck = list(rng.permutation(len(files)))
        f = files[deck.pop()]
        inputs += ["-i", str(f)]
        gain = volume * rng.uniform(0.75, 1.1) * 10 ** (sfx_gain_db(f) / 20)
        pitch = rng.uniform(0.92, 1.08)  # resampled: higher = shorter and brighter
        tone = f"highpass=f={rng.uniform(150, 600):.0f}" if rng.random() < 0.5 else f"lowpass=f={rng.uniform(4500, 12000):.0f}"
        chains.append(f"[{k + 1}:a]silenceremove=start_periods=1:start_threshold=-40dB,{fmt},asetrate={48000 * pitch:.0f},"
                      f"aresample=48000,{tone},volume={gain:.3f},adelay={max(0, round((t - SFX_LEAD) * 1000))}:all=1[s{k}]")
    n = len(times)
    chains.append(f"[0:a]{fmt}[main]")
    chains.append(f"[main]{''.join(f'[s{k}]' for k in range(n))}amix=inputs={n + 1}:normalize=0:duration=first,"
                  f"alimiter=limit=0.95[aout]")
    tmp = out_path.with_name(out_path.stem + ".sfx.mp4")
    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(chains), "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(tmp), "-loglevel", "error"], check=True)
    os.replace(tmp, out_path)
    print(f"  card sounds: {n} from {len(files)} file(s) in {config.POLAROID_SFX_DIR}", flush=True)


def beat_collage(beat_times, audio_start, start_frame, n, every):
    """(frames, land_at) of the collage outro starting at edit frame start_frame: card k hits the table on beat b0 + k*every
    (b0 = the first beat it can drop onto), the pile then holds COLLAGE_HOLD_BEATS beats. Beats past the detected ones are
    extrapolated at the average spacing."""
    avg = (beat_times[-1] - beat_times[0]) / (len(beat_times) - 1)

    def bt(i):
        return beat_times[i] if i < len(beat_times) else beat_times[-1] + avg * (i - len(beat_times) + 1)

    def frame(i):  # edit frame of beat i, relative to the collage's start
        return round((bt(i) - audio_start) * FPS) - start_frame
    drop = round(polaroid.LAND * 0.6)  # frames from the start of a drop to its impact
    b0 = bisect.bisect_left(beat_times, audio_start + (start_frame + drop) / FPS)
    land_at = [frame(b0 + k * every) - drop for k in range(n)]
    return frame(b0 + (n - 1) * every + COLLAGE_HOLD_BEATS), land_at


def make_tiktok(pipeline_dir, out_name=None, audio=None, beat_times_file=DEFAULT_BEAT_TIMES,
                start_time=None, lead_in_seconds=2.0, subcuts_per_image=2, beats_per_cut=2,
                last_weight=2.0, zoom=None, stills=False, order=None, opts: EditOptions = None) -> Path:
    """Beat-synced edit over the music (see the module docstring)."""
    global FPS
    opts = opts or EditOptions()
    TRIMS.clear()
    TRIMS.update(opts.trims or {})
    TRIMS_END.clear()
    TRIMS_END.update(opts.trims_end or {})
    _set_polaroid(opts)
    clips = stage_sources(pipeline_dir, stills, order, opts.as_still, opts.stills_from)
    if len(clips) < 2:
        raise FileNotFoundError(f"Need at least 2 stages (clips or stills) in {pipeline_dir}")
    FPS = edit_fps(clips)
    collage_files = collage_stills(pipeline_dir, opts) if opts.collage else []
    if opts.card_sounds:
        card_sound_files()  # fail before rendering, not after
    if collage_files:
        last_weight = 1.0  # the pile is the ending: no long final hold before it

    beat_times_file = config.data_path(beat_times_file)  # relative paths (audio/<Song>.json) live in the data dir
    beat_times, file_start = load_beats(beat_times_file)
    start_time = file_start if start_time is None else start_time
    if start_time is None:
        raise ValueError(f"{beat_times_file} has no start_time; pass --start-time")
    audio = config.data_path(audio) if audio else Path(beat_times_file).with_suffix(".mp3")
    if not audio.is_file():
        raise FileNotFoundError(f"no audio: pass --audio, or put {audio.name} next to the beat file")
    first_cut = start_time + lead_in_seconds  # the cuts start on the first beat after the lead-in
    punch_start = beat_times[min(bisect.bisect_left(beat_times, first_cut), len(beat_times) - 1)]

    def timing(punch):
        """(audio_start, lead_in): the music plays from the start time; the stage-1 lead-in fills up to the first cut's beat.
        No lead-in asked for: the edit opens on that beat."""
        audio_start = punch if lead_in_seconds <= 0 else max(0.0, min(start_time, punch))
        return audio_start, punch - audio_start
    audio_start, lead_in = timing(punch_start)

    # Scale each stage's cut count to its own clip length instead of a fixed subcuts_per_image everywhere, so a longer
    # i2v render actually gets shown instead of mostly sitting unused behind a couple of cuts. Stills have no inherent
    # duration to scale against, so they keep the flat --subcuts-per-image count.
    avg_beat = (beat_times[-1] - beat_times[0]) / (len(beat_times) - 1)
    avg_cut_dur = beats_per_cut * avg_beat
    body_clips = clips[1:] if lead_in > 0 else clips
    subcut_counts = [
        (1 if POLAROID else subcuts_per_image) if clip.suffix == ".png" or subcuts_per_image == 1
        else max(1, min(MAX_SUBCUTS_PER_STAGE, round(usable_duration(clip) / avg_cut_dur)))
        for clip in body_clips
    ]
    n_cuts = sum(subcut_counts)
    # One cut per stage: hold each clip for its whole length, rounded to whole beats.
    one_cut = subcuts_per_image == 1
    cut_beats = ([fit_beats(usable_duration(c), avg_beat) if c.suffix != ".png" else beats_per_cut for c in body_clips]
                 if one_cut else beats_per_cut)
    if one_cut and body_clips[-1].suffix != ".png":
        # Final hold: last_weight x the beats, but no longer than the clip can be slowed to fill.
        cap = 2 * int(MAX_STRETCH * usable_duration(body_clips[-1]) / avg_beat / 2)
        cut_beats[-1] = max(cut_beats[-1], min(2 * round(cut_beats[-1] * last_weight / 2), cap))
    punch_start, durations = beat_cut_durations(beat_times, first_cut, n_cuts, cut_beats, 1.0 if one_cut else last_weight, fit_grid=one_cut)
    if one_cut:
        audio_start, lead_in = timing(punch_start)
    cuts = plan_cuts(clips, lead_in, durations, subcut_counts, punch_in=zoom is not None)
    if one_cut:
        for c in cuts:
            if not c.lead_in and c.clip.suffix != ".png":
                c.stretch = c.frames / FPS / usable_duration(c.clip)  # fill the cut exactly: never freeze
    total_frames = sum(c.frames for c in cuts)
    collage = beat_collage(beat_times, audio_start, total_frames, len(collage_files), opts.collage_beats) if collage_files else None
    if collage:
        total_frames += collage[0]
        print(f"  collage outro: {len(collage_files)} cards, one every {opts.collage_beats} beats, {collage[0]} frames", flush=True)

    print(f"{pipeline_dir.name}: {len(clips)} stages (first = lead-in), crops {subcut_counts} x {cut_beats} beats = "
          f"{n_cuts} cuts (last={cuts[-1].frames} frames), lead-in {lead_in:.2f}s, "
          f"punch-in {f'{zoom}x' if zoom else 'off'}, music from {audio_start:.2f}s, first cut at {punch_start:.2f}s, "
          f"total {total_frames} frames / {total_frames / FPS:.3f}s at {FPS} fps", flush=True)

    build_dir = pipeline_dir / "_debug" / "tiktok_build_beatsync_dynamic_video_flare"
    shutil.rmtree(build_dir, ignore_errors=True)  # no stale cuts from a previous layout
    build_dir.mkdir(parents=True)
    names = _cut_names(cuts)

    def render(item):
        i, cut = item
        path = build_dir / f"{names[i]}.mp4"
        build_clip(cut, path, zoom)
        return path

    with ThreadPoolExecutor(max_workers=len(_render_pool())) as pool:  # zoompan is single-threaded; cuts are independent files
        clip_paths = list(pool.map(render, enumerate(cuts)))  # map keeps cut order for the concat
    if collage:
        clip_paths.append(build_collage(collage_files, *collage, build_dir / "cut_999_collage.mp4", opts.card_size))

    out_path = _out_path(pipeline_dir, out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    voices = voice_segments(cuts, set(opts.mute_clip_audio)) if opts.clip_audio else []
    if opts.clip_audio:
        print(f"  clip audio: {len(voices)} of {len({c.seg for c in cuts})} clips carry sound (music x{opts.music_volume}, "
              f"voice x{opts.clip_volume}, duck {opts.duck})", flush=True)
    mux(clip_paths, build_dir, audio, audio_start, total_frames / FPS, out_path, voices, opts.clip_volume, opts.music_volume,
        opts.duck, opts.fade_out)
    if opts.card_sounds:
        times = card_times(cuts, collage_land=collage[1] if collage else (), collage_start=total_frames - collage[0] if collage else None)
        if times:
            add_card_sounds(out_path, times, opts.card_sound_volume, seed=len(times))
    print(f"  -> {out_path}")
    return out_path


# xfade transitions offered for a no-music edit ("cut" = hard cuts, no xfade)
TRANSITIONS = ("fade", "fadeblack", "fadewhite", "dissolve", "fadeslow", "smoothleft", "smoothup", "slideleft", "wipeleft",
               "circleopen", "radial", "zoomin", "hblur", "pixelize", "cut")
STILL_SECONDS = 3.0  # how long a still holds in a no-music edit


def scenario_cuts(clips, still_seconds, transition_seconds):
    """One full-length cut per stage for a no-music edit (clips skip zoompan in build_clip: flare=False), plus the transition length in frames (shortened to fit the shortest cut)."""
    cuts = []
    for i, clip in enumerate(clips):
        secs = still_seconds if clip.suffix == ".png" else usable_duration(clip)
        cuts.append(Cut(clip, max(2, int(secs * FPS + 1e-6)), "wide", "in" if i % 2 == 0 else "out", flare=False, seg=i))
    t = min(round(transition_seconds * FPS), min(c.frames for c in cuts) // 2)
    return cuts, max(0, t)


def _scenario_input(k, c, cut_paths, lens, fmt, muted, clip_volume, chains) -> list:
    """Input k's video chain (a pre-rendered still, or the clip graded inline) and its audio chain (silence when muted/no
    audio); returns this input's ffmpeg -i arguments."""
    if k in cut_paths:
        ins = ["-i", str(cut_paths[k])]
        vf = "null"
    else:
        ins, filters = cut_filters(c)
        vf = ",".join(filters)
    chains.append(f"[{k}:v]{vf},trim=end_frame={c.frames},settb=AVTB,fps={FPS},setpts=PTS-STARTPTS[v{k}]")
    stem = c.clip.stem
    mute = stem[6:].isdigit() and int(stem[6:]) in muted
    if k not in cut_paths and has_audio(c.clip) and not mute:
        chains.append(f"[{k}:a]asetpts=PTS-STARTPTS,{fmt},volume={clip_volume},apad,atrim=duration={lens[k]:.4f}[a{k}]")
    else:
        chains.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={lens[k]:.4f}[a{k}]")
    return ins


def _fade_chains(total, fade_in, fade_out, chains) -> None:
    """The final fade in/out on both picture and sound."""
    vf, af = [], []
    fi, fo = min(max(0.0, fade_in), total / 2), min(max(0.0, fade_out), total / 2)
    if fi > 0:
        vf.append(f"fade=t=in:d={fi:.3f}")
        af.append(f"afade=t=in:d={fi:.3f}")
    if fo > 0:
        vf.append(f"fade=t=out:st={total - fo:.3f}:d={fo:.3f}")
        af.append(f"afade=t=out:st={total - fo:.3f}:d={fo:.3f}")
    chains.append(f"[vj]{','.join(vf) or 'null'},format=yuv420p[vout]")
    chains.append(f"[aj]{','.join(af) or 'anull'},alimiter=limit=0.95[aout]")


def scenario_mux(cut_paths, cuts, t_frames, transition, out_path, muted=(), clip_volume=1.0, fade_in=0.0, fade_out=FADE_OUT):
    """Grade the clips and join every cut with xfade (acrossfade for the clips' own audio) in ONE ffmpeg pass, so each frame is
    encoded once; cut_paths = {cut index: pre-rendered file} for the stills (their zoompan drift). Each transition overlaps
    two cuts by t_frames, so the total is sum(frames) - (n-1) * t_frames."""
    n = len(cuts)
    lens = [c.frames / FPS for c in cuts]
    total_frames = sum(c.frames for c in cuts) - (n - 1) * t_frames
    total, d = total_frames / FPS, t_frames / FPS
    inputs, chains = [], []
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
    for k, c in enumerate(cuts):  # input k = cut k (a clip seeked past its trim, or a still's pre-rendered file)
        inputs += _scenario_input(k, c, cut_paths, lens, fmt, muted, clip_volume, chains)
    if t_frames == 0:
        chains.append("".join(f"[v{k}][a{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=1[vj][aj]")
    else:
        v, a, end = "v0", "a0", lens[0]
        for k in range(1, n):
            chains.append(f"[{v}][v{k}]xfade=transition={transition}:duration={d:.4f}:offset={end - d:.4f}[x{k}]")
            chains.append(f"[{a}][a{k}]acrossfade=d={d:.4f}:c1=tri:c2=tri[y{k}]")
            v, a, end = f"x{k}", f"y{k}", end - d + lens[k]
        chains.append(f"[{v}]null[vj]")
        chains.append(f"[{a}]anull[aj]")
    _fade_chains(total, fade_in, fade_out, chains)

    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(chains), "-map", "[vout]", "-map", "[aout]",
                    "-frames:v", str(total_frames), "-t", f"{total:.4f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", str(out_path), "-loglevel", "error"], check=True)
    return total_frames


def make_scenario(pipeline_dir, out_name=None, stills=False, order=None, transition="fade", transition_seconds=0.6,
                  still_seconds=STILL_SECONDS, fade_in=0.0, opts: EditOptions = None) -> Path:
    """No-music edit: every stage in full, joined by transitions, over the clips' own audio (see the module docstring)."""
    global FPS
    opts = opts or EditOptions()
    if transition not in TRANSITIONS:
        raise ValueError(f"transition must be one of {', '.join(TRANSITIONS)}")
    TRIMS.clear()
    TRIMS.update(opts.trims or {})
    TRIMS_END.clear()
    TRIMS_END.update(opts.trims_end or {})
    _set_polaroid(opts)
    clips = stage_sources(pipeline_dir, stills, order, opts.as_still, opts.stills_from)
    if len(clips) < 2:
        raise FileNotFoundError(f"Need at least 2 stages (clips or stills) in {pipeline_dir}")
    FPS = edit_fps(clips)
    collage_files = collage_stills(pipeline_dir, opts) if opts.collage else []
    if opts.card_sounds:
        card_sound_files()
    cuts, t_frames = scenario_cuts(clips, still_seconds, transition_seconds)
    if transition == "cut":
        t_frames = 0
    land_at = []
    if collage_files:  # the outro: a cut of its own; its first card lands once the transition into it is over
        land_at = [t_frames + round(k * COLLAGE_EVERY * FPS) for k in range(len(collage_files))]
        cuts.append(Cut(Path("_collage"), land_at[-1] + round(COLLAGE_HOLD * FPS), flare=False, seg=len(cuts)))
    total_frames = sum(c.frames for c in cuts) - (len(cuts) - 1) * t_frames
    print(f"{pipeline_dir.name}: {len(clips)} stages, no music, transition {transition} {t_frames / FPS:.2f}s, "
          f"fade in {fade_in}s / out {opts.fade_out}s, total {total_frames} frames / {total_frames / FPS:.3f}s at {FPS} fps", flush=True)

    build_dir = pipeline_dir / "_debug" / "tiktok_build_scenario"
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True)
    def render(i):
        path = build_dir / f"cut_{i:03d}_{cuts[i].clip.stem}.mp4"
        build_clip(cuts[i], path)
        return i, path

    with ThreadPoolExecutor(max_workers=len(_render_pool())) as pool:  # only stills are pre-rendered; clips are graded in the final pass
        paths = dict(pool.map(render, [i for i, c in enumerate(cuts) if c.clip.suffix == ".png"]))
    if collage_files:
        paths[len(cuts) - 1] = build_collage(collage_files, cuts[-1].frames, land_at, build_dir / "cut_999_collage.mp4", opts.card_size)

    out_path = _out_path(pipeline_dir, out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_mux(paths, cuts, t_frames, transition, out_path, set(opts.mute_clip_audio), opts.clip_volume, fade_in, opts.fade_out)
    times = card_times(cuts, t_frames, land_at) if opts.card_sounds else []
    if times:
        add_card_sounds(out_path, times, opts.card_sound_volume, seed=len(times))
    print(f"  -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pipeline_dir", type=Path)
    ap.add_argument("--out-name", default=None, help="Output file name, written to <folder>/edits/ (default: <run folder name>.mp4); an absolute path is used as is")
    ap.add_argument("--audio", type=Path, default=None, help="Track to mux in and cut to (default: the .mp3 next to --beat-times-file)")
    ap.add_argument("--beat-times-file", type=Path, default=DEFAULT_BEAT_TIMES, help='audio/<track>.json: {"start_time": s, "beats": [s, ...]}')
    ap.add_argument("--start-time", type=float, default=None, help="Override the beat file's start_time (s in --audio where the video starts)")
    ap.add_argument("--lead-in-seconds", type=float, default=2.0, help="Calm hold on stage 1 before the first cut")
    ap.add_argument("--subcuts-per-image", type=int, default=2,
                    help="Crops per stage for stills (video clips scale automatically by their own length instead, capped at "
                         f"{MAX_SUBCUTS_PER_STAGE})")
    ap.add_argument("--beats-per-cut", type=int, default=2, help="Beats each cut holds for")
    ap.add_argument("--last-weight", type=float, default=2.0, help="How many times longer the final cut holds")
    ap.add_argument("--stills", action="store_true", help="Use stage_N.png stills even where i2v clips exist")
    ap.add_argument("--order", type=int, nargs="+", default=None,
                    help="Stage numbers in play order (first = lead-in), e.g. --order 3 2 1 4. "
                         "Repeats/drops allowed. Default: ascending stage-number order.")
    ap.add_argument("--trim", nargs="+", default=[], metavar="ID=SECONDS",
                    help=f"Per-stage seconds cut from the start of that stage's clip (default {TRIM_START} for the others)")
    ap.add_argument("--trim-end", nargs="+", default=[], metavar="ID=SECONDS",
                    help="Per-stage seconds cut from the end of that stage's clip (default 0)")
    ap.add_argument("--zoom", type=float, nargs="?", const=1.5, default=None,
                    help="Punch-in mode: each stage's first cut shows the full image, later cuts punch in to this zoom "
                         "on a body part and hold it (pan keeps drifting). --zoom alone = 1.5x; omit for plain cuts")
    ap.add_argument("--clip-audio", action="store_true",
                    help="Layer the clips' own generated audio (dialogue, breathing, ambience) over the music: each clip's sound plays once at its stage's start")
    ap.add_argument("--clip-volume", type=float, default=1.0, help="Level of the clip-audio layer")
    ap.add_argument("--music-volume", type=float, default=1.0, help="Level of the music (with --clip-audio)")
    ap.add_argument("--duck", type=float, default=0.0, help="How hard the music ducks under the clip audio, 0 = off (with --clip-audio)")
    ap.add_argument("--fade-out", type=float, default=FADE_OUT, help=f"Seconds the audio fades out over at the end (default {FADE_OUT}; 0 = hard stop)")
    ap.add_argument("--as-still", type=int, nargs="+", default=[], metavar="ID",
                    help="Mix: these stages show their stage_N.png still (cut like --stills), the rest their clip")
    ap.add_argument("--stills-from", type=Path, default=None, help="Folder the --as-still stills come from (default pipeline_dir)")
    ap.add_argument("--no-music", action="store_true",
                    help="Scenario edit: no beats; every stage in full, joined by --transition, over the clips' own audio")
    ap.add_argument("--transition", default="fade", choices=TRANSITIONS, help="No-music edit: xfade transition between stages (cut = hard cuts)")
    ap.add_argument("--transition-seconds", type=float, default=0.6, help="No-music edit: length of each transition")
    ap.add_argument("--still-seconds", type=float, default=STILL_SECONDS, help="No-music edit: how long a still holds")
    ap.add_argument("--fade-in", type=float, default=0.0, help="No-music edit: seconds the edit fades in from black")
    ap.add_argument("--mute-clip-audio", type=int, nargs="+", default=[], metavar="ID", help="Stage ids whose clip audio is left out")
    ap.add_argument("--polaroid", action="store_true", help="Show every still as a polaroid card (one cut per still)")
    ap.add_argument("--collage", type=int, nargs="+", default=[], metavar="ID", help="End on these stages' polaroid cards piling up")
    ap.add_argument("--collage-from", type=Path, default=None, help="Folder the collage stills come from (default --stills-from, else pipeline_dir)")
    ap.add_argument("--collage-beats", type=int, default=2, help="Beats between collage landings")
    ap.add_argument("--card-sounds", action="store_true", help=f"A camera sound on every card landing, picked from {config.POLAROID_SFX_DIR}")
    ap.add_argument("--card-sound-volume", type=float, default=0.8, help="Level of the card sounds")
    ap.add_argument("--card-size", type=int, default=None, help="Polaroid card width in px of the 1080-wide frame (default: by card count)")
    a = ap.parse_args()
    trims = {int(k): float(v) for k, v in (t.split("=") for t in a.trim)}
    trims_end = {int(k): float(v) for k, v in (t.split("=") for t in a.trim_end)}
    opts = EditOptions(trims, trims_end, a.clip_audio, a.clip_volume, a.music_volume, a.duck, tuple(a.mute_clip_audio),
                       a.fade_out, tuple(a.as_still), a.stills_from, a.polaroid, tuple(a.collage), a.collage_from, a.card_size,
                       a.collage_beats, a.card_sounds, a.card_sound_volume)
    if a.no_music:
        make_scenario(a.pipeline_dir, a.out_name, a.stills, a.order, a.transition, a.transition_seconds, a.still_seconds,
                      a.fade_in, opts)
        return
    make_tiktok(a.pipeline_dir, a.out_name, a.audio, a.beat_times_file, a.start_time, a.lead_in_seconds,
                a.subcuts_per_image, a.beats_per_cut, a.last_weight, a.zoom, a.stills, a.order, opts)


if __name__ == "__main__":
    main()
