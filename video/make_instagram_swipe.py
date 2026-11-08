#!/usr/bin/env python3
"""
Silent, swipe-transition edit from a pipeline's i2v clips (pipeline_dir/videos/stage_N.mp4)
or its stage_N.png stills (--stills forces stills) -- mimics scrolling through someone's
photo/video posts by swiping horizontally between them. No app chrome (no UI, icons,
usernames) is drawn -- just the photos/clips themselves with a swipe transition, and no
audio track.

Each stage holds for its own clip length minus --trim-start-seconds (skips the i2v render's
often-unsettled opening frames), or --hold-seconds for a still -- full-bleed cover-cropped to
the frame with no Ken-Burns pan/zoom, a still stays static and a clip just plays. Consecutive
stages are joined by a --swipe-seconds horizontal xfade (slideleft by default, as if swiping to
the next post; --direction right for slideright, or --transition for any other ffmpeg xfade
transition name -- e.g. coverleft/revealright for a photo sliding over/off the next one, or
diagtl/diagtr/diagbl/diagbr for a diagonal corner-wipe peel look. ffmpeg's xfade has no true
page-curl transition, so these are the closest built-in approximations of a "peel away").

Usage:
    python3 video/make_instagram_swipe.py data/pipelines/<run> [--stills] [--direction left|right] [--transition diagtl]

Stages play in stage-number order by default; pass --order to play a different sequence
(repeats/drops allowed) without renaming files -- see make_edit.py's stage_sources(),
reused here, for why renaming breaks motion_prompts.json/stage_durations.json.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_edit import stage_sources, clip_duration, WIDTH, HEIGHT, FPS  # noqa: E402

DIRECTIONS = {"left": "slideleft", "right": "slideright"}


def build_segment(source, out_path, hold_seconds, trim_start_seconds):
    """Scale+crop one stage (video or still) to WIDTHxHEIGHT, cover-fill, no pan/zoom.
    A still is held for hold_seconds; a clip plays its own length minus trim_start_seconds
    (skipped from the front, clamped so a short clip never goes to zero/negative length)."""
    filters = [
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={WIDTH}:{HEIGHT}",
        "setsar=1",
        f"fps={FPS}",
    ]
    if source.suffix == ".png":
        cmd = ["-loop", "1", "-framerate", str(FPS), "-t", str(hold_seconds), "-i", str(source)]
    else:
        trim = min(trim_start_seconds, max(clip_duration(source) - 0.2, 0.0))
        cmd = ["-i", str(source)] + (["-ss", str(trim)] if trim > 0 else [])
    subprocess.run(["ffmpeg", "-y", *cmd, "-vf", ",".join(filters), "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    str(out_path), "-loglevel", "error"], check=True)


def make_instagram_swipe(pipeline_dir, out_name="instagram_edit.mp4", hold_seconds=1.4,
                         swipe_seconds=0.35, direction="left", transition=None,
                         trim_start_seconds=0.05, stills=False, order=None):
    sources = stage_sources(pipeline_dir, stills, order)
    if len(sources) < 2:
        raise FileNotFoundError(f"Need at least 2 stages (clips or stills) in {pipeline_dir}")

    build_dir = pipeline_dir / "_debug" / "instagram_swipe_build"
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True)

    segments = []
    for i, src in enumerate(sources):
        seg_path = build_dir / f"seg_{i:03d}_{src.stem}.mp4"
        build_segment(src, seg_path, hold_seconds, trim_start_seconds)
        dur = hold_seconds if src.suffix == ".png" else clip_duration(seg_path)
        segments.append((seg_path, dur))

    transition = transition or DIRECTIONS[direction]
    inputs = []
    filter_parts = []
    for i, (seg_path, _) in enumerate(segments):
        inputs += ["-i", str(seg_path)]
        filter_parts.append(f"[{i}:v]settb=AVTB[v{i}]")

    running = segments[0][1]
    label = "v0"
    for i in range(1, len(segments)):
        offset = running - swipe_seconds
        next_label = f"x{i}"
        filter_parts.append(
            f"[{label}][v{i}]xfade=transition={transition}:duration={swipe_seconds}:"
            f"offset={offset:.3f}[{next_label}]"
        )
        running += segments[i][1] - swipe_seconds
        label = next_label

    filter_complex = ";".join(filter_parts)
    out_path = pipeline_dir / out_name
    print(f"{pipeline_dir.name}: {len(segments)} stages, hold {hold_seconds}s each "
          f"(clips trimmed {trim_start_seconds}s off the start), {swipe_seconds}s "
          f"{transition} between, total {running:.2f}s, silent", flush=True)
    subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
                    "-map", f"[{label}]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", str(out_path), "-loglevel", "error"], check=True)
    print(f"  -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pipeline_dir", type=Path)
    ap.add_argument("--out-name", default="instagram_edit.mp4")
    ap.add_argument("--hold-seconds", type=float, default=1.4, help="How long a still image holds")
    ap.add_argument("--swipe-seconds", type=float, default=0.35, help="Duration of each swipe transition")
    ap.add_argument("--direction", choices=["left", "right"], default="left",
                    help="Swipe direction (left = next post slides in from the right)")
    ap.add_argument("--transition", default=None,
                    help="Any ffmpeg xfade transition name, overriding --direction "
                         "(e.g. coverleft, revealright, diagtl/tr/bl/br for a corner-peel look)")
    ap.add_argument("--trim-start-seconds", type=float, default=0.05,
                    help="Seconds skipped off the front of each video clip (not stills)")
    ap.add_argument("--stills", action="store_true", help="Use stage_N.png stills even where i2v clips exist")
    ap.add_argument("--order", type=int, nargs="+", default=None,
                    help="Stage numbers in play order, e.g. --order 3 2 1 4. Repeats/drops allowed.")
    a = ap.parse_args()
    make_instagram_swipe(a.pipeline_dir, a.out_name, a.hold_seconds, a.swipe_seconds,
                         a.direction, a.transition, a.trim_start_seconds, a.stills, a.order)


if __name__ == "__main__":
    main()
