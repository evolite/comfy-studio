#!/usr/bin/env python3
"""
Silent, continuously-scrolling edit from a pipeline's i2v clips (pipeline_dir/videos/stage_N.mp4)
or its stage_N.png stills (--stills forces stills) -- mimics scrolling straight down through
someone's feed rather than swiping between discrete posts (see make_instagram_swipe.py for the
swipe version). No app chrome, no audio track.

Mechanism: each stage is cover-cropped to WIDTHxHEIGHT, all of them are stacked into one tall
WIDTHx(HEIGHT*N) strip VIDEO, then ffmpeg's `scroll` filter (see
https://salivity.github.io/ffmpeg/article/f-fmpeg-scroll-filter-how-to-continuously-scroll-video)
shifts that strip vertically frame by frame at a constant speed while a fixed WIDTHxHEIGHT window
crops the visible viewport -- so the whole feed glides past continuously, no cuts. Speed is
derived from --seconds-per-stage so each stage gets roughly that long in view.

A stage's clip starts playing once it's VISIBLE_FRACTION_TO_START_PLAYBACK scrolled into view
(mid-scroll-in, not only once fully settled), then plays forward-then-reverse on a seamless
ping-pong loop for as long as that stage stays in view, rather than freezing once its own short
footage runs out.

Usage:
    python3 video/make_instagram_scroll.py data/pipelines/<run> [--seconds-per-stage 1.4] [--direction down|up] [--stills]

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
from make_edit import stage_sources, WIDTH, HEIGHT, FPS  # noqa: E402

VISIBLE_FRACTION_TO_START_PLAYBACK = 0.4  # a stage's clip starts playing once it's this far scrolled into view


FRAMES_ARG = "-frames:v"  # ffmpeg input/output option; constant because SonarQube flags the repeated literal


def build_layer(source, out_path, total_frames, delay_seconds):
    """Cover-crop one stage to WIDTHxHEIGHT and pad/hold it to exactly total_frames long.
    A still is just held throughout. A clip is held on its first frame until delay_seconds,
    then plays forward-then-reverse on a loop (a seamless ping-pong -- the reversed pass ends
    back on the same first frame the forward pass started on) for however long it stays in
    view, rather than freezing once its own short footage runs out."""
    norm_filters = [
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={WIDTH}:{HEIGHT}",
        "setsar=1",
        f"fps={FPS}",
    ]
    if source.suffix == ".png":
        subprocess.run(["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(source),
                        "-vf", ",".join(norm_filters), FRAMES_ARG, str(total_frames),
                        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                        str(out_path), "-loglevel", "error"], check=True)
        return

    norm_path = out_path.with_name(out_path.stem + "_norm.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", str(source), "-vf", ",".join(norm_filters),
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    str(norm_path), "-loglevel", "error"], check=True)

    pingpong_path = out_path.with_name(out_path.stem + "_pp.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", str(norm_path), "-filter_complex",
                    "[0]split[a][b];[b]reverse[br];[a][br]concat=n=2:v=1:a=0[pp]",
                    "-map", "[pp]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", str(pingpong_path), "-loglevel", "error"], check=True)

    subprocess.run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(pingpong_path),
                    "-vf", f"tpad=start_duration={delay_seconds:.4f}:start_mode=clone",
                    FRAMES_ARG, str(total_frames), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", str(out_path), "-loglevel", "error"], check=True)


def make_instagram_scroll(pipeline_dir, out_name="instagram_scroll_edit.mp4",
                          seconds_per_stage=1.4, direction="down", stills=False, order=None):
    sources = stage_sources(pipeline_dir, stills, order)
    if len(sources) < 2:
        raise FileNotFoundError(f"Need at least 2 stages (clips or stills) in {pipeline_dir}")

    build_dir = pipeline_dir / "_debug" / "instagram_scroll_build"
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True)

    n = len(sources)
    total_duration = seconds_per_stage * n
    total_frames = max(1, round(total_duration * FPS))
    # Time for the viewport to scroll one full stage-height (n-1 gaps between n stage
    # positions spread across the whole timeline -- same basis the scroll speed itself uses).
    slot_time = total_duration / max(n - 1, 1)

    layer_paths = []
    for i, src in enumerate(sources):
        layer_path = build_dir / f"layer_{i:03d}.mp4"
        # Slot i is fully settled (100% visible) at t = arrival_index * slot_time when
        # scrolling down (slot 0 first); scrolling up visits slots in the opposite order.
        # Start playback VISIBLE_FRACTION_TO_START_PLAYBACK of a slot-height earlier, i.e.
        # while it's still scrolling into view rather than waiting for it to fully settle.
        arrival_index = i if direction == "down" else (n - 1 - i)
        delay_seconds = max(0.0, (arrival_index - (1 - VISIBLE_FRACTION_TO_START_PLAYBACK)) * slot_time)
        build_layer(src, layer_path, total_frames, delay_seconds)
        layer_paths.append(layer_path)

    # v is a fraction of the FULL STRIP's height moved per frame (scroll wraps the whole
    # loaded frame, not just the viewport). We need to move (n-1)/n of the strip's height
    # over total_frames frames to go from showing stage 1 at top to the last stage filling
    # the viewport.
    speed = ((n - 1) / n) / total_frames
    if direction == "up":
        speed = -speed

    # vstack+scroll+crop all happen in one filter graph so the WIDTHx(HEIGHT*N) intermediate
    # is never itself encoded -- libx264 rejects a frame that tall as an output size, but the
    # filter graph has no such limit; only the final cropped-down WIDTHxHEIGHT frame gets encoded.
    stack_inputs = []
    for p in layer_paths:
        stack_inputs += ["-i", str(p)]
    filter_complex = (
        "".join(f"[{i}:v]" for i in range(n)) + f"vstack=inputs={n}[stack];"
        f"[stack]scroll=v={speed:.8f}:vpos=0[scrolled];"
        f"[scrolled]crop={WIDTH}:{HEIGHT}:0:0[out]"
    )

    out_path = pipeline_dir / out_name
    kind = "stills" if stills or all(p.suffix == ".png" for p in sources) else "clips"
    print(f"{pipeline_dir.name}: {n} stages ({kind}) stacked into a {WIDTH}x{HEIGHT * n} strip, "
          f"scrolling {direction} over {total_duration:.2f}s ({total_frames} frames), "
          f"~{seconds_per_stage}s/stage, clips start {VISIBLE_FRACTION_TO_START_PLAYBACK * 100:.0f}% "
          f"scrolled in, silent", flush=True)
    subprocess.run([
        "ffmpeg", "-y", *stack_inputs, "-filter_complex", filter_complex, "-map", "[out]",
        FRAMES_ARG, str(total_frames), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", str(out_path), "-loglevel", "error",
    ], check=True)
    print(f"  -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pipeline_dir", type=Path)
    ap.add_argument("--out-name", default="instagram_scroll_edit.mp4")
    ap.add_argument("--seconds-per-stage", type=float, default=1.4,
                    help="Roughly how long each stage stays in view while scrolling past")
    ap.add_argument("--direction", choices=["down", "up"], default="down",
                    help="down = feed scrolls upward past the viewport (default, like scrolling down a feed)")
    ap.add_argument("--stills", action="store_true", help="Use stage_N.png stills even where i2v clips exist")
    ap.add_argument("--order", type=int, nargs="+", default=None,
                    help="Stage numbers in play order, e.g. --order 3 2 1 4. Repeats/drops allowed.")
    a = ap.parse_args()
    make_instagram_scroll(a.pipeline_dir, a.out_name, a.seconds_per_stage, a.direction, a.stills, a.order)


if __name__ == "__main__":
    main()
