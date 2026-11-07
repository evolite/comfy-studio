#!/usr/bin/env python3
"""
One-off: animate a single image (i2v) into oneoffs/<timestamp>[_<name>]/clip.mp4.

Usage:
  python3 video/animate_one.py image.png "motion prompt" [--duration 2.9] \
      [--lora NAME=STRENGTH ...] [--seed N] [--engine minimax|ltx] [--name NAME] [--dry-run]

Durations: tiers from AGENTS.md (1.0, 2.0, 2.9, 5.0) are tested; other values are passed straight through. LoRA names as in loras.json ("video").
--dry-run builds and saves graph.json without uploading or spending credits.
"""
import argparse
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
import comfy_api as api
import animate_pipeline as ap_



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("motion", help="i2v motion prompt")
    ap.add_argument("--duration", type=float, default=2.9, help="seconds; tiers 1.0/2.0/2.9/5.0 are tested, longer is not")
    ap.add_argument("--lora", action="append", default=[], metavar="NAME=STRENGTH")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--engine", choices=list(ap_.VIDEO_ENGINES), default=ap_.DEFAULT_ENGINE)
    ap.add_argument("--template", type=Path, default=None, help="base graph instead of the engine's template")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.image.is_file():
        sys.exit(f"no such image: {args.image}")
    loras = {}
    for item in args.lora:
        name, _, val = item.partition("=")
        try:
            loras[name] = float(val)
        except ValueError:
            sys.exit(f"--lora expects NAME=STRENGTH, got {item!r}")
    ap_.check_loras(loras, "--lora", args.engine)
    seed = args.seed if args.seed is not None else secrets.randbits(32)

    image_name = args.image.name if args.dry_run else api.upload_image(args.image)
    template = json.loads(args.template.read_text()) if args.template else ap_.template_for(args.engine)
    graph = ap_.build_graph(template, image_name, args.motion, args.duration, loras, seed, engine=args.engine)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = api.config.ONEOFFS_DIR / (f"{ts}_{args.name}" if args.name else ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "graph.json").write_text(json.dumps(graph, indent=2))
    (run_dir / "render_settings.json").write_text(json.dumps(
        {"image": str(args.image), "engine": args.engine, "motion": args.motion, "duration": args.duration, "loras": loras, "seed": seed}, indent=2))
    print(f"loras={loras or 'none'} seed={seed}\ngraph: {run_dir / 'graph.json'}")
    if args.dry_run:
        print("Dry run — not calling the API.")
        return

    job = ap_.submit_and_wait(graph, "oneoff")
    saved = api.save_outputs(job, run_dir / "_debug")
    videos = [d for _, d in saved if d.suffix == ".mp4"]
    if not videos:
        sys.exit("no .mp4 output found among saved outputs")
    clip = run_dir / "clip.mp4"
    clip.write_bytes(videos[0].read_bytes())
    print(f"-> {clip}")


if __name__ == "__main__":
    main()
