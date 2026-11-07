#!/usr/bin/env python3
"""
Generic pipeline runner: give it a stage set (JSON file), it submits everything as one batch, waits, and saves
the 9 (or however many) final images flat into one pipeline folder.

TikTok-style edits are a separate step now -- see video/make_tiktok.py (run it against
the pipeline folder once the images/videos you want are ready).

STAGES FILE (JSON): a list of stage objects, each:
  {"id": 1, "krea_scene": "natural language...", "il_scene": "tag, tag, ..."}
  - krea_scene: the WHOLE Krea2 prompt. Write the character out in every
    stage (hair, build, accessories...), tailored to the shot -- e.g. leave
    face/body detail out of a from-behind or covered stage, since Krea2
    steers toward whatever the prompt describes. Nothing is prepended.
  - il_scene: the WHOLE Illustrious tag list, character tags included
    (verified against the Danbooru tag list!). Only the quality prefix is prepended.
  - face_wildcard (optional, with "face_detailer_enabled": true): FaceDetailer's
    per-face description, e.g. "green eyes, freckles, smirk".
  Stage order in the list = image numbering (stage_1.png, stage_2.png, ...). `id` is optional
  here; if present it must equal the stage's position (1..N).
  Any other field (seed, loras, il_denoise, il_background, ...) overrides the
  run-wide setting of the same name for that stage.
  Run-wide tunables (upscale_factor, il_steps, upscale_model, ... -- keys in pipeline/graph_settings.py) go in
  an optional top-level object: {"settings": {...}, "stages": [...]}.

Usage:
    python3 pipeline/run_pipeline.py \\
        --pipeline-name my_new_concept \\
        --stages data/stage_sets/my_new_concept_stages.json \\
        [--seed 12345] [--dry-run]

Output: pipelines/<timestamp>_<pipeline-name>/
        stage_1.png .. stage_N.png, _debug/ (raw API data)
"""
import argparse
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comfy_api
import generate
import graph_settings
import generate_bgswap
import generate_moss
import generate_krea2

ENGINES = {"full": generate, "krea2": generate_krea2, "bgswap": generate_bgswap, "moss": generate_moss}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=ENGINES, default="full",
                    help="full = Krea2 + Illustrious; krea2 = photoreal only; bgswap = photoreal her in an anime room; moss = MOSS-Image base render only (local)")
    ap.add_argument("--pipeline-name", required=True, help="Short label, used in the output folder name")
    ap.add_argument("--stages", required=True, type=Path, help="Stages JSON (list)")
    ap.add_argument("--seed", type=int, default=None, help="Seed for every stage without its own (default: one random seed)")
    ap.add_argument("--dry-run", action="store_true", help="Build every graph, don't call the API")
    args = ap.parse_args()

    engine = ENGINES[args.engine]
    data = json.loads(args.stages.read_text())
    settings, stages = (data.get("settings", {}), data.get("stages")) if isinstance(data, dict) else ({}, data)
    if not isinstance(stages, list) or not stages:
        sys.exit(f"{args.stages} must contain a non-empty JSON list of stage objects (or {{\"settings\": {{...}}, \"stages\": [...]}})")
    graph_settings.check_settings(args.engine, settings)
    comfy_api.check_stage_ids(stages)

    seed = args.seed if args.seed is not None else secrets.randbits(48)
    common = {"seed": seed, **settings}
    print(f"Pipeline: {args.pipeline_name} | {args.engine} | {len(stages)} stages | seed={seed}", flush=True)

    if args.dry_run:
        for i, stage in enumerate(stages, 1):
            graph = comfy_api.stage_graph(engine, stage, common)
            print(f"  stage_{i}: graph built OK, {len(graph)} nodes")
        print("Dry run — not calling the API.")
        return

    jobs = comfy_api.run_batch(engine, stages, common, pipeline_name=args.pipeline_name)["jobs"]
    print(f"\n=== All {len(stages)} stages complete ===", flush=True)
    for j in jobs:
        print(f"{j['name']}: {j['final_path'] or '(no final.png — state=' + j['state'] + ')'}", flush=True)
    succeeded = sum(1 for j in jobs if j["final_path"])
    if succeeded < len(stages):
        print(f"\n{succeeded}/{len(stages)} stages succeeded.", flush=True)


if __name__ == "__main__":
    main()
