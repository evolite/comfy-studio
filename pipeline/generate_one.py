#!/usr/bin/env python3
"""
One-off single-image generator. For when you just want one picture, not a
full multi-stage pipeline.

Prompts are complete, same as a stage in a stage set: --krea-scene holds the
character description written out alongside the scene, --il-scene holds the
character tags alongside the scene tags. Nothing is prepended except the
Illustrious quality prefix and LoRA triggers.

Usage
-----
    python3 pipeline/generate_one.py \\
        --krea-scene "An adult woman with ... She sits ... Photo." \\
        --il-scene "mature female, 1girl, ..., sitting, ..." \\
        --name quick_test

Krea2-only engine (no Illustrious pass, no FaceDetailer — see AGENTS.md):
    python3 pipeline/generate_one.py --engine krea2 --krea-scene "..."

Output: data/oneoffs/<timestamp>[_<name>]/final.png, plus _debug/ with the raw
graph and every intermediate node's image (same convention as run_pipeline.py).
"""
import argparse
import json
import secrets
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comfy_api
import generate as g_full
import generate_krea2 as g_krea2
import graph_settings
import image_loras

ONEOFF_DIR = comfy_api.config.ONEOFFS_DIR


def _parse_loras(items: list) -> dict:
    loras = {}
    for item in items:
        if "=" not in item:
            sys.exit(f"--lora expects NAME=STRENGTH, got: {item!r}")
        name, value = item.split("=", 1)
        loras[name.strip()] = float(value)
    return loras


def _parse_settings(items: list) -> dict:
    settings = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            sys.exit(f"--set expects KEY=VALUE, got: {item!r}")
        try:
            settings[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            settings[key.strip()] = value  # bare string, e.g. --set upscale_model=4x_foo.pth
    return settings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["full", "krea2"], default="full",
                     help="'full' = Krea2+Illustrious two-pass (default). "
                          "'krea2' = Krea2-only, no anime pass, no FaceDetailer.")

    ap.add_argument("--krea-scene", required=True, help="Whole Krea2 prompt: character description + scene, natural language")
    ap.add_argument("--il-scene", default="", help="Whole Illustrious prompt: character + scene Danbooru tags (full engine only)")
    ap.add_argument("--il-negative", default=None, help="Override the default Illustrious negative prompt")
    ap.add_argument("--lora", action="append", default=[], metavar="NAME=STRENGTH",
                     help="Explicitly set a registered LoRA's strength, e.g. --lora background_detail=0.6. "
                          "Repeatable. Names: see loras.json.")
    ap.add_argument("--face-detailer", action="store_true", help="Full engine only: run FaceDetailer (off by default, same as run_pipeline.py)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="Engine tunable (keys: pipeline/graph_settings.py), value parsed as JSON, e.g. "
                          "--set il_denoise=0.2 --set upscale_model=4x_foo.pth. Repeatable.")

    ap.add_argument("--seed", type=int, default=None, help="Fix the seed (default: random)")
    ap.add_argument("--name", default=None, help="Label used in the output folder name")
    ap.add_argument("--dry-run", action="store_true", help="Build the graph and save it, but don't call the API")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    g = g_krea2 if args.engine == "krea2" else g_full

    if args.engine == "full" and not args.il_scene:
        print("Warning: --il-scene is empty; the Illustrious pass will run on tags alone.", file=sys.stderr)

    seed = args.seed if args.seed is not None else secrets.randbits(48)

    loras = _parse_loras(args.lora)
    print(f"LoRAs this run: { {n: s for n, s in image_loras.resolve(loras).items() if s} }")

    settings = _parse_settings(args.set)
    graph_settings.check_settings(args.engine, settings)

    build_kwargs = dict(
        krea_scene=args.krea_scene,
        seed=seed,
        loras=loras or None,
        **settings,
    )
    if args.engine == "full":
        build_kwargs.update(
            il_scene=args.il_scene,
            il_negative=args.il_negative,
            face_detailer_enabled=args.face_detailer,
        )

    graph = g.build_graph(**build_kwargs)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = f"{ts}_{args.name}" if args.name else ts
    run_dir = ONEOFF_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "graph.json").write_text(json.dumps(graph, indent=2))
    print(f"engine={args.engine} seed={seed}")
    print(f"Patched graph saved: {run_dir / 'graph.json'}")

    if args.dry_run:
        print("Dry run — not calling the API.")
        return

    print("Submitting to Comfy Cloud...", flush=True)
    prompt_id, submit_resp = comfy_api.submit(graph)
    print(f"prompt_id: {prompt_id}", flush=True)
    (run_dir / "submit_response.json").write_text(json.dumps(submit_resp, indent=2))

    print("Polling...", flush=True)
    status = comfy_api.poll_status(prompt_id, interval=5, timeout=args.timeout)
    (run_dir / "final_status.json").write_text(json.dumps(status, indent=2))
    state = status.get("status") or status.get("state")

    if state != "success":
        print(f"Job did not succeed (status: {state}). See {run_dir}/final_status.json")
        sys.exit(1)

    job = comfy_api.fetch_job(prompt_id)
    (run_dir / "job_full.json").write_text(json.dumps(job, indent=2))
    debug_dir = run_dir / "_debug"
    comfy_api.save_outputs(job, debug_dir, g.NODE_LABELS, g.FINAL_IMAGE_NODE)

    src = debug_dir / "final.png"
    if src.exists():
        dest = run_dir / "final.png"
        shutil.copyfile(src, dest)
        print(f"\n-> {dest}")
    else:
        print(f"Succeeded but no final.png found — check {debug_dir}")


if __name__ == "__main__":
    main()
