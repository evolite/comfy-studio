#!/usr/bin/env python3
"""
SeedVR2 upscale/crisp-up of existing images on Comfy Cloud (paid, one job per
image x variant). For A/B-ing the finish step before wiring it into an engine.

    python3 pipeline/seedvr_one.py data/pipelines/<name>/stage_2.png
    python3 pipeline/seedvr_one.py a.png b.png --variants 3b 7b --resolution 1680

--resolution is the target short edge (SeedVR2 `resolution`); default 2x the input's.
Variants: 3b = seedvr2_ema_3b_fp16, 3b8 = seedvr2_ema_3b_fp8_e4m3fn, 7b = seedvr2_ema_7b_sharp_fp16.
Output: data/oneoffs/<timestamp>_seedvr/<image>_<variant>.png
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comfy_api
import graph_settings

MODELS = {"3b": "seedvr2_ema_3b_fp16.safetensors", "3b8": "seedvr2_ema_3b_fp8_e4m3fn.safetensors", "7b": "seedvr2_ema_7b_sharp_fp16.safetensors"}


def graph(image_name: str, model: str, resolution: int, color: str) -> dict:
    return {"1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            **graph_settings.seedvr_nodes(["1", 0], resolution, model, color),
            "5": {"class_type": "SaveImage", "inputs": {"images": ["seedvr:up", 0], "filename_prefix": "seedvr"}}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--variants", nargs="+", choices=list(MODELS), default=["3b"])
    ap.add_argument("--resolution", type=int, help="target short edge (default: 2x input)")
    ap.add_argument("--color", default="lab", choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"])
    args = ap.parse_args()

    from PIL import Image
    out = comfy_api.config.ONEOFFS_DIR / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_seedvr"
    out.mkdir(parents=True)
    for img in args.images:
        name = comfy_api.upload_image(img)
        res = args.resolution or 2 * min(Image.open(img).size)
        for v in args.variants:
            pid, _ = comfy_api.submit(graph(name, MODELS[v], res, args.color))
            print(f"[{img.name} {v} -> {res}px] job {pid}", flush=True)
            status = comfy_api.poll_status(pid, timeout=1800)
            job = comfy_api.fetch_job(pid)
            if status.get("status") != "success" and status.get("state") != "success":
                print(f"  failed: {status}"); continue
            comfy_api.save_outputs(job, out / "_debug", final_node="5")
            if (out / "_debug" / "final.png").exists():
                (out / "_debug" / "final.png").rename(out / f"{img.stem}_{v}.png")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
