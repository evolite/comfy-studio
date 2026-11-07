#!/usr/bin/env python3
"""Can a ComfyUI server (a RunPod pod, or the local test container) run every animate graph? Builds the H3 graph (with and
without audio, with RIFE; LTX is not on the volume) and once per H3 LoRA in loras.json, and lists every node class or model file
the server's /object_info lacks. Exit 1 when anything is missing.

  webui/.venv/bin/python runpod/check.py http://127.0.0.1:8288
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "video"), str(ROOT / "pipeline")]
import animate_pipeline as ap  # noqa: E402
import lora_registry  # noqa: E402
from targets import local  # noqa: E402

DUMMY_IMAGE = "x.png"  # the graph only needs a filename; the file is never read


def graphs():
    yield "minimax", ap.build_graph(ap.template_for("minimax"), DUMMY_IMAGE, "x", 4, rife=True)
    yield "minimax+audio", ap.build_graph(ap.template_for("minimax", True), DUMMY_IMAGE, "x", 4, rife=True, audio=True)
    for name, engine in lora_registry.video_engines().items():
        if engine == "minimax":
            yield f"minimax + {name}", ap.build_graph(ap.template_for(engine), DUMMY_IMAGE, "x", 4, {name: 1.0}, engine=engine)


def main():
    server = sys.argv[1].rstrip("/")
    oi = json.loads(local._call(server, "/object_info"))
    missing = {}
    for label, g in graphs():
        for p in local.check(g, server, ap.CHECK_FIELDS, oi):
            missing.setdefault(p.split(": ", 1)[1], []).append(label)  # drop the node id: the same file shows up in many graphs
    for problem, labels in missing.items():
        print(f"MISSING {problem}  (in {', '.join(labels[:3])}{' …' if len(labels) > 3 else ''})")
    print("all animate graphs check out" if not missing else f"{len(missing)} problem(s)")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
