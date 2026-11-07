# comfy-studio

A local studio for multi-stage AI image sets and short clips, built on [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and
[Comfy Cloud](https://cloud.comfy.org). A project is a list of stages (one shot each); every stage goes through the same
pipeline, and a web UI shows the board:

I built this primarily to get an AI first experience, where basically the AI makes the prompts, the workflows, manages the models, the loras etc
But it has other side effects.

Some warnings:


1. **THIS IS VIBE CODED** (but guided in a slightly sensible way, and i use sonarqube to fix outrageous issues.)
2. **I literally can't tell you how to run this shit or how it works, use an AI CLI agent to set it up.**
3. **I run Arch Linux, some things might not be compatible on your system, again, make claude or something fix it**

It's built around some steps

1. **Preview** — Krea2 (or MOSS-Image) base render, several takes per stage
2. **Upscale** — Illustrious refine with ControlNets, or a latent hires pass; optional face detailer
3. **Finish** — SeedVR2 (optional)
4. **Animate** — image-to-video with MiniMax H3 or LTX 2.3, on Comfy Cloud or your own RunPod pod
5. **Interpolate** — RIFE to 48 fps
6. **Edit** — a frame-exact, beat-synced vertical edit of the clips (or a no-music scenario cut)

Image and video steps run locally (free) or on Comfy Cloud (credits). LoRAs, voices and model choices are libraries you edit
in the web UI.

> **18+ / your responsibility.** The tool doesn't restrict what you generate. You're responsible for using models and LoRAs
> under their licenses and for the content you make. Never depict real people without their consent, and never depict
> minors in sexual content.

## Requirements
- Python 3.14, Node 20+ (for building the web UI), ffmpeg
- A local ComfyUI and/or a Comfy Cloud API key (or a runpod api key). The local target checks the server's `/object_info` before running and
  names any missing node or model.
- Optional: a RunPod account and network volume for the RunPod animate target (`runpod/`)

## Setup
```sh
git clone <this repo> comfy-studio && cd comfy-studio
python -m venv webui/.venv && webui/.venv/bin/pip install -r requirements.txt
(cd webui/ui && npm ci && npm run build)
cp -r data.example data            # your data: libraries, projects, audio (git-ignored)
cp .env.example .env               # add COMFY_CLOUD_KEY and anything else you need
webui/.venv/bin/uvicorn app:app --app-dir webui/server --host 127.0.0.1 --port 8191
```
Open http://127.0.0.1:8191. `contrib/systemd/` has user units for the web UI, the status dashboard and a nightly idle
restart of ComfyUI. Adjust their paths to where you cloned the repo.

> **Keep the web UI on 127.0.0.1.** It has no login, it can start paid cloud and RunPod jobs, and it reads and writes
> files in your data folder. Don't expose it to a network or forward its port.

## Your data and the defaults
`defaults/` holds neutral starter libraries. Anything you change goes to `data/library/`: the first save from the web UI
writes your copy, and previous versions are kept in `data/library/_bak/`.

| File | What it is |
|---|---|
| `loras.json` | image + i2v LoRAs, triggers, strengths, combos ([docs/loras.md](docs/loras.md)) |
| `voices.json` | named voices that fill a motion's `{voice}` slots |
| `models.json` | the model a new project starts with, and the model dropdowns (add your cloud uploads here) |
| `local_models.json` | your local ComfyUI's file names and local-only models (merged per key) |
| `runpod_models.json` | what the RunPod volume holds (`runpod/seed.py`) |

## First project
```sh
python3 project.py 20260101_my_project new --engine krea2
# write the stages into data/pipelines/20260101_my_project/project.json (or use the web UI), then
python3 project.py 20260101_my_project generate --dry-run
python3 project.py 20260101_my_project generate --preview --local
```
`data.example/pipelines/example_project/` is a filled-in two-stage sample. `python3 project.py --help` and
[AGENTS.md](AGENTS.md) describe the whole pipeline.

## Tests
```sh
webui/.venv/bin/python -m pytest
```

## License
MIT, see [LICENSE](LICENSE). Model weights, LoRAs and the ComfyUI custom nodes the templates use have their own licenses.
