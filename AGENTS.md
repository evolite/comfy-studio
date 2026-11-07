# AGENTS.md — comfy-studio

Python layer over Comfy Cloud (`https://cloud.comfy.org`) and a local ComfyUI: multi-stage image sets (Krea2 / Illustrious /
MOSS-Image) → short i2v clips (MiniMax H3, LTX 2.3) → beat-synced vertical edits, driven by a web UI. This file is the map;
usage details live in `project.py --help` and each script's docstring. Personal rules belong in a git-ignored
`CLAUDE.local.md` (e.g. one that imports `@data/private/AGENTS.local.md`).

## Rules
- **Cloud jobs spend credits** (images, i2v, re-rolls) and the RunPod target bills per hour; `--dry-run`, local runs and
  ffmpeg edits are free.
- **Nothing personal in tracked files.** Libraries, model choices, prompt guides, stage sets, audio and output live in `data/`
  (git-ignored, `STUDIO_DATA_DIR`); the repo ships neutral starters in `defaults/` and a skeleton in `data.example/`.
- Paths, URLs and machine settings live in `config.py` (env / `.env` overrides; key `COMFY_CLOUD_KEY`) — never hardcode them.
- Don't restart `comfy-studio` or the local ComfyUI while jobs run (check the web UI's `/api/jobs` / ComfyUI `/queue` first).

## Layout (run from the repo root)
```
project.py     per-project CLI (the commands below)
config.py      every path/URL/machine setting; library_file / library_merged / save_library
pipeline/      image steps: steps.py (preview, il, face, hires, finish, main) run on a target (targets/cloud.py, targets/local.py,
               same interface) through runner.py; engines generate*.py + templates/; graph_settings.py (settings keys);
               takes.py (preview takes); civitai_meta.py (Civitai-ready copies); lora_registry.py, voice_registry.py
video/         animate_pipeline.py (i2v), interpolate.py (RIFE), make_tiktok.py (edit), templates/
runpod/        the RunPod animate pod's volume (seed.py fills it from the manifest), preflight (check.py), start command
               (watchdog.sh); lifecycle in pipeline/runpod_pod.py
webui/         web UI + API (FastAPI + React, `webui/ui`, `npm run build`), service `comfy-studio` on :8191 (127.0.0.1, no auth)
dashboard.py   status page for local ComfyUI + cloud (:8190)
restart_comfyui.py  nightly idle restart of the local ComfyUI (skips while jobs run)
defaults/      neutral starter libraries: loras.json, voices.json, models.json, local_models.json, runpod_models.json
data.example/  the data/ skeleton (cp -r data.example data)
docs/          loras.md (chaining, combos, adding one)
contrib/       systemd units, KDE Plasma widgets for dashboard.py
tests/         pytest: `webui/.venv/bin/python -m pytest` (fixed test library in tests/fixtures/library)
data/          git-ignored: library/ (your copies of the defaults/ files; _bak/ = previous versions), pipelines/<name>/,
               oneoffs/, audio/, logs/, civitai/, stage_sets/, app_settings.json
```

## Libraries
A file in `data/library/` replaces its `defaults/` starter (`config.library_file`); `models.json` and `local_models.json`
are merged per top-level key instead (`config.library_merged`). Web UI edits (LoRAs, voices pages) always write the
`data/library` copy through `config.save_library`, which keeps the previous versions in `data/library/_bak/`.
- `loras.json`: image + i2v LoRAs and combos (`docs/loras.md`; catalog: `python3 pipeline/lora_registry.py`).
- `voices.json`: named voices filling a motion's `{voice...}` slots (`pipeline/voice_registry.py`; web UI `#/voices`).
- `models.json`: `new_project` (what `new` writes into settings), `moss_unet`, `choices` (the web UI's model dropdowns).
- `local_models.json`: your local ComfyUI's file names and the local-only UNets/VAEs (cloud runs refuse them).
- `runpod_models.json`: what the RunPod volume holds (`runpod/seed.py`).

## New project
1. `python3 project.py <date_name> new [--engine full|krea2|bgswap|moss] [--oneoff]` (e.g. `20261026_rainy_night_visit`).
2. Write `krea_scene`, `motion`, `duration` and LoRAs per stage into `project.json`; check `generate --dry-run` builds.
3. Run the preview, upscale next, `animate` when the stills are right.

## Local vs cloud
- **Local is free**: `--preview --local`, then `--local` (IL) or `--hires --local`. Cloud `--preview` is the cheap cloud step;
  cloud IL/upscale/finish is where credits go.
- Some models are local only: LoRAs with `cloud: false` and UNets/VAEs in `local_only_*` (`local_models.json`). A project
  using them runs locally; cloud runs refuse them.

## project.json
`{"engine", "seed"?, "video_engine"?, "voice"?, "settings": {...}, "stages": [...]}`. `settings` = run-wide tunables (keys in
`pipeline/graph_settings.py`, unknown keys error); a stage can override any of them; a key absent everywhere takes the
template's value. Each stage: stable `id` (names its `stage_N` files; gaps fine, order = array order), `krea_scene`,
`il_scene`, `loras`, `motion`, `duration`, `video_loras`, optional `seed`/`video_seed`/`video_engine`/`voice`. LoRA
triggers are appended automatically; a `voice` (stage, else project) fills the motion's `{voice}` slots; `voices` (`{"S2": name}`) gives another
speaker its own for its `:S2` slots. A stage is
self-contained: describe the character in every `krea_scene`. Past runs' `_debug/*/graph.json` hold the exact prompts,
seeds and settings used.

## Pipeline (`project.py <name> …`)
| Step | Command | Output |
|---|---|---|
| ① Preview | `generate [N..] --preview [--local] [--takes K]` | `preview/` (every render kept as a take) |
| ② Upscale | `generate [N..] --local` (IL) / `--il` (cloud) / `--hires --local` (SesquiLSR) | `IL/`, `hires/` |
| (face) | `generate N.. --face [--local]` | replaces `IL/stage_N.png` |
| ③ Finish | `generate [N..] --finish [--local] [--from hires\|IL]` (SeedVR2, optional) | `finish/` |
| ④ Animate | `animate [folder] [N..] [--rife] [--target cloud\|runpod]` (paid; `--rife` = Animate + RIFE in the same job) | `<folder>/videos/` (+ `interp/videos/` with `--rife`) |
| ⑤ Interpolate | `interpolate [folder] [N..]` (RIFE ×2 → 48 fps, local) | `interp/videos/` |
| ⑥ Edit | `edit [folder] [animate\|stills] [N..] [make_tiktok args]` | `<folder>/edits/` |
| ⑦ Civitai | web UI Civitai tab: best still (with Civitai metadata) + clip per picked stage | `data/civitai/<project>_<time>/` |

No N = every stage missing that output; with N = redo those (the old file moves to `_debug/`). The web UI board has one phase
per step. `status` prints a per-stage summary; `show [N..]` prints everything about those stages as JSON (what an agent reads
for a web-UI ref `@<project>#<id>`).

## Animate + edit
- **i2v model** = `video_engine` (project, a stage may override): `minimax` (MiniMax H3, default) or `ltx` (LTX 2.3 /
  Sulphur 2 checkpoint, uploaded to the cloud). Each has its own template, frame grid and LoRAs (`VIDEO_ENGINES` in
  `video/animate_pipeline.py`; adding a model = a template + one entry there + `lora_registry.VIDEO_ENGINES`).
- **Animate target** (⚙ page switch, default `cloud`; `--target`): Comfy Cloud, or your RunPod pod (H3 only), which `animate`
  starts on demand and which **bills per hour while up** and deletes itself after `RUNPOD_IDLE_MINUTES` idle. Pod settings:
  `RUNPOD_*` in `config.py`. "Start batch" (runpod target only) queues Animate clicks server-side until the ⚙ batch size is
  reached, so a batch shares one pod start (`Manager.animate_batch` in `webui/server/jobs.py`).
- **`animate` spends credits for every missing clip it finds** — never use it as a dry check, and never start one while another
  is in flight. `duration` is free-form (1–8 s, snapped to the 24 fps frame grid).
- **No music** (`music.none` in project.json): `edit` becomes a scenario cut (`--no-music`): every stage in full, joined by an
  xfade `transition`, over the clips' own audio.
- `edit` (`video/make_tiktok.py`) is **frame-exact** (cumulative integer frame counts). After changing it, verify the total frame
  count and PSNR-compare against a previous edit.

## Comfy Cloud API quirks
- Header `X-API-Key`; submit `POST /api/prompt` `{"prompt": graph}`; `GET /api/job/{id}/status` = state, `GET /api/jobs/{id}` =
  outputs; download `GET /api/view?filename=…&subfolder=…&type=output`; upload `POST /api/upload/image` (field `image`).
- User-uploaded models aren't in `object_info` combos but work; add them to `choices` in your `models.json`.

## Node-ID fragility
Templates are hand-traced API graphs and `NODE_*` / `SEED_NODES` constants hardcode their node ids. A ComfyUI re-export can
shift ids and patches then land nowhere silently — re-derive by `class_type` / `_meta.title` / inputs.
