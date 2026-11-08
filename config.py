"""Single source of paths, endpoints and machine-specific settings; every other module imports them from here.

Code lives in the repo; everything the app produces or consumes lives in DATA_DIR (default `<repo>/data`, override with
STUDIO_DATA_DIR, which is also how the tests run against a throwaway tree). Settings come from the environment or `.env`.
Libraries (LoRAs, voices, model choices, the RunPod manifest) are read from DATA_DIR/library/<name> when it exists, else from
the neutral starters in defaults/ (library_file); web UI edits always land in DATA_DIR/library (save_library, with backups).
"""
import json
import os
import shlex
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("STUDIO_DATA_DIR") or ROOT / "data").expanduser().resolve()
PIPELINES_DIR = DATA_DIR / "pipelines"  # one folder per project
ONEOFFS_DIR = DATA_DIR / "oneoffs"      # single-image / single-clip runs
LOGS_DIR = DATA_DIR / "logs"            # timing baselines, edit timing
AUDIO_DIR = DATA_DIR / "audio"          # music + beat files (<Song>.json beside <Song>.mp3)
POLAROID_SFX_DIR = AUDIO_DIR / "sfx" / "polaroid"  # the user's camera sounds (shutter, flash, eject...): an edit's card sounds pick from them
CIVITAI_DIR = Path(os.environ.get("CIVITAI_DIR") or DATA_DIR / "civitai").expanduser()  # Civitai-tagged copies of stills, to drag onto Civitai

STAGE_SETS_DIR = DATA_DIR / "stage_sets"  # importable stage sets (source material, private)
DEFAULTS_DIR = ROOT / "defaults"          # neutral starter libraries shipped with the code
LIBRARY_DIR = DATA_DIR / "library"        # your own libraries; a file here replaces its defaults/ starter
LIBRARY_BACKUPS = 20                      # previous versions kept in LIBRARY_DIR/_bak per file (data/ has no git history)
LORAS_FILE = LIBRARY_DIR / "loras.json"   # the LoRA library (image + i2v), edited by hand or from the web UI's LoRAs page
VOICES_FILE = LIBRARY_DIR / "voices.json"  # the voice library: named voices a stage's motion fills its {voice...} slots from
# the project holding one sample clip per voice: its first stage is the template (still + motion with {voice} slots + the line)
VOICE_SAMPLES_PROJECT = os.environ.get("VOICE_SAMPLES_PROJECT", "voice_samples")

LOCAL_COMFY_URL = os.environ.get("LOCAL_COMFY_URL", "http://127.0.0.1:8188")
STUDIO_URL = os.environ.get("STUDIO_URL", "http://127.0.0.1:8191")  # the comfy-studio web UI/API (its job list)
# a local batch that reuses the loaded models still starts with a full free when memory is tight (systemd-oomd kills ComfyUI at 90% swap)
LOCAL_FREE_MIN_RAM_GB = float(os.environ.get("LOCAL_FREE_MIN_RAM_GB", "2"))   # server's free RAM below this
LOCAL_FREE_MAX_SWAP_PCT = float(os.environ.get("LOCAL_FREE_MAX_SWAP_PCT", "40"))  # or swap use above this (only measured when the server is this machine)
COMFYUI_SERVICE =os.environ.get("COMFYUI_SERVICE", "comfyui.service")  # systemd --user unit of the local ComfyUI
COMFYUI_MODELS_DIR = Path(os.environ.get("COMFYUI_MODELS_DIR", "~/comfy/ComfyUI/models")).expanduser()  # the local ComfyUI's models/ (hashed for Civitai metadata)
FILE_MANAGER = shlex.split(os.environ.get("FILE_MANAGER", "xdg-open"))  # "reveal in folder": command + the path is appended
PUBLISH_DIR = Path(os.environ["PUBLISH_DIR"]).expanduser() if os.environ.get("PUBLISH_DIR") else None  # finished edits are copied here; unset = off
PUSHOVER_USER = os.environ.get("PUSHOVER_USER", "")  # the web UI's 🔔 pushes here when the job queue empties; unset = off
_token_file = ROOT / ".pushover_token"
PUSHOVER_TOKEN = os.environ.get("PUSHOVER_TOKEN") or (_token_file.read_text().strip() if _token_file.is_file() else "")
DEFAULT_BEATS = os.environ.get("DEFAULT_BEATS", "audio/default.json")  # the beat file an edit uses when a project picks none (relative to DATA_DIR)
APP_SETTINGS_FILE = DATA_DIR / "app_settings.json"  # what the web UI's ⚙ page saves (git-ignored with data/); wins over env

# RunPod: `animate` can run on a pod of our own instead of Comfy Cloud (⚙ animate target; pipeline/runpod_pod.py, runpod/)
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID", "")      # the network volume runpod/seed.py filled (models + ComfyUI)
RUNPOD_DATACENTER = os.environ.get("RUNPOD_DATACENTER", "")  # where that volume lives; the pod must be there too
# an H3 UNet the pod graphs load instead of the stock one (Comfy Cloud keeps the stock one): a distilled turbo finetune, so the
# pod drops the stock turbo LoRA and runs 8 steps of euler (it must be in the RunPod manifest); "" = the stock minimax_h3_fl2va
# UNet (+ turbo LoRA) on the pod too. The ⚙ page's "Pod H3 model" wins over this (config.runpod_h3_unet())
RUNPOD_H3_UNET = os.environ.get("RUNPOD_H3_UNET", "")
STOCK_H3_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"  # the template's UNet
# REST gpuTypeIds, tried in this order: the RTX PRO 6000s first (~2.2x a PRO 4500 per clip, ~1.6x its cost per clip), then the
# other Blackwells with >= 32 GB, cheapest first (H3's UNet is 21 GB; NVFP4 needs Blackwell), without the $6-8/hr B200/B300.
# Comma-separated to override.
RUNPOD_GPUS = [g.strip() for g in os.environ.get("RUNPOD_GPUS", ",".join([
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",  # not the Workstation Edition: ~1130 s/clip vs ~100 (pod fqpfmm5vu5ptf8)
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "NVIDIA RTX PRO 4500 Blackwell", "NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 5000 Blackwell"])).split(",") if g.strip()]
RUNPOD_GPU_WAIT_MINUTES = float(os.environ.get("RUNPOD_GPU_WAIT_MINUTES", "30"))  # none of RUNPOD_GPUS free: keep asking this long
RUNPOD_IDLE_MINUTES = float(os.environ.get("RUNPOD_IDLE_MINUTES", "2"))  # the pod deletes itself after this long with nothing to do
# (counted once ComfyUI is up: the boot gets its own limit, runpod_pod.READY_TIMEOUT); a batch queues every clip up front, so 2 min
# only ends a pod between separate Animate clicks, and the next one pays a cold start (~1-3 min boot + ~1.5 min model load)


def library_file(name: str) -> Path:
    """A library file: yours in LIBRARY_DIR when it exists, else the neutral starter in defaults/."""
    mine = LIBRARY_DIR / name
    return mine if mine.is_file() else DEFAULTS_DIR / name


def library_merged(name: str) -> dict:
    """A keyed library file: the defaults/ starter with your LIBRARY_DIR copy's top-level keys laid over it."""
    data = json.loads((DEFAULTS_DIR / name).read_text())
    mine = LIBRARY_DIR / name
    if mine.is_file():
        data.update(json.loads(mine.read_text()))
    return data


def save_library(path: Path, data: dict) -> None:
    """Atomic write of a library file; the version it replaces goes to <its folder>/_bak/<name>.<time> (newest LIBRARY_BACKUPS kept)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        bak = path.parent / "_bak"
        bak.mkdir(exist_ok=True)
        (bak / f"{path.name}.{time.strftime('%Y%m%d-%H%M%S')}").write_bytes(path.read_bytes())
        for old in sorted(bak.glob(f"{path.name}.*"))[:-LIBRARY_BACKUPS]:
            old.unlink()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


RUNPOD_MANIFEST = library_file("runpod_models.json")  # what the RunPod volume holds (runpod/seed.py fills it from this)
RUNPOD_IMAGE = json.loads(RUNPOD_MANIFEST.read_text())["image"]  # the template image the volume was seeded for


def app_settings() -> dict:
    try:
        return json.loads(APP_SETTINGS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def animate_target() -> str:
    """Where `animate` renders: "cloud" (Comfy Cloud, the default) or "runpod" (our pod), the ⚙ page's switch."""
    return app_settings().get("animate_target") or "cloud"


def runpod_h3_unet() -> str:
    """The H3 UNet pod graphs load: the ⚙ page's pick when one is saved ("" = stock), else RUNPOD_H3_UNET."""
    s = app_settings()
    return s["runpod_h3_unet"] if "runpod_h3_unet" in s else RUNPOD_H3_UNET


def runpod_h3_choices() -> list:
    """The finetuned H3 UNets the RunPod volume holds (its manifest's diffusion_models with "h3" in the name, the stock one aside)."""
    files = json.loads(RUNPOD_MANIFEST.read_text()).get("files", {})
    names = [Path(p).name for p in files if p.startswith("models/diffusion_models/")]
    return sorted(n for n in names if "h3" in n.lower() and n != STOCK_H3_UNET)


def animate_batch_size() -> int:
    """How many stages the web UI's RunPod "Start batch" accumulates before firing one animate job (the ⚙ page's value)."""
    try:
        return max(1, int(app_settings().get("animate_batch_size") or 4))
    except (TypeError, ValueError):
        return 4


def pushover() -> tuple:
    """(user key, app token) for the 🔔 pushes: the ⚙ page's values, else PUSHOVER_USER / PUSHOVER_TOKEN; ("", "") = off."""
    s = app_settings()
    return s.get("pushover_user") or PUSHOVER_USER, s.get("pushover_token") or PUSHOVER_TOKEN


def data_path(p) -> Path:
    """A user-supplied path: absolute as is, relative ones (e.g. `audio/Song.json`) resolve inside DATA_DIR."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else DATA_DIR / p


def cloud_key_present() -> bool:
    return bool(os.environ.get("COMFY_CLOUD_KEY"))
