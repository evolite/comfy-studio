"""This machine's local-ComfyUI model names (defaults/local_models.json, overridable per key in data/library/local_models.json).

Kept as data: renaming a file or adding a local-only UNet is an edit to the JSON, not to code. The cloud target reads the
same list to refuse a local-only UNet, which is why it lives apart from both targets.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: config.py
import config  # noqa: E402


def local_models() -> dict:
    return config.library_merged("local_models.json")


def local_lora(file: str, krea: bool) -> str:
    """A bare cloud LoRA name -> its path in this machine's loras folder (Krea2 ones in `krea_lora_folder`, Illustrious ones in
    `il_lora_folder`); a name that already has a folder is kept."""
    if "/" in file:
        return file
    m = local_models()
    return f"{m['krea_lora_folder' if krea else 'il_lora_folder']}{file}"
