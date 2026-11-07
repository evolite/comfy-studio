#!/usr/bin/env python3
"""Fill a RunPod `/workspace` with what animate needs, on top of the official ComfyUI template (the manifest's `image`).
The manifest is config.RUNPOD_MANIFEST (data/library/runpod_models.json, else defaults/runpod_models.json), copied next to this
script as models.json.

The template installs ComfyUI and its venv into `comfyui_dir` on its first boot. This moves that ComfyUI to `comfyui.ref`, adds
our custom nodes (pinned, their requirements installed into that venv), the `packages` built from source into that venv (e.g.
SageAttention's CUDA kernels, for the `arch` the pods' GPUs have; the image ships nvcc) and every file in `files`: a `url` one is downloaded to
`<name>.part`, sha256-checked, then renamed (Civitai files come from Hugging Face mirrors found by sha256 on civarchive.com); one
with only a `page` is downloaded by hand. Any file with a `page` is first looked for in the --from folders by its sha256 (whatever
it is called) and copied in under its manifest name. The image tag stays pinned: a newer
image would rsync its own ComfyUI back over the workspace (re-running this script moves it forward again).
Idempotent: files already there are skipped; what is still missing is listed with the page to get it from.

Runs inside the template container (stdlib only) after its first boot:
  python3 /seed/seed.py [--from /drop] [/seed/models.json]
HF_TOKEN is sent to Hugging Face when set (the current files are ungated).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def auth_header(url: str) -> list:
    if "huggingface.co" in url and os.environ.get("HF_TOKEN"):
        return ["-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}"]
    return []


PART_SUFFIX = ".part"


def download(url: str, dest: Path, digest: str) -> None:
    part = dest.with_name(dest.name + PART_SUFFIX)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -C - resumes an interrupted .part; curl drops the auth header on the cross-host redirect to the CDN
    subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-A", "comfy-studio-seed", *auth_header(url), "-o", str(part), url], check=True)
    got = sha256(part)
    if got != digest:
        part.unlink()
        sys.exit(f"{dest.name}: sha256 {got} != manifest {digest} (deleted the download)")
    part.rename(dest)


def from_drop(todo: dict, comfy: Path, drops: list) -> None:
    """Copy hand-downloaded files in: every file in the drop folders whose size is near a missing one's (`gb` is rounded, Civitai
    sizes are KiB) and whose sha256 matches."""
    want = {f["sha256"]: rel for rel, f in todo.items() if "page" in f}
    sizes = [todo[rel]["gb"] for rel in want.values()]
    for p in (p for d in drops for p in Path(d).iterdir() if p.is_file()):
        if want and any(abs(p.stat().st_size / 1e9 - gb) <= max(0.02, gb * 0.05) for gb in sizes) and (rel := want.pop(sha256(p), None)):
            dest = comfy / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest.with_name(dest.name + PART_SUFFIX))
            dest.with_name(dest.name + PART_SUFFIX).rename(dest)
            del todo[rel]
            print(f"{rel} <- {p}", flush=True)


def install_package(comfy: Path, name: str, spec: dict) -> None:
    """pip-install `repo`@`ref` into the venv, compiled for TORCH_CUDA_ARCH_LIST=`arch` (no GPU needed for that), once per ref: a
    marker in the venv remembers what was built, since a build takes many minutes."""
    venv = next(comfy.glob(".venv*"))
    marker = venv / f"seed-built-{name}"
    if marker.is_file() and marker.read_text() == spec["ref"]:
        print(f"package {name} @ {spec['ref'][:12]} (built before)", flush=True)
        return
    # the image's cuda-minimal-build lacks some library headers torch's own headers include (cusparse.h); torch's pip CUDA wheels
    # carry them under site-packages/nvidia/*/include, so both compilers get those dirs
    inc = sorted(str(d) for d in Path("/usr/local/lib").glob("python3*/dist-packages/nvidia/*/include"))
    env = {**os.environ, "PIP_CONSTRAINT": "", "TORCH_CUDA_ARCH_LIST": spec["arch"], "MAX_JOBS": str(os.cpu_count() or 4),
           "EXT_PARALLEL": "4", "NVCC_APPEND_FLAGS": " ".join(["--threads 4"] + [f"-I{d}" for d in inc]),
           "CPATH": ":".join(inc + [os.environ.get("CPATH", "")]).strip(":")}
    print(f"building {name} @ {spec['ref'][:12]} for compute capability {spec['arch']} (takes a while)", flush=True)
    subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "install", "--no-build-isolation", "-q",
                    f"git+{spec['repo']}@{spec['ref']}"], check=True, env=env)
    marker.write_text(spec["ref"])
    print(f"package {name} @ {spec['ref'][:12]}", flush=True)


def install_node(comfy: Path, name: str, spec: dict) -> None:
    d = comfy / "custom_nodes" / name
    if not d.is_dir():
        subprocess.run(["git", "clone", "-q", spec["repo"], str(d)], check=True)
    if subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() != spec["ref"]:
        subprocess.run(["git", "-C", str(d), "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", spec["ref"]], check=True)
    if spec.get("requirements"):
        venv_python = next(comfy.glob(".venv*/bin/python"))
        subprocess.run([str(venv_python), "-m", "pip", "install", "-q", "-r", str(d / spec["requirements"])], check=True)
    print(f"node {name} @ {spec['ref'][:12]}", flush=True)


def update_comfyui(comfy: Path, spec: dict) -> None:
    """Move the template's ComfyUI (a plain git repo of its baked release) to `ref`, "latest" = the newest v* release tag, and
    install its requirements. The image's PIP_CONSTRAINT pins the old release's frontend packages, so it is lifted here; torch
    stays the image's (requirements.txt leaves it unpinned)."""
    git = ["git", "-C", str(comfy)]
    if subprocess.run(git + ["remote", "get-url", "origin"], capture_output=True).returncode:
        subprocess.run(git + ["remote", "add", "origin", spec["repo"]], check=True)
    subprocess.run(git + ["fetch", "-q", "--tags", "--force", "origin"], check=True)
    ref = spec["ref"]
    if ref == "latest":
        tags = subprocess.run(git + ["tag", "-l", "v*", "--sort=-v:refname"], capture_output=True, text=True, check=True).stdout.split()
        ref = tags[0]
    subprocess.run(git + ["-c", "advice.detachedHead=false", "checkout", "-q", "--force", ref], check=True)
    venv_python = next(comfy.glob(".venv*/bin/python"))
    env = {**os.environ, "PIP_CONSTRAINT": ""}
    subprocess.run([str(venv_python), "-m", "pip", "install", "-q", "-r", str(comfy / "requirements.txt")], check=True, env=env)
    print(f"ComfyUI @ {ref}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", nargs="?", type=Path, default=Path(__file__).with_name("models.json"))
    ap.add_argument("--from", dest="drops", action="append", default=[], help="folder with hand-downloaded `page` files (repeatable)")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    comfy = Path(manifest["comfyui_dir"])
    if not (comfy / "main.py").is_file():
        sys.exit(f"no ComfyUI at {comfy}: boot the template ({manifest['image']}) on this workspace once first")
    update_comfyui(comfy, manifest["comfyui"])
    for name, spec in manifest["custom_nodes"].items():
        install_node(comfy, name, spec)
    for name, spec in manifest.get("packages", {}).items():
        install_package(comfy, name, spec)
    todo = {rel: f for rel, f in manifest["files"].items() if not (comfy / rel).is_file()}
    from_drop(todo, comfy, args.drops)
    fetch = {rel: f for rel, f in todo.items() if "url" in f}
    print(f"{len(manifest['files']) - len(todo)} files present, {len(fetch)} to download ({sum(f['gb'] for f in fetch.values()):.1f} GB)", flush=True)
    for i, (rel, f) in enumerate(fetch.items(), 1):
        print(f"[{i}/{len(fetch)}] {rel} ({f['gb']} GB)", flush=True)
        download(f["url"], comfy / rel, f["sha256"])
    manual = {rel: f for rel, f in todo.items() if "url" not in f}
    for rel, f in manual.items():
        print(f"MISSING {rel} ({f['gb']} GB): download it from {f['page']} into a --from folder", flush=True)
    print("workspace complete" if not manual else f"{len(manual)} file(s) still to download by hand", flush=True)
    sys.exit(1 if manual else 0)


if __name__ == "__main__":
    main()
