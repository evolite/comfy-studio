"""Civitai copies of stills: the A1111 `parameters` text from the graph that made an image, model hashes, the tagged PNG."""
import hashlib
import io
import json
import os

from PIL import Image

import civitai_meta
import config
import takes

GRAPH = {
    "30:6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a woman on a bed"}},
    "30:10": {"class_type": "UNETLoader", "inputs": {"unet_name": "unet_a.safetensors"}},
    "30:300": {"class_type": "LoraLoaderModelOnly", "_meta": {"title": "Load LoRA (style_a)"},
               "inputs": {"lora_name": "krea/lora_t.safetensors", "strength_model": 1.0}},
    "30:301": {"class_type": "LoraLoaderModelOnly", "_meta": {"title": "Load LoRA (detail)"},
               "inputs": {"lora_name": "krea/lora_c.safetensors", "strength_model": 0}},
    "30:3": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 8, "cfg": 1, "sampler_name": "euler", "scheduler": "beta"}},
}


def png(w=64, h=96, text=None):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "red").save(buf, "PNG")
    return buf.getvalue()


def setup(tmp_path, monkeypatch):
    models = tmp_path / "models"
    (models / "diffusion_models").mkdir(parents=True)
    (models / "loras" / "krea").mkdir(parents=True)
    (models / "diffusion_models" / "unet_a.safetensors").write_bytes(b"unet bytes")
    (models / "loras" / "krea" / "lora_t.safetensors").write_bytes(b"lora bytes")
    monkeypatch.setattr(config, "COMFYUI_MODELS_DIR", models)
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(civitai_meta, "_index", None)
    monkeypatch.setattr(civitai_meta, "_fetch_version", lambda h: None)  # no network
    d = tmp_path / "proj"
    (d / "preview").mkdir(parents=True)
    src = tmp_path / "render.png"
    src.write_bytes(png())
    tid = takes.add(d, 1, src, GRAPH, {"prompt": "a woman on a bed"}, make_current=True)
    return d, tid


def autov2(b):
    return hashlib.sha256(b).hexdigest()[:10].upper()


def test_preview_params_carry_prompt_sampler_and_hashes(tmp_path, monkeypatch):
    d, tid = setup(tmp_path, monkeypatch)
    text = civitai_meta.params(d, "preview", 1)
    assert text.startswith("a woman on a bed <lora:style_a:1>\nNegative prompt: \n")  # the 0-strength LoRA is left out
    assert "Steps: 8, Sampler: Euler, Schedule type: Beta, CFG scale: 1, Seed: 42, Size: 64x96" in text
    assert f"Model hash: {autov2(b'unet bytes')}, Model: unet_a" in text
    assert f'Lora hashes: "style_a: {autov2(b"lora bytes")}"' in text
    assert json.loads((tmp_path / "logs" / "model_hashes.json").read_text())  # cached
    assert civitai_meta.params(d, "preview", 1, take=tid) == text


def test_civitai_resources_name_each_known_version(tmp_path, monkeypatch):
    d, _ = setup(tmp_path, monkeypatch)
    calls = []
    ids = {autov2(b"unet bytes"): 11, autov2(b"lora bytes"): 22}
    monkeypatch.setattr(civitai_meta, "_fetch_version", lambda h: calls.append(h) or ids.get(h))
    text = civitai_meta.params(d, "preview", 1)
    assert text.endswith('Civitai resources: [{"type":"checkpoint","modelVersionId":11},{"type":"lora","weight":1.0,"modelVersionId":22}]')
    civitai_meta.params(d, "preview", 1)
    assert len(calls) == 2  # cached after the first lookup


def test_unreachable_civitai_leaves_the_resources_out_and_retries(tmp_path, monkeypatch):
    d, _ = setup(tmp_path, monkeypatch)
    def down(h):
        raise OSError("offline")
    monkeypatch.setattr(civitai_meta, "_fetch_version", down)
    assert "Civitai resources" not in civitai_meta.params(d, "preview", 1)
    assert not (tmp_path / "logs" / "civitai_versions.json").exists()


def test_a_model_not_on_this_machine_keeps_its_name_only(tmp_path, monkeypatch):
    d, _ = setup(tmp_path, monkeypatch)
    os.remove(config.COMFYUI_MODELS_DIR / "diffusion_models" / "unet_a.safetensors")
    monkeypatch.setattr(civitai_meta, "_index", None)
    text = civitai_meta.params(d, "preview", 1)
    assert "Model hash" not in text and "Model: unet_a" in text


def test_upscales_carry_the_base_render_plus_their_upscaler(tmp_path, monkeypatch):
    d, _ = setup(tmp_path, monkeypatch)
    base = civitai_meta.params(d, "preview", 1)
    for f in ("hires", "IL", "finish"):
        (d / f).mkdir()
        (d / f / "stage_1.png").write_bytes(png(96, 144))
    dbg = d / "IL" / "_debug" / "stage_1"
    dbg.mkdir(parents=True)
    (dbg / "graph.json").write_text(json.dumps({"7": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x_Siax.pth"}}}))
    expect = lambda up: base.replace("Size: 64x96", "Size: 96x144").replace(", Version:", f", Hires upscaler: {up}, Version:")
    assert civitai_meta.params(d, "hires", 1) == expect("SesquiLSR")
    assert civitai_meta.params(d, "IL", 1) == expect("4x_Siax")
    assert civitai_meta.params(d, "finish", 1) == expect("SesquiLSR + SeedVR2")


def test_tagged_png_is_a_copy_with_the_parameters_chunk(tmp_path, monkeypatch):
    d, _ = setup(tmp_path, monkeypatch)
    f = d / "preview" / "stage_1.png"
    before = f.read_bytes()
    data = civitai_meta.tagged_png(f, "hello\nSteps: 8")
    assert Image.open(io.BytesIO(data)).info["parameters"] == "hello\nSteps: 8"
    assert f.read_bytes() == before  # the original is untouched
