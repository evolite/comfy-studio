import json

import animate_pipeline as ap

TEMPLATES = [ap.DEFAULT_TEMPLATE, ap.AUDIO_TEMPLATE]


def test_build_graph_patches_the_hardcoded_node_ids():
    # a re-exported template that shifts node ids would make these patches land nowhere (AGENTS.md "Node-ID fragility")
    for path in TEMPLATES:
        g = ap.build_graph(json.loads(path.read_text()), "img.png", "PROMPT", 2.9, {"unlock": 1.0}, 123)
        assert g["114"]["inputs"]["image"] == "img.png"
        assert g["105:104"]["inputs"]["prompt"] == "PROMPT"
        assert g["105:111"]["inputs"]["value"] == 2.9
        assert g[ap.SEED_NODE]["inputs"]["noise_seed"] == 123
        loras = [n for n in g.values() if n["class_type"] == "LoraLoaderModelOnly" and "M3_Unlocked" in str(n["inputs"].get("lora_name"))]
        assert len(loras) == 1


def test_audio_template_has_an_audio_track_and_the_plain_one_does_not():
    audio, plain = (json.loads(p.read_text()) for p in (ap.AUDIO_TEMPLATE, ap.DEFAULT_TEMPLATE))
    assert "audio" in audio["105:91"]["inputs"]
    assert "audio" not in plain["105:91"]["inputs"]


def test_animate_rife_interpolates_the_decoded_frames_beside_the_24fps_clip():
    for path, audio in ((ap.DEFAULT_TEMPLATE, None), (ap.AUDIO_TEMPLATE, ["105:205", 0])):
        plain = ap.build_graph(json.loads(path.read_text()), "img.png", "PROMPT", 2.9)
        g = ap.build_graph(json.loads(path.read_text()), "img.png", "PROMPT", 2.9, rife=True)
        assert g["rife:vfi"]["class_type"] == "RIFE VFI" and g["rife:vfi"]["inputs"]["frames"] == ["105:10", 0]
        assert g["rife:video"]["inputs"]["fps"] == 48 and g["rife:video"]["inputs"].get("audio") == audio
        assert g[ap.RIFE_SAVE]["inputs"]["video"] == ["rife:video", 0]
        assert g["92"] == plain["92"] and g["105:91"] == plain["105:91"]  # the 24 fps clip is unchanged


def test_ltx_graph_patches_its_nodes_and_snaps_to_8k_plus_1():
    t = ap.template_for("ltx")
    g = ap.build_graph(t, "img.png", "PROMPT", 2.0, {"x.safetensors": 0.8, "off.safetensors": 0}, 7, 0.5, engine="ltx")
    assert g["ltx:image"]["inputs"]["image"] == "img.png" and g["ltx:prompt"]["inputs"]["text"] == "PROMPT"
    assert g["ltx:length"]["inputs"]["value"] == ap.frames("ltx", 2.0) == 49 and g["ltx:seed"]["inputs"]["noise_seed"] == 7
    assert g["ltx:size"]["inputs"]["megapixels"] == 0.5
    assert g["ltx:lora00"]["inputs"] == {"lora_name": "x.safetensors", "strength_model": 0.8, "model": ["ltx:ckpt", 0]}
    assert "ltx:lora01" not in g and g["ltx:guider"]["inputs"]["model"] == ["ltx:lora00", 0]
    assert "audio" not in g["ltx:video"]["inputs"] and "ltx:audio_decode" not in g  # silent unless the stage asks for audio
    loud = ap.build_graph(t, "img.png", "P", 2.0, engine="ltx", audio=True)
    assert loud["ltx:video"]["inputs"]["audio"] == ["ltx:audio_decode", 0]
    r = ap.build_graph(t, "img.png", "P", 2.0, rife=True, engine="ltx", audio=True)
    assert r["rife:vfi"]["inputs"]["frames"] == ["ltx:decode", 0] and r["rife:video"]["inputs"]["audio"] == ["ltx:audio_decode", 0]
    assert r[ap.RIFE_SAVE]["inputs"]["filename_prefix"] == "video/LTX23_rife"


def test_ltx_template_links_resolve():
    g = ap.template_for("ltx")
    for nid, node in g.items():
        for v in node["inputs"].values():
            if isinstance(v, list):
                assert v[0] in g, f"{nid} -> missing {v[0]}"


def test_frames_grids():
    assert [ap.frames("minimax", s) for s in (1.0, 2.0, 2.9, 5.0)] == [39, 56, 73, 124]
    assert [ap.frames("ltx", s) for s in (0.5, 1.0, 2.0, 5.0)] == [17, 25, 49, 121]
