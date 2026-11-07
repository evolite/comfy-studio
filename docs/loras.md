# How LoRAs work

**The library is `loras.json`**: yours in `data/library/loras.json`, else the starter in `defaults/loras.json` (the
first save from the web UI writes your copy; the version it replaces goes to `data/library/_bak/`). Every image and i2v LoRA with its files, triggers, suggested strength,
one-line note, tips (prompt phrasing, tested strengths, A/B paths) and source link. Edit it on the web UI's **LoRAs** page
(header link, `#/loras`) or by hand; fields are documented in `pipeline/lora_registry.py`, and every save is validated.
Names never change once created (projects reference them); an entry can only be removed while no stage uses it.

## What's available: `python3 pipeline/lora_registry.py`
This doc names no LoRAs on purpose; the library changes from the UI. **Pick only from the catalog this prints**, read fresh
each time: every switched-on LoRA per kind (strength or slider range, cloud / local only, note, tips), the combos whose
members are all on, the voices that are on, and the names that are off. With the local ComfyUI up it also checks each image
file is in its loras folder (`MISSING locally` = registered but not on this machine).
- **on / off** (`disabled`, the LoRAs page toggle; voices have the same flag): off = retired. The stage editor stops offering
  it; stages that already use it still render, but the dry-run prints a warning and `project.py <name> show` lists it under
  `switched_off_loras`, so swap it when touching that stage.
- **cloud / local only** (`cloud`): a cloud run refuses a `cloud: false` LoRA before queueing anything.

## Combos (`combos` in loras.json)
A combo is a named, tested set of LoRAs with their strengths, e.g. `finish` = `motion_a` 1 + `motion_b` 0.7. It is a preset: the stage editor's *combos* chips copy its LoRAs into the stage's `loras` /
`video_loras` (a stage can tune one strength after), so the pipeline never sees combos. Make one with *save as combo…*
in the stage editor (the stage's current LoRAs), edit or delete on the LoRAs page. A combo's `tips` show beside the motion
while it is applied. Deleting a LoRA takes it out of its combos.

## Adding one
- **Local file**: drop it in the local ComfyUI's `models/loras/` (Krea2 ones in `krea/`, IL ones at the root or `IL/`),
  press *rescan* on the LoRAs page, then *register* it from "In the local loras folder" (pass guessed from the folder,
  name/trigger from the filename) or *add to* an existing entry's missing pass. *Ignore* hides files that aren't
  Krea2/IL LoRAs (ControlNets, Pony/SDXL). New entries start **local only**.
- **Cloud upload**: Comfy Cloud's `GET /api/experiment/models/loras` lists only the public catalog, not your uploads,
  so availability there is the manual `cloud` flag ("uploaded to Comfy Cloud"). Trust the exact filename of the upload.
  A cloud run of a stage that uses a LoRA with `cloud: false` stops before anything is queued (`CloudTarget.check`, and
  `animate_pipeline.py` for i2v).
- **i2v**: video LoRAs run on the cloud only; add them by upload filename. A stage may also use a literal
  `.safetensors` name in `video_loras`; the LoRAs page offers to register those.

## Image LoRAs (Krea2 / Illustrious) — stage field `loras`
Templates carry no LoRA nodes. A stage's `loras` (`{"name": strength}`, decimals ok, order = chain order) is the only
switch; `pipeline/image_loras.py` chains one `LoraLoaderModelOnly` per entry at build time — Krea2 pass off `30:10`
(nodes `30:300+k`, feeds `30:3`), Illustrious pass (`full` engine only, entries with an `il` file) off `433:368`
(nodes `433:300+k`, feeds `433:366`/`434:392`/`434:421`). An entry's `default` turns it on for every stage that
doesn't mention it; leave it unset for character LoRAs, which belong only on stages of that character. Trigger words are appended while a LoRA is active — keep them out of scene text.
Locally, Krea2-chain files are looked up in `krea_lora_folder` and the rest in `il_lora_folder`
(`defaults/local_models.json`, overridden by `data/library/local_models.json`; a file stored with a folder, e.g. `IL/x.safetensors`, is used as is).

## MiniMax i2v LoRAs — stage field `video_loras`
`animate_pipeline.py` chains one `LoraLoaderModelOnly` per entry (in order, nodes `105:300+k`) after the turbo switch
`105:122` onto the base template; strength 0 is skipped. The 8-step turbo LoRA
(`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16`) is always on in the template (Boolean `105:126`). Per-LoRA prompt phrasing lives in
each entry's `tips`.

## Other i2v models (`video_engine`)
A video entry's `engine` names the model it was trained for (absent = `minimax`; `ltx` = LTX 2.3 / Sulphur 2). The stage editor only
offers LoRAs for the stage's model, and `animate` refuses one that's on for another model. LTX LoRAs chain the same way, after
the checkpoint (nodes `ltx:loraNN`).
