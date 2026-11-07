# data/ — your part of comfy-studio

Everything personal lives here, and git ignores it. Start from this skeleton (`cp -r data.example data`) or point
`STUDIO_DATA_DIR` at another folder.

```
library/     your libraries; a file here replaces the starter in ../defaults/ (models.json and local_models.json are
             merged per key). The web UI writes loras.json / voices.json here and keeps old versions in library/_bak/.
pipelines/   one folder per project (project.json + every render); example_project/ is a two-stage SFW sample
oneoffs/     single-image / single-clip runs
audio/       music: <Song>.mp3 + <Song>.json beat file ({"start_time": s, "beats": [s, ...]}); DEFAULT_BEATS picks the house track
stage_sets/  importable stage sets
logs/        timing baselines
private/     anything else of yours: notes, prompt guides, agent rules (e.g. AGENTS.local.md imported by CLAUDE.local.md)
```
