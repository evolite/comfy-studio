"""Comfy Cloud REST client, as the one-off scripts and the dashboard use it. The implementation lives in targets/cloud.py (CloudTarget
is what the project commands run through, see runner.py); this module keeps the short names those callers import."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py
import config  # noqa: E402,F401  # callers use comfy_api.config.ONEOFFS_DIR
from stagegraph import check_stage_ids, stage_graph  # noqa: E402,F401
from targets.cloud import (CLIENT_ID, CLOUD_BASE, MAX_RESTARTS, OUTPUT_DIR, ROOT, STALL_SECONDS, TERMINAL, api_key, cancel, download,  # noqa: E402,F401
                           fetch_job, get, job_state, poll_status, run_batch, save_outputs, stalled, submit, upload_image, _request)
