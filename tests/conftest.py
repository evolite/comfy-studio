"""Tests run against a throwaway data dir: config reads STUDIO_DATA_DIR once, so set it before anything imports config."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["STUDIO_DATA_DIR"] = tempfile.mkdtemp(prefix="studio_test_")
# a fixed test library (LoRAs, local model names) instead of defaults/, so the tests don't follow the starter set
shutil.copytree(ROOT / "tests" / "fixtures" / "library", Path(os.environ["STUDIO_DATA_DIR"]) / "library")
os.environ["CIVITAI_DIR"] = os.path.join(os.environ["STUDIO_DATA_DIR"], "civitai")  # .env may point these at real folders
os.environ["PUBLISH_DIR"] = ""  # off
os.environ["PUSHOVER_USER"] = os.environ["PUSHOVER_TOKEN"] = "test"  # tests patch jobs.send_pushover; nothing is sent
for p in (ROOT, ROOT / "pipeline", ROOT / "video", ROOT / "webui" / "server"):
    sys.path.insert(0, str(p))


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """The local target waits a few seconds after unloading models and between polls; the fake server needs neither."""
    from targets import local
    monkeypatch.setattr(local.time, "sleep", lambda s: None)
