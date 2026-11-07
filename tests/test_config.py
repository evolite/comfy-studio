import os
from pathlib import Path

import config


def test_data_dir_comes_from_the_environment():
    assert config.DATA_DIR == Path(os.environ["STUDIO_DATA_DIR"]).resolve()
    assert config.PIPELINES_DIR == config.DATA_DIR / "pipelines"
    assert config.AUDIO_DIR.parent == config.DATA_DIR


def test_data_path_resolves_relative_paths_inside_the_data_dir():
    assert config.data_path("audio/Song.json") == config.DATA_DIR / "audio" / "Song.json"
    assert config.data_path("/abs/file.mp3") == Path("/abs/file.mp3")
