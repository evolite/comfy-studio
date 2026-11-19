import numpy as np
import pytest

import loop_clip


def periodic(n, period, drift=0.0, size=64, seed=0):
    """Frames of a bright bar sweeping with the given period, plus a slow drift and a little noise."""
    rng = np.random.default_rng(seed)
    x = np.arange(size)
    out = []
    for t in range(n):
        pos = (np.sin(2 * np.pi * t / period) + 1) * (size - 8) / 2
        f = np.where(np.abs(x - pos) < 4, 200.0, 20.0) + drift * t + rng.normal(0, 1, size)
        out.append(np.tile(f, 4))
    return np.array(out, dtype=np.float32)


def test_the_loop_is_a_whole_number_of_periods():
    i, j = loop_clip.find_loop(periodic(200, 22), 0, 200, 48)
    assert (j - i) % 22 in (0, 1, 21)


def test_the_loop_stays_inside_the_range():
    i, j = loop_clip.find_loop(periodic(240, 20), 100, 230, 48)
    assert 100 <= i and j <= 230


def test_a_range_shorter_than_a_loop_is_refused():
    with pytest.raises(ValueError, match="too short"):
        loop_clip.find_loop(periodic(60, 20), 0, 40, 48)


def test_loop_api_refuses_a_missing_clip(tmp_path):
    from test_api import client
    c = client()
    c.post("/api/projects", json={"name": "20260101_loop_t", "engine": "krea2"})
    r = c.post("/api/projects/20260101_loop_t/stages/1/loop", json={"folder": "interp", "start": 0, "end": 2})
    assert r.status_code == 404
