import numpy as np
import pytest

import loop_clip


def periodic(n, period, drift=0.0, seed=0):
    """Thumbnail-sized frames of a bright bar sweeping with the given period, plus a slow drift and a little noise."""
    rng = np.random.default_rng(seed)
    w, h = loop_clip.THUMB
    x = np.arange(w)
    out = []
    for t in range(n):
        pos = (np.sin(2 * np.pi * t / period) + 1) * (w - 8) / 2
        row = np.where(np.abs(x - pos) < 4, 200.0, 20.0) + drift * t
        out.append((np.tile(row, (h, 1)) + rng.normal(0, 1, (h, w))).ravel())
    return np.array(out, dtype=np.float32)


def test_a_region_that_drifts_away_and_back_is_matched_too():
    """The bar repeats every 20 frames; a patch in one corner (a head) is only in place before frame 60 and after 140:
    the loop must not join a frame with the patch to one without it."""
    f = periodic(240, 20)
    w, h = loop_clip.THUMB
    img = f.reshape(-1, h, w)
    img[60:140, :16, :16] += 120
    i, j = loop_clip.find_loop(img.reshape(len(f), -1), 0, 240, 48)
    has = lambda k: 60 <= k < 140
    assert has(i) == has(j - 1) == has(j)


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
