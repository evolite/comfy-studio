"""Animate chain: the stages' start folders, and the hand-off of a clip's last frame to the next stage."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import animate_pipeline as clips
import project as P
import steps


def _ctx(tmp_path, ids=(3, 1, 5)):
    return SimpleNamespace(d=tmp_path, clip_from="hires", stages={i: {"id": i, "motion": "m"} for i in ids})


def test_chain_plan_follows_board_order_and_starts_later_stages_from_chain(tmp_path):
    ctx = _ctx(tmp_path)
    (tmp_path / "hires").mkdir()
    (tmp_path / "hires" / "stage_3.png").write_bytes(b"png")
    order, inputs = P.chain_plan(ctx, tmp_path / "hires", [5, 3, 1])
    assert order == [3, 1, 5] and inputs == {3: "hires", 1: "chain", 5: "chain"}
    ctx.chain_inputs = inputs
    assert steps._clip_from(ctx, 3) == "hires" and steps._clip_from(ctx, 5) == "chain" and steps._clip_from(ctx) == "hires"


def test_chain_plan_resumes_from_an_earlier_hand_off_and_needs_a_start(tmp_path):
    ctx = _ctx(tmp_path)
    (tmp_path / "chain").mkdir()
    (tmp_path / "chain" / "stage_1.png").write_bytes(b"png")
    assert P.chain_plan(ctx, tmp_path / "hires", [1, 5])[1] == {1: "chain", 5: "chain"}
    with pytest.raises(SystemExit, match="no hires/stage_3.png"):
        P.chain_plan(ctx, tmp_path / "hires", [3, 5])
    with pytest.raises(SystemExit, match="two or more"):
        P.chain_plan(ctx, tmp_path / "hires", [3])


def test_hand_off_writes_the_next_stages_first_frame(tmp_path):
    last = tmp_path / "last.png"
    last.write_bytes(b"frame1")
    ctx = SimpleNamespace(d=tmp_path, chain_next={3: 1})
    job = SimpleNamespace(n=3, tag="stage_3", debug_dir=tmp_path)
    steps._hand_off(ctx, job, {clips.LAST_SAVE: last})
    assert (tmp_path / "chain" / "stage_1.png").read_bytes() == b"frame1"
    steps._hand_off(ctx, SimpleNamespace(n=1, tag="stage_1", debug_dir=tmp_path), {})  # the chain's last stage: nothing to hand on
    with pytest.raises(RuntimeError, match="no last frame"):
        steps._hand_off(ctx, job, {})
