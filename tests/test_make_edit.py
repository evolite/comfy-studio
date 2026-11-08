import shutil
import subprocess
from pathlib import Path

import pytest

import make_edit as mt


def test_frame_counts_come_from_cumulative_boundaries():
    durations = [0.9333] * 30
    counts = mt.durations_to_frame_counts(durations)
    assert all(c >= 1 for c in counts)
    assert sum(counts) == round(sum(durations) * mt.FPS)  # no drift from per-cut rounding


def test_beat_cut_durations_span_real_beats_and_hold_the_last_cut():
    beats = [i * 0.5 for i in range(40)]
    punch, durations = mt.beat_cut_durations(beats, 4.0, 3, 2, 2.0)
    assert punch == 4.0
    assert durations == pytest.approx([1.0, 1.0, 2.0])


def test_plan_cuts_groups_cuts_per_stage_appearance():
    clips = [Path("stage_1.mp4"), Path("stage_2.mp4"), Path("stage_3.mp4")]
    cuts = mt.plan_cuts(clips, 2.0, [1.0, 1.0, 1.0], [2, 1])
    assert cuts[0].lead_in and cuts[0].clip == clips[0]
    assert [c.seg for c in cuts[1:]] == [1, 1, 2]
    assert cuts[2].offset == pytest.approx(cuts[1].frames / mt.FPS)  # sub-cuts play on through the same clip


def test_stage_sources_mix_clips_and_stills_from_another_folder(tmp_path):
    clips, stills = tmp_path / "interp", tmp_path / "finish"
    (clips / "videos").mkdir(parents=True)
    stills.mkdir()
    for n in (1, 2, 3):
        (clips / "videos" / f"stage_{n}.mp4").touch()
        (stills / f"stage_{n}.png").touch()
    (stills / "stage_4.png").touch()
    got = mt.stage_sources(clips, as_still=[2, 4], stills_dir=stills)
    assert got == [clips / "videos/stage_1.mp4", stills / "stage_2.png", clips / "videos/stage_3.mp4", stills / "stage_4.png"]
    assert mt.stage_sources(clips, order=[4, 1, 4], as_still=[4], stills_dir=stills) == \
        [stills / "stage_4.png", clips / "videos/stage_1.mp4", stills / "stage_4.png"]
    with pytest.raises(FileNotFoundError, match=r"\[4\]"):
        mt.stage_sources(clips, order=[1, 4], as_still=[2], stills_dir=stills)  # 4 has no clip
    assert mt.stage_sources(stills) == [stills / f"stage_{n}.png" for n in (1, 2, 3, 4)]  # no clips: plain stills


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_voice_segments_place_each_clip_once_and_honour_mutes(tmp_path):
    def make(name, audio):
        p = tmp_path / name
        args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3:r=24"]
        args += ["-f", "lavfi", "-i", "sine=d=3", "-shortest"] if audio else []
        subprocess.run([*args, "-pix_fmt", "yuv420p", str(p)], check=True)
        return p
    silent, talky, talky2 = make("stage_1.mp4", False), make("stage_2.mp4", True), make("stage_3.mp4", True)
    mt.TRIMS.clear()
    cuts = mt.plan_cuts([silent, talky, talky2], 2.0, [1.0, 1.0, 1.0], [2, 1])
    segs = mt.voice_segments(cuts)
    assert [s[0] for s in segs] == [talky, talky2]            # the silent lead-in clip is skipped
    start, length = segs[0][1], segs[0][2]
    assert start == pytest.approx(cuts[0].frames / mt.FPS)     # plays at its stage's start...
    assert length == pytest.approx((cuts[1].frames + cuts[2].frames) / mt.FPS)  # ...once, across both sub-cuts
    assert [s[0] for s in mt.voice_segments(cuts, muted={2})] == [talky2]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
@pytest.mark.parametrize("transition", ["fade", "cut"])
def test_no_music_edit_plays_every_stage_in_full_through_transitions(tmp_path, transition):
    run = tmp_path / "run"
    (run / "videos").mkdir(parents=True)
    for n, d in ((1, 2), (2, 3)):  # stage 1 carries a tone, stage 2 is silent
        tone = ["-f", "lavfi", "-i", f"sine=d={d}"] if n == 1 else []
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=red:s=64x96:d={d}:r=24", *tone,
                        "-shortest", str(run / "videos" / f"stage_{n}.mp4")], check=True)
    out = mt.make_scenario(run, str(tmp_path / "out.mp4"), transition=transition, transition_seconds=0.5, fade_in=0.2)
    frames = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                                 "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout)
    full = 54 + 84  # each clip minus its 0.2 s trim, at the 30 fps edit
    assert frames == (full if transition == "cut" else full - 15)  # one 0.5 s overlap
    assert mt.has_audio(out)
    out = mt.make_scenario(run, str(tmp_path / "end.mp4"), transition=transition, transition_seconds=0.5,
                             opts=mt.EditOptions(trims_end={2: 1.0}))
    short = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries",
                                "stream=nb_read_frames", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout)
    assert short == frames - 30  # --trim-end 2=1: one second off stage 2's end
