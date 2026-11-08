"""Job manager: runs `project.py` as a subprocess, streams its stdout, parses per-stage progress.

Lanes: `cloud` (paid cloud jobs), `local` (local ComfyUI jobs) and `runpod` (animate on our RunPod pod, when that is the ⚙
animate target) each run one job at a time (concurrent i2v submissions break the backend; one GPU locally and on the pod);
`free` (edit, dry-run, new) runs immediately.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

import config
from projects import PREVIEW_GATED, ROOT, ApiError, load, preview_approved, project_dir
import voice_registry  # noqa: E402  # pipeline/, on sys.path via projects

sys.path.insert(0, str(ROOT / "pipeline"))
import comfy_api  # noqa: E402  # cloud base URL + key; cancel needs to reach the Comfy queue
import runpod_pod  # noqa: E402
import steps  # noqa: E402

# every finished edit is copied to config.PUBLISH_DIR when set (overwriting the previous one of the same project + source folder
# + kind: a beat-synced and a no-music edit each keep their own copy)

# how long the last edit of each kind took (edits of a kind are all about the same size) -> estimate + progress bar for the next one
EDIT_TIMING = config.LOGS_DIR / "edit_timing.json"
FRAMES_RE = re.compile(r"total (\d+) frames")  # make_edit's plan line; scales the estimate when the edit is longer/shorter than the last
EDIT_KINDS = ("beat", "nomusic")


def send_pushover(title, message):
    """One Pushover message (config.pushover()); failures are logged, never raised. Returns whether Pushover accepted it."""
    user, token = config.pushover()
    data = urllib.parse.urlencode({"token": token, "user": user, "title": title,
                                   "message": message}).encode()
    try:
        urllib.request.urlopen("https://api.pushover.net/1/messages.json", data, timeout=10).read()
        return True
    except OSError as e:  # URLError, and a read timeout (TimeoutError, not a URLError)
        print(f"pushover: {e}", file=sys.stderr)
        return False


def edit_kind(req):
    """'nomusic' for a no-music (scenario) edit, else 'beat'."""
    return "nomusic" if req.get("no_music") else "beat"


def _timings():
    try:
        data = json.loads(EDIT_TIMING.read_text())
    except (OSError, ValueError):
        return {}
    return {"beat": data} if "seconds" in data else data  # older file: one beat-synced timing


def edit_timing(kind="beat"):
    return _timings().get(kind)


def save_edit_timing(kind, seconds, frames):
    data = _timings()
    data[kind] = {"seconds": round(seconds, 1), "frames": frames}
    try:
        EDIT_TIMING.parent.mkdir(parents=True, exist_ok=True)
        EDIT_TIMING.write_text(json.dumps(data))
    except OSError:
        pass


# action -> (project.py args, cost, lane). The render steps come from pipeline/steps.py (one entry per step x target); the rest is here.
ACTIONS = {
    "dry_run": (["generate", "--dry-run"], "free", "free"),
    "local_new": (["generate", "--local", "--new"], "paid", "local"),  # fresh cloud preview, then the local IL pass
    **{action: (["generate", *flags], "free" if target == "local" else "paid", target)
       for (_, target), (action, flags) in steps.ACTION_IDS.items()},
    "animate": (["animate"], "paid", "cloud"),
    "animate_rife": (["animate", "--rife"], "paid", "cloud"),  # the same cloud job also interpolates -> interp/videos/
    "interpolate": (["interpolate"], "free", "local"),
    "edit": (["edit"], "free", "free"),
    "recover": (["recover"], "free", "free"),  # save cloud clips an interrupted animate run left behind (started by the server itself)
}
ANIMATE = ("animate", "animate_rife")  # build_argv pins their --target (the ⚙ setting at submit time); runpod ones run in the runpod lane
STAGE_RE = re.compile(r"\s*\[stage_(\d+(?:\.t\d+)?)\]\s*(.*)")  # stage_N, or stage_N.tK for one of K preview takes
EXPECTED_RE = re.compile(r"expected (\d+)s")  # `[stage_N] queued (krea2) expected 31s`: the baseline in logs/timings.json
PROCESSED_RE = re.compile(r"processed (\d+)s")  # `[stage_N] done (waited 40s, processed 31s)`: ComfyUI's own execution time
MEMFREE_RE = re.compile(r"same models as the last run, but (.+) -- freeing memory first")  # targets/local.py: memory check fired at batch start
CLOUD_ID_RE = re.compile(r"submitted -> ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
LOCAL_DEFAULT = config.LOCAL_COMFY_URL
EDIT_FLAGS = {"audio": "--audio", "beat_times_file": "--beat-times-file", "start_time": "--start-time",
              "lead_in_seconds": "--lead-in-seconds", "subcuts_per_image": "--subcuts-per-image",
              "beats_per_cut": "--beats-per-cut", "last_weight": "--last-weight",
              "clip_volume": "--clip-volume", "music_volume": "--music-volume", "duck": "--duck",
              "fade_out": "--fade-out", "transition": "--transition", "transition_seconds": "--transition-seconds",
              "still_seconds": "--still-seconds", "fade_in": "--fade-in", "collage_beats": "--collage-beats",
              "card_sound_volume": "--card-sound-volume"}


class Job:
    def __init__(self, jid, project, action, argv, req):
        self.id, self.project, self.action, self.argv, self.req = jid, project, action, argv, req
        self.cost, self.lane = ACTIONS[action][1:]
        if action in ANIMATE and argv[argv.index("--target") + 1] == "runpod":
            self.lane = "runpod"  # its own lock: a pod session doesn't hold up the cloud image jobs
        self.status = "queued"  # queued | running | done | failed | cancelled
        self.lines = []
        self.stage_status = {}
        self.returncode = None
        self.created = time.time()
        self.started = self.ended = None
        self.proc = None
        self.eta, self.took, self.cursor = {}, {}, None  # per-stage expected / processed seconds; when the current stage began
        self.last_done = None  # when the latest stage finished: per_stage = (last_done - started) / finished stages
        self.frames = None  # edit jobs: frame count of the plan, once make_edit has printed it
        self.mem_freed = None  # why the local batch had to free memory first (memory check), else None
        self.cloud_ids = []  # prompt ids this job submitted (Comfy Cloud, or the pod), so cancel can pull them off the remote queue
        self.cond = threading.Condition()

    def key(self):
        r = self.req
        return (self.project, self.action, r.get("folder") or "", tuple(r.get("stages") or ()), int(r.get("takes") or 1),
                edit_kind(r) if self.action == "edit" else "")

    def info(self, with_lines=False):
        d = {"id": self.id, "project": self.project, "action": self.action, "cost": self.cost, "lane": self.lane,
             "status": self.status, "returncode": self.returncode, "stages": self.stage_status, "req": self.req,
             "created": self.created, "started": self.started, "ended": self.ended, "argv": " ".join(self.argv[2:]),
             "last": self.lines[-1] if self.lines else "", "progress": self.progress(), "per_stage": self.per_stage(), "took": self.took, "mem_freed": self.mem_freed}
        if with_lines:
            d["lines"] = self.lines
        return d

    def per_stage(self):
        """Wall-clock seconds per finished stage (a preview take counts as one), model loading included, or None before the first."""
        n = sum(v in ("done", "failed") for v in self.stage_status.values())
        return round((self.last_done - self.started) / n, 1) if n and self.started and self.last_done else None

    def progress(self):
        """Time-based progress from the baselines: {fraction, left (s), exact (every stage had a baseline)}, or None without any.
        The current stage counts for the time elapsed since the last one finished, capped just under its baseline."""
        if self.action == "edit":
            last = edit_timing(edit_kind(self.req))
            if self.status != "running" or not self.started or not last:
                return None
            est = last["seconds"] * (self.frames / last["frames"] if self.frames and last.get("frames") else 1)
            spent = time.time() - self.started
            return {"fraction": min(spent / est, 0.97) if est else 0.97, "left": max(est - spent, 0), "exact": True}
        return self._render_progress()

    def _render_progress(self):
        """Progress for a render-step job (everything but an edit): see progress()."""
        if self.status != "running" or not self.eta or self.cursor is None:
            return None
        sids = list(self.stage_status)
        mean = sum(self.eta.values()) / len(self.eta)
        finished = [x for x in sids if self.stage_status[x] in ("done", "failed")]
        todo = [x for x in sids if x not in finished]
        done_t = sum(self.took.get(x, self.eta.get(x, mean)) for x in finished)
        remaining = sum(self.eta.get(x, mean) for x in todo)
        elapsed = min(time.time() - self.cursor, 0.97 * self.eta.get(todo[0], mean)) if todo else 0
        total = done_t + remaining
        exact = all(x in self.eta for x in sids)
        if self.lane == "cloud":  # cloud jobs overlap, so they are not timed one after another: weight the finished ones by their baselines
            spent = time.time() - self.started if self.started else 0
            pace = spent / done_t if finished and done_t else 1  # how much faster/slower than the baselines this run is going
            return {"fraction": min(done_t / total, 0.99) if total else 1, "left": max(remaining * pace - (0 if finished else spent), 0), "exact": exact}
        return {"fraction": min((done_t + elapsed) / total, 1) if total else 1, "left": max(remaining - elapsed, 0), "exact": exact}

    def emit(self, line):
        with self.cond:
            self.lines.append(line)
            if self.action == "edit":
                f = FRAMES_RE.search(line)
                if f:
                    self.frames = int(f.group(1))
            f = MEMFREE_RE.search(line)
            if f:
                self.mem_freed = f.group(1)
            c = CLOUD_ID_RE.search(line)
            if c:
                self.cloud_ids.append(c.group(1))
            self._stage_line(line)
            self.cond.notify_all()

    def _stage_line(self, line):
        """A `[stage_N] ...` line: its new status and whatever timing data it carries."""
        m = STAGE_RE.match(line)
        if not m:
            return
        s = m.group(2).lower()
        if "skipping" in s:  # a stage the run leaves alone ("already exists, skipping"): not part of this job
            return
        if "failed" in s:
            st = "failed"
        elif s.startswith(("done", "->")):
            st = "done"  # the cloud prints `-> path` when a result is saved
        elif s.startswith("queued"):
            st = "queued"
        else:
            st = "running"
        self.stage_status[m.group(1)] = st
        e, t = EXPECTED_RE.search(line), PROCESSED_RE.search(line)
        if e:
            self.eta[m.group(1)] = int(e.group(1))
        if st in ("queued", "running") and self.cursor is None:
            self.cursor = time.time()
        if st in ("done", "failed"):
            self.took[m.group(1)] = int(t.group(1)) if t else self.eta.get(m.group(1)) or time.time() - (self.cursor or time.time())
            self.cursor = self.last_done = time.time()


class Manager:
    def __init__(self):
        self.jobs = {}
        self.order = deque(maxlen=200)
        self.lock = threading.Lock()
        self.seq = 0
        self.lane_locks = {"cloud": threading.Lock(), "local": threading.Lock(), "runpod": threading.Lock()}
        self.notify_armed_at = None  # the 🔔: on since this time, pushes each time the queue empties until turned off
        self.notify_reported = set()  # ids of the jobs the 🔔 already pushed about
        self.animate_batch = {}  # (project, action, folder) -> {"stages": [ids], "added": ts}: armed "Start batch" clicks, sent as one job

    def busy(self):
        return any(j.status in ("queued", "running") for j in self.jobs.values())

    def arm_notify(self, on):
        self.notify_armed_at = time.time() if on else None
        self.notify_reported = set()
        self._maybe_notify()

    def _maybe_notify(self):
        with self.lock:
            since = self.notify_armed_at
            if since is None or self.busy():
                return
            ended = [j for j in self.jobs.values()  # a cancelled job gets `ended` later, so it must not push a second time then
                     if (j.ended or time.time()) >= since and j.id not in self.notify_reported]
            if not ended:
                return  # idle with nothing new since the last push: wait for the next jobs
            self.notify_reported.update(j.id for j in ended)
        counts = {st: sum(j.status == st for j in ended) for st in ("done", "failed", "cancelled")}
        msg = "Queue empty" + (" — " + ", ".join(f"{n} {st}" for st, n in counts.items() if n) if ended else "")
        msg += "".join(f"\n✗ {j.id} {j.action}" for j in ended if j.status == "failed")  # never the project name: pushes show on the lock screen
        threading.Thread(target=send_pushover, args=("Comfy Studio", msg), daemon=True).start()

    def build_argv(self, project, action, req):
        if action not in ACTIONS:
            raise ApiError(f"unknown action {action!r}")
        d = project_dir(project)
        stages = [int(s) for s in req.get("stages") or []]
        folder = req.get("folder") or ""
        if folder and not (d / folder).is_dir():
            raise ApiError(f"no folder {folder!r}")
        cfg = load(project)
        self._check_preview_gate(action, cfg, stages)
        if action in ANIMATE:
            self._check_motion(cfg, stages)
        base = ACTIONS[action][0]
        argv = [sys.executable, str(ROOT / "project.py"), project, *base]
        if action in ("animate", "animate_rife", "interpolate"):
            return argv + self._clip_argv(action, req, folder, stages)
        if action == "edit":
            return argv + self.edit_args(d, project, req, folder, stages)
        return argv + self._staged_argv(base, req, folder, stages)

    @staticmethod
    def _clip_argv(action, req, folder, stages):
        """The arguments of a step that works on a folder's clips (animate, interpolate)."""
        argv = ([folder] if folder else []) + list(map(str, stages))
        if action in ANIMATE:
            argv += ["--target", config.animate_target()]
        if action == "interpolate" and req.get("server"):
            argv += ["--server", req["server"]]
        return argv

    @staticmethod
    def _check_motion(cfg: dict, stages: list) -> None:
        """Animate is the paid step: refuse up front (a 400 naming the stages) when one has no motion prompt."""
        by_pos = {s.get("id", pos): s for pos, s in enumerate(cfg["stages"], 1)}
        bad = [n for n in stages if not str(by_pos.get(n, {}).get("motion", "")).strip()]
        if bad:
            raise ApiError(f"stages {bad}: no `motion` prompt -- write one in the stage drawer first (Animate is the "
                           "paid step, so the run refuses to start with an empty motion)")
        voices = voice_registry.load()["voices"]
        unfilled = [f"stage {n}: {'; '.join(p)}" for n in stages if n in by_pos
                    and (p := voice_registry.problems(by_pos[n].get("motion", ""), voice_registry.cast_of(by_pos[n], cfg), voices))]
        if unfilled:
            raise ApiError(" | ".join(unfilled))

    @staticmethod
    def _check_preview_gate(action, cfg, stages):
        """Approvals are a web-UI board concept (one-offs skip them): refuse the stages whose preview isn't ✓'d yet."""
        if action not in PREVIEW_GATED or cfg.get("oneoff"):
            return
        want = set(stages)
        blocked = [s["id"] for s in cfg["stages"] if (not want or s["id"] in want) and not preview_approved(s)]
        if blocked:
            raise ApiError(f"{action}: preview not approved for stage {', '.join(map(str, blocked))}")

    @staticmethod
    def _staged_argv(base, req, folder, stages):
        """argv tail for the staged generate actions: --server, --from the reviewed upscale, --takes."""
        argv = []
        if req.get("server") and "--local" in base:
            argv += ["--server", req["server"]]
        if "--finish" in base and folder in steps.UPSCALES:  # finish from the upscale being reviewed
            argv += ["--from", folder]
        takes = int(req.get("takes") or 1)
        if takes > 1 and "--preview" in base:
            if not 1 <= takes <= 8:
                raise ApiError("takes must be 1-8")
            argv += ["--takes", str(takes)]
        return argv + list(map(str, stages))

    @staticmethod
    def _still_flags(d, req, flags: list) -> list:
        """--as-still stages (a stills/clip mix), from stills_from (validated; "" = the project root)."""
        as_still = [int(i) for i in (req.get("still_stages") or [])]
        if not as_still or req.get("mode") == "stills":
            return flags
        src = req.get("stills_from") or ""
        if src and (not re.fullmatch(r"[\w-]+", src) or not (d / src).is_dir()):
            raise ApiError(f"no stills folder {src!r}")
        return flags + ["--as-still", *map(str, as_still), "--stills-from", str(d / src)]

    @staticmethod
    def _polaroid_flags(d, req, flags: list) -> list:
        """Polaroid stills, the collage outro (its stages, from collage_from: validated, "" = the project root) and the card size."""
        if req.get("polaroid"):
            flags = flags + ["--polaroid"]
        if req.get("card_sounds"):
            flags = flags + ["--card-sounds"]
        collage = [int(i) for i in (req.get("collage") or [])]
        if collage:
            src = req.get("collage_from") or ""
            if src and (not re.fullmatch(r"[\w-]+", src) or not (d / src).is_dir()):
                raise ApiError(f"no stills folder {src!r}")
            flags = flags + ["--collage", *map(str, collage), "--collage-from", str(d / src)]
        if req.get("card_size") not in (None, ""):
            size = int(req["card_size"])
            if not 300 <= size <= 1080:
                raise ApiError("card size must be 300-1080 px")
            flags = flags + ["--card-size", str(size)]
        return flags

    @staticmethod
    def _audio_flags(req, flags: list) -> list:
        """The clip-audio flags: --no-music (a scenario edit) or --clip-audio, each with its mutes."""
        if not req.get("no_music") and not (req.get("clip_audio") and req.get("mode") != "stills"):
            return flags
        return flags + (["--no-music"] if req.get("no_music") else ["--clip-audio"]) + Manager._mute_flags(req)

    @staticmethod
    def edit_args(d, project, req, folder, stages):
        words = ([folder] if folder else []) + ([req["mode"]] if req.get("mode") in ("animate", "stills") else []) + list(map(str, stages))
        flags = Manager._name_flag(project, req)
        for k, flag in EDIT_FLAGS.items():
            if req.get(k) not in (None, ""):
                flags += [flag, str(req[k])]
        flags += Manager._trim_flags(req)
        if req.get("zoom"):
            flags += ["--zoom"] + ([] if req["zoom"] is True else [str(req["zoom"])])
        flags = Manager._still_flags(d, req, flags)
        flags = Manager._polaroid_flags(d, req, flags)
        flags = Manager._audio_flags(req, flags)
        return words + (flags if words else ["--"] + flags)

    @staticmethod
    def _name_flag(project, req):
        name = req.get("out_name") or f"{project}_{'nomusic' if req.get('no_music') else 'edit'}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        name = Path(name).name
        if not name.endswith(".mp4"):
            name += ".mp4"
        return ["--out-name", name]  # make_edit writes it to <folder>/edits/

    @staticmethod
    def _trim_flags(req):
        flags = []
        trims = {int(k): float(v) for k, v in (req.get("trims") or {}).items() if v not in (None, "")}
        if trims:
            flags += ["--trim", *(f"{k}={v}" for k, v in trims.items())]
        trims_end = {int(k): float(v) for k, v in (req.get("trims_end") or {}).items() if v not in (None, "") and float(v) > 0}
        if trims_end:
            flags += ["--trim-end", *(f"{k}={v}" for k, v in trims_end.items())]
        return flags

    @staticmethod
    def _mute_flags(req):
        muted = [int(i) for i in (req.get("mute_clip_audio") or [])]
        return ["--mute-clip-audio", *map(str, muted)] if muted else []

    def submit(self, project, action, req):
        argv = self.build_argv(project, action, req)
        with self.lock:
            self.seq += 1
            job = Job(f"j{self.seq}", project, action, argv, req)
            for j in self.jobs.values():
                if j.status in ("queued", "running") and j.key() == job.key():
                    raise ApiError(f"{action} already {j.status} as {j.id}", 409)
            self.jobs[job.id] = job
            self.order.append(job.id)
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _submit_batch(self, key, stages):
        """Submit one merged animate job for a batch (call outside self.lock: submit takes it). A duplicate of a
        queued/running job just returns that job: the batch is already covered."""
        project, action, folder = key
        try:
            return self.submit(project, action, {"action": action, "stages": stages, "folder": folder}).info()
        except ApiError as e:
            if e.status != 409:
                raise
            with self.lock:
                for j in self.jobs.values():
                    job_folder = j.req.get("folder") or ""
                    same = j.project == project and j.action == action and job_folder == folder
                    same = same and sorted(map(int, j.req.get("stages") or ())) == sorted(stages)
                    if j.status in ("queued", "running") and same:
                        return j.info()
            raise

    def add_to_batch(self, project, action, req):
        """The web UI's armed "Start batch" (RunPod target): queue animate clicks, from any project, and fire them all once
        the ⚙ batch size is reached, so the whole batch shares one pod start and one model load. A job covers one project,
        action and folder: the batch's jobs run one after another in the runpod lane."""
        if action not in ANIMATE:
            raise ApiError(f"{action} can't be batched", 400)
        stages = [int(s) for s in req.get("stages") or []]
        if not stages:
            raise ApiError("a batch needs stage ids", 400)
        key = (project, action, req.get("folder") or "")
        with self.lock:
            entry = self.animate_batch.setdefault(key, {"stages": [], "added": time.time()})
            entry["stages"] = [*dict.fromkeys([*entry["stages"], *stages])]  # keep order, drop repeats
            total = sum(len(e["stages"]) for e in self.animate_batch.values())
        size = config.animate_batch_size()
        if total < size:
            return {"job": None, "pending": total, "size": size}
        sent = self.flush_batch()["sent"]
        job = next((j for j in sent if j["project"] == project and j["action"] == action and (j["req"].get("folder") or "") == key[2]), None)
        return {"job": job, "pending": 0, "size": size}

    def flush_batch(self):
        """Fire every pending batch, even below the batch size. One that can't be submitted stays queued and its error is
        raised once the others have gone out."""
        with self.lock:
            pending = {k: e["stages"] for k, e in self.animate_batch.items()}
        sent, errors = [], []
        for key, stages in pending.items():
            try:
                sent.append(self._submit_batch(key, stages))
            except ApiError as e:
                errors.append(f"{key[0]}: {e}")
                continue
            with self.lock:
                if self.animate_batch.get(key, {}).get("stages") == stages:
                    del self.animate_batch[key]  # unchanged since the snapshot: the whole batch went out
        if errors:
            raise ApiError("; ".join(errors), 400)
        return {"sent": sent, "pending": self.batch_state()}

    def clear_batch(self):
        with self.lock:
            self.animate_batch.clear()

    def batch_state(self):
        with self.lock:
            return [{"project": p, "action": a, "folder": f, "stages": e["stages"], "added": e["added"]}
                    for (p, a, f), e in sorted(self.animate_batch.items())]

    def _run(self, job):
        lock = self.lane_locks.get(job.lane)
        if lock:
            lock.acquire()
        try:
            if job.status == "cancelled":
                return
            job.status, job.started = "running", time.time()
            job.emit("$ " + " ".join(job.argv[1:]))
            job.proc = subprocess.Popen(job.argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
            for line in job.proc.stdout:
                job.emit(line.rstrip("\n"))
            job.returncode = job.proc.wait()
            if job.status != "cancelled":
                job.status = "done" if job.returncode == 0 else "failed"
            if job.status == "done" and job.action == "edit":
                save_edit_timing(edit_kind(job.req), time.time() - job.started, job.frames)
                self.publish_edit(job)
        except Exception as e:  # noqa: BLE001
            job.emit(f"server error: {e}")
            job.status = "failed"
        finally:
            job.ended = time.time()
            with job.cond:
                job.cond.notify_all()
            if lock:
                lock.release()
            self._maybe_notify()

    @staticmethod
    def publish_edit(job):
        """Copy the edit this job rendered to config.PUBLISH_DIR (if set) as <project>_<folder>.mp4 (<project>.mp4 for the root;
        a no-music edit adds _nomusic), replacing the previous one of the same kind."""
        if not config.PUBLISH_DIR:
            return
        try:
            folder = job.req.get("folder") or ""
            made = project_dir(job.project) / folder / "edits" / job.argv[job.argv.index("--out-name") + 1]
            suffix = "_nomusic" if edit_kind(job.req) == "nomusic" else ""
            dst = config.PUBLISH_DIR / (f"{job.project}_{folder}{suffix}.mp4" if folder else f"{job.project}{suffix}.mp4")
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(made, tmp)
            os.replace(tmp, dst)
            job.emit(f"published -> {dst}")
        except Exception as e:  # noqa: BLE001
            job.emit(f"publish failed: {e}")

    def cancel(self, jid):
        job = self.jobs.get(jid)
        if not job:
            raise ApiError("no such job", 404)
        if job.status == "queued":
            job.status = "cancelled"
        elif job.status == "running" and job.proc:
            job.status = "cancelled"
            try:  # the whole group: project.py's children (animate_pipeline.py...) keep submitting and hold the lane otherwise
                os.killpg(job.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            threading.Thread(target=self._cancel_remote, args=(job,), daemon=True).start()
        self._maybe_notify()
        return job

    def retry(self, jid):
        """Submit a failed/cancelled job again with the same request (argv rebuilt from today's settings). A job naming its
        stages only reruns the ones that didn't finish, so the stages it already rendered aren't paid for twice; with none
        left unfinished (or no stage list) the request goes out unchanged."""
        job = self.jobs.get(jid)
        if not job:
            raise ApiError("no such job", 404)
        if job.status not in ("failed", "cancelled"):
            raise ApiError(f"only a failed or cancelled job can be retried ({jid} is {job.status})", 400)
        req = dict(job.req)
        stages = [int(s) for s in req.get("stages") or ()]
        if stages and job.action != "edit":  # an edit's stages pick the clips of one cut: always all of them
            done = {k.split(".")[0] for k, v in job.stage_status.items() if v == "done"}
            failed = {k.split(".")[0] for k, v in job.stage_status.items() if v != "done"}
            left = [s for s in stages if str(s) in failed or str(s) not in done]
            if left:
                req["stages"] = left
        return self.submit(job.project, job.action, req)

    def _cancel_remote(self, job):
        """Killing project.py leaves whatever it already submitted in Comfy's queue; remove those too."""
        try:
            if job.lane == "runpod":
                n = self._cancel_pod(job)
            else:
                n = self._cancel_cloud(job) if job.cloud_ids else 0
                n += self._cancel_local(job) if job.lane == "local" else 0
            job.emit(f"cancelled: removed {n} job(s) from the ComfyUI queue" if n else "cancelled: nothing left in the ComfyUI queue")
        except Exception as e:  # noqa: BLE001
            job.emit(f"cancelled, but could not clear the ComfyUI queue: {e}")

    @staticmethod
    def _post_json(call, path, body):
        return call(path, json.dumps(body).encode(), {"Content-Type": "application/json"}, "POST")

    def _cancel_cloud(self, job):
        if not config.cloud_key_present():
            return 0  # without a key the cloud queue is unreachable, and _request exits on that
        try:
            running = json.dumps(json.loads(comfy_api._request("/api/queue")).get("queue_running"))
        # _request exits on a 402 (the account can't pay): the cancel goes on without the cloud queue
        except (urllib.error.URLError, ValueError, SystemExit):  # NOSONAR(S5754)
            running = ""
        n = 0
        for pid in job.cloud_ids:
            try:
                if pid in running:
                    self._post_json(comfy_api._request, "/api/interrupt", {"prompt_id": pid})
                self._post_json(comfy_api._request, "/api/queue", {"delete": [pid]})
                n += 1
            except (urllib.error.URLError, SystemExit):  # NOSONAR(S5754) already finished / not found, or a 402 exit
                pass
        return n

    QUEUE_PATH = "/queue"

    @staticmethod
    def _caller(server):
        def call(path, data=None, headers=None, method="GET"):
            req = urllib.request.Request(server + path, data=data, headers=headers or {}, method=method)
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.read()
        return call

    def _cancel_pod(self, job):
        """The prompts this job queued on the pod: deleted, the running one interrupted (the pod stays up; its watchdog ends it)."""
        pod = runpod_pod.find()
        if not pod or not job.cloud_ids:
            return 0
        call = self._caller(runpod_pod.url(pod["id"]))
        try:
            running = json.dumps(json.loads(call(self.QUEUE_PATH)).get("queue_running"))
        except (urllib.error.URLError, ValueError):
            return 0
        for pid in job.cloud_ids:
            if pid in running:
                self._post_json(call, "/interrupt", {"prompt_id": pid})
        self._post_json(call, self.QUEUE_PATH, {"delete": job.cloud_ids})
        return len(job.cloud_ids)

    def _cancel_local(self, job):
        server = (job.req.get("server") or LOCAL_DEFAULT).rstrip("/")
        call = self._caller(server)
        wanted = {str(s) for s in job.req.get("stages") or []}

        def mine(item):  # LocalTarget.submit labels each prompt "<project> stage N[ ...]"
            extra = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
            label = extra.get("label", "")
            m = re.match(rf"{re.escape(job.project)} stage (\d+)", label)
            return bool(m) and (not wanted or m.group(1) in wanted)

        try:
            q = json.loads(call(self.QUEUE_PATH))
        except (urllib.error.URLError, ValueError):
            return 0  # server not up: nothing to clear
        pending = [i[1] for i in q.get("queue_pending", []) if mine(i)]
        if pending:
            self._post_json(call, self.QUEUE_PATH, {"delete": pending})
        if any(mine(i) for i in q.get("queue_running", [])):
            call("/interrupt", b"{}", {"Content-Type": "application/json"}, "POST")
            pending.append("running")
        return len(pending)

    def recover_orphans(self):
        """At startup: a project with i2v_pending records (animate runs killed by the last restart) gets a `recover` job that
        waits for those cloud renders and saves them."""
        from projects import PIPELINES
        for d in sorted(PIPELINES.iterdir()) if PIPELINES.is_dir() else ():
            if (d / "project.json").exists() and any(d.glob("**/_debug/i2v_pending/stage_*.json")):
                try:
                    self.submit(d.name, "recover", {})
                except ApiError:
                    pass

    def list(self):
        return [self.jobs[i].info() for i in reversed(self.order) if i in self.jobs]


manager = Manager()
