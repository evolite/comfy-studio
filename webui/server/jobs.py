"""Jobs: what the web server submits, cancels and reads. The jobs themselves run in the queue daemon (queue_daemon.py, the
comfy-queue user service), which starts each one's `project.py` as its own process and parses its output into per-stage
progress; both sides share the durable store (jobstore.py), so restarting the web server (or the daemon) leaves running
jobs alone.

Lanes: `cloud` (paid cloud jobs), `local` (local ComfyUI jobs) and `runpod` (animate on our RunPod pod, when that is the ⚙
animate target) each run one job at a time (concurrent i2v submissions break the backend; one GPU locally and on the pod);
`free` (edit, dry-run, new) runs immediately.

A failed job in a lane (not `free`: its failures don't fix themselves) is submitted again RETRY_AFTER later, up to
RETRY_TIMES times in a row (overnight: a RunPod GPU that isn't free, a cloud timeout); cancelling the failed job stops it.
"""
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import jobstore  # first: it puts the repo root (config.py) on sys.path
import config  # noqa: E402
from projects import PREVIEW_GATED, ROOT, ApiError, load, preview_approved, project_dir
import voice_registry  # noqa: E402  # pipeline/, on sys.path via projects

sys.path.insert(0, str(ROOT / "pipeline"))
import runpod_pod  # noqa: E402
import steps  # noqa: E402
import targets  # noqa: E402

# how long the last edit of each kind took (edits of a kind are all about the same size) -> estimate + progress bar for the next one
EDIT_TIMING = config.LOGS_DIR / "edit_timing.json"
EDIT_KINDS = ("beat", "nomusic")
RETRY_AFTER, RETRY_TIMES = 30 * 60, 16  # seconds; 16 x 30 min = a night


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
    """'nomusic' for a full-length (scenario) edit -- no music, or music without beat sync -- else 'beat'."""
    return "nomusic" if req.get("no_music") or req.get("beat_sync") is False else "beat"


def _timings():
    try:
        data = json.loads(EDIT_TIMING.read_text())
    except (OSError, ValueError):
        return {}
    return data


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
    "animate_rife": (["animate"], "paid", "cloud"),  # i2v + RIFE in the same job: <folder>/videos/ and interp/videos/
    "edit": (["edit"], "free", "free"),
}
ANIMATE = ("animate_rife",)  # build_argv pins their --target (the ⚙ setting at submit time); runpod ones run in the runpod lane
EVENT = "::event "  # pipeline/events.py PREFIX: a machine-readable progress line (stage status, timings, prompt ids)
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
        self.attempt = 1  # auto-retries count up from the job they retry
        self.retry_at = None  # when this failed job is submitted again (None: it won't be)
        self.prompt_ids = []  # prompt ids this job queued (any backend), so cancel can pull them off that queue
        self.stage_prompts = {}  # stage tag -> its latest prompt id (cancel one stage)
        self.last = ""  # the latest log line (the jobs panel's one-line status)
        self.pid = None  # the daemon's wrapper process (its process group = the job), so another daemon can reattach or kill it
        self.cancel_requested = False
        self.log_offset = 0  # bytes of the log the daemon has parsed (a reattaching daemon reads on from here)

    # the state that goes into jobstore (everything but the log lines, which live in the job's log file)
    STORED = ("id", "project", "action", "argv", "req", "cost", "lane", "status", "stage_status", "returncode", "created", "started",
              "ended", "eta", "took", "cursor", "last_done", "frames", "mem_freed", "attempt", "retry_at", "prompt_ids", "stage_prompts", "last", "pid", "log_offset")

    def to_data(self) -> dict:
        return {k: getattr(self, k) for k in self.STORED}

    @classmethod
    def from_data(cls, d: dict) -> "Job":
        j = cls.__new__(cls)
        j.lines, j.proc, j.stage_prompts = [], None, {}  # fields a row stored before they existed start empty
        j.last, j.pid, j.prompt_ids, j.attempt, j.retry_at, j.mem_freed, j.frames, j.log_offset = "", None, [], 1, None, None, None, 0
        for k in cls.STORED:
            if k in d:
                setattr(j, k, d[k])
        j.cancel_requested = bool(d.get("cancel"))
        return j

    def log_lines(self) -> list:
        """The job's shown lines (its `::event` lines are state, not text)."""
        return self.lines or [x for x in jobstore.read_log(self.id)[0] if not x.startswith(EVENT)]

    def key(self):
        r = self.req
        return (self.project, self.action, r.get("folder") or "", tuple(r.get("stages") or ()), int(r.get("takes") or 1),
                edit_kind(r) if self.action == "edit" else "")

    def info(self, with_lines=False):
        d = {"id": self.id, "project": self.project, "action": self.action, "cost": self.cost, "lane": self.lane,
             "status": self.status, "returncode": self.returncode, "stages": self.stage_status, "req": self.req,
             "created": self.created, "started": self.started, "ended": self.ended, "argv": " ".join(self.argv[2:]),
             "last": self.lines[-1] if self.lines else self.last, "progress": self.progress(), "per_stage": self.per_stage(), "took": self.took, "mem_freed": self.mem_freed,
             "attempt": self.attempt, "retry_at": self.retry_at}
        if with_lines:
            d["lines"] = self.log_lines()
        return d

    def per_stage(self):
        """Wall-clock seconds per finished stage (a preview take counts as one), model loading included, or None before the first."""
        n = sum(v in ("done", "failed") for v in self.stage_status.values())  # a cancelled stage took no full render
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
        finished = [x for x in sids if self.stage_status[x] in ("done", "failed", "cancelled")]
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
        """One log line (the daemon calls it for each line it reads from the job's log): an `::event` line
        (pipeline/events.py) updates the job's state, any other is shown."""
        if line.startswith(EVENT):
            try:
                self._event(json.loads(line[len(EVENT):]))
            except (ValueError, TypeError, KeyError):
                pass
            return
        self.lines.append(line)
        self.last = line

    def _event(self, e: dict):
        if e.get("note") == "memfree":
            self.mem_freed = e.get("reason")
        elif e.get("note") == "frames":
            self.frames = int(e["n"])
        elif e.get("stage") is not None and e.get("status") != "skipped":  # a skipped stage isn't part of this job
            self._stage_event(str(e["stage"]), e)

    def _stage_event(self, sid, e):
        """A stage's new status and whatever timing data it carries; `submitted` (to Comfy Cloud or the pod) is running.
        Every prompt id is kept, so a cancel can pull it off the backend's queue (queue_daemon target_of)."""
        st = {"submitted": "running"}.get(e["status"], e["status"])
        if e.get("prompt_id"):
            self.stage_prompts[sid] = e["prompt_id"]
            if e["prompt_id"] not in self.prompt_ids:
                self.prompt_ids.append(e["prompt_id"])
        self.stage_status[sid] = st
        if e.get("expected"):
            self.eta[sid] = int(e["expected"])
        if st in ("queued", "running") and self.cursor is None:
            self.cursor = time.time()
        if st in ("done", "failed"):
            self.took[sid] = int(e["processed"]) if e.get("processed") else self.eta.get(sid) or time.time() - (self.cursor or time.time())
            self.cursor = self.last_done = time.time()



class Manager:
    """The web server's side: submits (validated argv into jobstore), cancels, retries and reads jobs, and holds "Start batch"
    clicks as `held` jobs in the same store."""

    def __init__(self):
        self.lock = threading.Lock()

    @staticmethod
    def get(jid):
        d = jobstore.get(jid)
        return Job.from_data(d) if d else None

    @staticmethod
    def active(lane=None, project=None) -> list:
        return [j for j in map(Job.from_data, jobstore.by_status(*jobstore.ACTIVE))
                if (lane is None or j.lane == lane) and (project is None or j.project == project)]

    def busy(self):
        return bool(self.active())

    @staticmethod
    def notify_armed():
        return jobstore.setting("notify_armed_at") is not None

    @staticmethod
    def arm_notify(on):
        """The 🔔: from now on the daemon pushes each time the queue empties (until turned off)."""
        jobstore.set_setting("notify_armed_at", time.time() if on else None)
        jobstore.set_setting("notify_reported", [])

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
        if action in ANIMATE:
            return argv + self._clip_argv(action, req, folder, stages)
        if action == "edit":
            return argv + self.edit_args(d, project, req, folder, stages)
        return argv + self._staged_argv(base, req, folder, stages)

    @staticmethod
    def _clip_argv(action, req, folder, stages):
        """animate's arguments: the stills' folder, the stages, and the ⚙ animate target at submit time."""
        return ([folder] if folder else []) + list(map(str, stages)) + ["--target", config.animate_target()]

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
        """The sound flags. The clips' own sound is the base; the music a layer over it from music_from_stage on:
        no_music = full-length stages, their sound only (--no-music); beat_sync False = the same layout with the track over it
        (--no-sync); else the beat-cut edit with the clips' sound under the music (--clip-audio; a stills edit has none)."""
        mutes = Manager._mute_flags(req)
        if req.get("no_music"):
            return flags + ["--no-music"] + mutes
        music = (["--ambient-music"] if req.get("ambient_music") else []) + \
                (["--music-from-stage", str(int(req["music_from_stage"]))] if req.get("music_from_stage") not in (None, "") else [])
        if req.get("beat_sync") is False:
            return flags + ["--no-sync"] + music + mutes
        clip = ["--clip-audio"] + mutes if req.get("clip_audio", True) and req.get("mode") != "stills" else []
        return flags + music + clip

    @staticmethod
    def edit_args(d, project, req, folder, stages):
        words = ([folder] if folder else []) + ([req["mode"]] if req.get("mode") in ("animate", "stills") else []) + list(map(str, stages))
        flags = Manager._name_flag(project, req)
        for k, flag in EDIT_FLAGS.items():
            if req.get(k) not in (None, ""):
                flags += [flag, str(req[k])]
        if req.get("zoom"):
            flags += ["--zoom"] + ([] if req["zoom"] is True else [str(req["zoom"])])
        flags = Manager._still_flags(d, req, flags)
        flags = Manager._polaroid_flags(d, req, flags)
        flags = Manager._audio_flags(req, flags)
        return words + (flags if words else ["--"] + flags)

    @staticmethod
    def _name_flag(project, req):
        if req.get("no_music"):
            kind = "nomusic"
        else:
            kind = "edit" if req.get("beat_sync", True) else "music"
        name = req.get("out_name") or f"{project}_{kind}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        name = Path(name).name
        if not name.endswith(".mp4"):
            name += ".mp4"
        return ["--out-name", name]  # make_edit writes it to <folder>/edits/

    @staticmethod
    def _mute_flags(req):
        muted = [int(i) for i in (req.get("mute_clip_audio") or [])]
        return ["--mute-clip-audio", *map(str, muted)] if muted else []

    def submit(self, project, action, req):
        argv = self.build_argv(project, action, req)
        job = Job("", project, action, argv, req)
        with self.lock:  # one submitter per process; the store's insert is atomic across processes
            for j in self.active():
                if j.key() == job.key():
                    raise ApiError(f"{action} already {j.status} as {j.id}", 409)
            return Job.from_data(jobstore.insert(job.to_data()))

    @staticmethod
    def _held() -> list:
        return [Job.from_data(d) for d in jobstore.by_status("held")]

    def add_to_batch(self, project, action, req):
        """The web UI's armed "Start batch" (RunPod target): animate clicks, from any project, wait as `held` jobs (one per
        project, action and folder; a click on the same one adds its stages) and are released together once the ⚙ batch
        size is reached, so the whole batch shares one pod start. Released, they run one after another in the runpod lane."""
        if action not in ANIMATE:
            raise ApiError(f"{action} can't be batched", 400)
        stages = [int(s) for s in req.get("stages") or []]
        if not stages:
            raise ApiError("a batch needs stage ids", 400)
        folder = req.get("folder") or ""
        with self.lock:
            same = next((j for j in self._held() if (j.project, j.action, j.req.get("folder") or "") == (project, action, folder)), None)
            stages = [*dict.fromkeys([*(same.req["stages"] if same else []), *stages])]  # keep order, drop repeats
            r = {"action": action, "stages": stages, "folder": folder}
            argv = self.build_argv(project, action, r)  # validated now: a bad click is refused, not held
            if same:
                same.req, same.argv = r, argv
                jobstore.save(same.to_data())
            else:
                job = Job("", project, action, argv, r)
                job.status = "held"
                jobstore.insert(job.to_data())
            total = sum(len(j.req["stages"]) for j in self._held())
        size = config.animate_batch_size()
        if total < size:
            return {"job": None, "pending": total, "size": size}
        sent = self.flush_batch()["sent"]
        job = next((j for j in sent if j["project"] == project and j["action"] == action and (j["req"].get("folder") or "") == folder), None)
        return {"job": job, "pending": 0, "size": size}

    def flush_batch(self):
        """Release every held job now, even below the batch size (one already queued or running the same way is dropped)."""
        with self.lock:
            busy = {j.key() for j in self.active()}
            held = self._held()
            for j in held:
                if j.key() in busy:
                    jobstore.cancel_held(j.id)
            jobstore.release_held()
        return {"sent": [self.get(j.id).info() for j in held], "pending": []}

    def clear_batch(self):
        jobstore.cancel_held()

    def batch_state(self):
        return [{"project": j.project, "action": j.action, "folder": j.req.get("folder") or "", "stages": j.req["stages"], "added": j.created}
                for j in sorted(self._held(), key=lambda j: (j.project, j.action, j.req.get("folder") or ""))]

    @staticmethod
    def target_of(job):
        """Where the job's prompts are queued: the local ComfyUI it ran on, the pod (if it is still there), or Comfy Cloud."""
        if job.lane == "runpod":
            pod = runpod_pod.find()
            return targets.get("local", server=runpod_pod.url(pod["id"])) if pod else None
        if job.lane == "local":
            return targets.get("local", server=job.req.get("server") or None)
        return targets.get("cloud")

    def cancel_stage(self, jid, tag):
        """Stop one stage of a running job (tag "5", or "5.t2" for a take): the job's render loop skips it (it reads the
        job's cancel file, jobstore.cancel_path) and its prompt comes off the backend's queue now; the other stages go on."""
        job = self.get(jid)
        if not job:
            raise ApiError("no such job", 404)
        if job.status != "running":
            raise ApiError(f"{jid} is {job.status}: only a running job's stages can be cancelled one by one", 400)
        if job.stage_status.get(tag) in ("done", "failed", "cancelled"):
            raise ApiError(f"stage {tag} already {job.stage_status[tag]}", 400)
        with open(jobstore.cancel_path(jid), "a") as f:
            f.write(f"stage_{tag}\n")
        jobstore.append_log(jid, f"stage {tag}: cancel requested")
        pid = job.stage_prompts.get(tag)
        if pid:
            threading.Thread(target=self._drop_prompt, args=(job, pid), daemon=True).start()
        return {"ok": True}

    def _drop_prompt(self, job, pid):
        try:
            target = self.target_of(job)
            if target:
                target.cancel([pid])
        except Exception as e:  # noqa: BLE001  # the render loop still skips the stage
            jobstore.append_log(job.id, f"could not pull its prompt off the queue: {e}")

    def cancel(self, jid):
        """A failed job waiting for its auto-retry: stop that; queued: cancelled at once; running: the daemon kills it and
        clears what it queued on the backend."""
        job = self.get(jid)
        if not job:
            raise ApiError("no such job", 404)
        if job.retry_at is not None:
            if jobstore.take_retry(jid):
                jobstore.append_log(jid, "auto-retry cancelled")
        elif job.status == "queued":
            if not jobstore.cancel_queued(jid):  # the daemon claimed it just now: it is running
                jobstore.request_cancel(jid)
        elif job.status == "running":
            jobstore.request_cancel(jid)
        return self.get(jid)

    def retry(self, jid):
        """Submit a failed/cancelled job again with the same request (argv rebuilt from today's settings). A job naming its
        stages only reruns the ones that didn't finish, so the stages it already rendered aren't paid for twice; with none
        left unfinished (or no stage list) the request goes out unchanged."""
        job = self.get(jid)
        if not job:
            raise ApiError("no such job", 404)
        if job.status not in ("failed", "cancelled"):
            raise ApiError(f"only a failed or cancelled job can be retried ({jid} is {job.status})", 400)
        jobstore.take_retry(jid)  # by hand, or the auto-retry itself: once
        req = dict(job.req)
        stages = [int(s) for s in req.get("stages") or ()]
        if stages and job.action != "edit":  # an edit's stages pick the clips of one cut: always all of them
            done = {k.split(".")[0] for k, v in job.stage_status.items() if v in ("done", "cancelled")}  # cancelled: on purpose
            failed = {k.split(".")[0] for k, v in job.stage_status.items() if v == "failed"}
            left = [s for s in stages if str(s) in failed or str(s) not in done]
            if not left:
                raise ApiError(f"{jid}: every stage finished or was cancelled; nothing to retry", 400)
            req["stages"] = left
        return self.submit(job.project, job.action, req)

    @staticmethod
    def list():
        """The jobs panel: everything but the held batch (the batch bar shows that)."""
        return [Job.from_data(d).info() for d in jobstore.recent() if d["status"] != "held"]


manager = Manager()
