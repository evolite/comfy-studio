"""The queue daemon (the comfy-queue user service): runs the jobs the web server puts in jobstore.

Each lane (cloud, local, runpod) runs one job at a time, oldest first; `free` jobs start at once. A job runs as
`bash -c '<project.py argv> >> <log> 2>&1; echo $? > <rc>'` in its own session, so it outlives this daemon: on start the
daemon reattaches to every job still marked running (wrapper alive: keep reading its log from the stored offset; gone: its
exit code from the rc file, else failed). While a job runs, the daemon reads its log as it grows, parses it (jobs.Job.emit)
and stores the state (at most every SAVE_EVERY s). It also kills jobs the server asked to cancel (and clears what they
queued on the backend), submits a failed lane job again RETRY_AFTER later (up to RETRY_TIMES), copies finished edits to
config.PUBLISH_DIR and pushes the 🔔 notification when the queue empties.

    python3 webui/server/queue_daemon.py
"""
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: config.py
import config  # noqa: E402
import jobstore  # noqa: E402
from jobs import RETRY_AFTER, RETRY_TIMES, Job, edit_kind, manager, save_edit_timing, send_pushover  # noqa: E402
from projects import ROOT, ApiError, project_dir  # noqa: E402
import runpod_pod  # noqa: E402  # pipeline/, on sys.path via jobs
import targets  # noqa: E402

TICK = 0.5          # seconds between looks at the store (new jobs, cancels, due retries)
SAVE_EVERY = 0.5    # a running job's state is stored at most this often (and always when it ends)
LANES = ("cloud", "local", "runpod")
WRAP = '"${@:3}" >> "$1" 2>&1; echo $? > "$2"'  # bash -c WRAP job <log> <rc> <argv...>


class Daemon:
    def __init__(self):
        self.live = {}  # job id -> Job this daemon follows
        self.lock = threading.Lock()

    # --- the loop ---------------------------------------------------------------------------------------------------
    def run(self):
        jobstore.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.reattach()
        print(f"comfy-queue: watching {jobstore.DB}", flush=True)
        while True:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001  # one bad row must not stop the queue
                print(f"comfy-queue: tick failed: {e!r}", file=sys.stderr, flush=True)
            time.sleep(TICK)

    def tick(self):
        for d in jobstore.by_status("queued"):
            if d["lane"] != "free" and self._lane_busy(d["lane"]):
                continue
            if jobstore.claim(d["id"]):
                self.start(Job.from_data(d))
        with self.lock:
            live = list(self.live.values())
        for job in live:
            if not job.cancel_requested and (jobstore.get(job.id) or {}).get("cancel"):
                self.kill(job)
        for d in jobstore.due_retries():
            self.auto_retry(d)
        self.maybe_notify()

    def _lane_busy(self, lane):
        with self.lock:
            return any(j.lane == lane for j in self.live.values())

    # --- one job --------------------------------------------------------------------------------------------------------
    def start(self, job):
        job.status, job.started, job.log_offset = "running", time.time(), 0
        jobstore.rc_path(job.id).unlink(missing_ok=True)
        jobstore.append_log(job.id, "$ " + " ".join(job.argv[1:]))
        job.proc = subprocess.Popen(["bash", "-c", WRAP, "job", str(jobstore.log_path(job.id)), str(jobstore.rc_path(job.id)), *job.argv],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
        job.pid = job.proc.pid
        jobstore.save(job.to_data())
        self.follow(job)

    def follow(self, job):
        with self.lock:
            self.live[job.id] = job
        threading.Thread(target=self._follow, args=(job,), daemon=True).start()

    def _follow(self, job):
        saved = 0.0
        try:
            while True:
                ended = not self._alive(job)  # checked before reading, so the last read gets everything it wrote
                self._read(job)
                if ended:
                    break
                if time.time() - saved >= SAVE_EVERY:
                    jobstore.save(job.to_data())
                    saved = time.time()
                time.sleep(0.4)
        except Exception as e:  # noqa: BLE001
            self.note(job, f"queue error: {e}")
        self.finish(job)

    @staticmethod
    def _read(job):
        lines, job.log_offset = jobstore.read_log(job.id, job.log_offset)
        for line in lines:
            job.emit(line)

    @staticmethod
    def _alive(job):
        if job.proc is not None:
            return job.proc.poll() is None
        try:  # a job a previous daemon started: its wrapper, if that pid is still it
            return str(jobstore.log_path(job.id)) in Path(f"/proc/{job.pid}/cmdline").read_bytes().decode(errors="replace")
        except (OSError, TypeError):
            return False

    def finish(self, job):
        try:
            rc = int(jobstore.rc_path(job.id).read_text().strip())
        except (OSError, ValueError):
            rc = None
        job.returncode = rc
        if job.cancel_requested:
            job.status = "cancelled"
        elif rc is None:
            job.status = "failed"
            self.note(job, "the job's process ended without an exit code (killed?)")
        else:
            job.status = "done" if rc == 0 else "failed"
        job.ended = time.time()
        if job.status == "done" and job.action == "edit":
            save_edit_timing(edit_kind(job.req), job.ended - job.started, job.frames)
            self.publish_edit(job)
            self._read(job)
        if job.status == "failed" and job.lane in LANES and job.attempt <= RETRY_TIMES:
            job.retry_at = time.time() + RETRY_AFTER
            self.note(job, f"auto-retry at {time.strftime('%H:%M', time.localtime(job.retry_at))} (attempt {job.attempt + 1} of {RETRY_TIMES + 1})")
            jobstore.set_retry(job.id, job.retry_at)
        jobstore.save(job.to_data())
        with self.lock:
            self.live.pop(job.id, None)

    @staticmethod
    def note(job, line):
        """A line of the daemon's own, once the job's output has been read: into the log and the state."""
        jobstore.append_log(job.id, line)
        job.log_offset = jobstore.read_log(job.id, job.log_offset)[1]
        job.emit(line)

    def kill(self, job):
        job.cancel_requested = True
        try:  # the whole group: project.py's children (animate_pipeline.py...) keep submitting and hold the lane otherwise
            os.killpg(job.pid, signal.SIGTERM)
        except (ProcessLookupError, TypeError):
            pass
        threading.Thread(target=self._cancel_remote, args=(job,), daemon=True).start()

    @staticmethod
    def auto_retry(d):
        if not jobstore.take_retry(d["id"]):
            return  # cancelled or retried by hand meanwhile
        try:
            new = manager.retry(d["id"])
        except ApiError as e:  # e.g. the same request already queued again by hand
            jobstore.append_log(d["id"], f"auto-retry skipped: {e}")
            return
        n = jobstore.get(new.id)
        n["attempt"] = d.get("attempt", 1) + 1
        jobstore.save(n)
        jobstore.append_log(d["id"], f"auto-retried as {new.id}")

    # --- start-up ----------------------------------------------------------------------------------------------------
    def reattach(self):
        """Jobs a previous daemon left running: follow the live ones from where their state was stored, close the rest."""
        for d in jobstore.by_status("running"):
            job = Job.from_data(d)
            if self._alive(job):
                print(f"comfy-queue: reattached {job.id} (pid {job.pid})", flush=True)
                self.follow(job)
            else:
                self._read(job)
                self.finish(job)

    # --- notifications -------------------------------------------------------------------------------------------------
    @staticmethod
    def maybe_notify():
        since = jobstore.setting("notify_armed_at")
        if since is None or jobstore.by_status(*jobstore.ACTIVE):
            return
        reported = set(jobstore.setting("notify_reported", []))
        ended = [d for d in jobstore.recent() if d["status"] in ("done", "failed", "cancelled")
                 and (d.get("ended") or time.time()) >= since and d["id"] not in reported]
        if not ended:
            return  # idle with nothing new since the last push: wait for the next jobs
        jobstore.set_setting("notify_reported", sorted(reported | {d["id"] for d in ended}))
        counts = {st: sum(d["status"] == st for d in ended) for st in ("done", "failed", "cancelled")}
        msg = "Queue empty — " + ", ".join(f"{n} {st}" for st, n in counts.items() if n)
        msg += "".join(f"\n✗ {d['id']} {d['action']}" for d in ended if d["status"] == "failed")  # never the project name: pushes show on the lock screen
        threading.Thread(target=send_pushover, args=("Comfy Studio", msg), daemon=True).start()

    # --- edits and cancels (moved from the in-server manager) --------------------------------------------------------
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
            jobstore.append_log(job.id, f"published -> {dst}")
        except Exception as e:  # noqa: BLE001
            jobstore.append_log(job.id, f"publish failed: {e}")

    def _cancel_remote(self, job):
        """Killing project.py leaves what it already queued on the backend: the job's target drops those prompts."""
        try:
            target = self.target_of(job)
            n = target.cancel(job.prompt_ids) if target and job.prompt_ids else 0
            jobstore.append_log(job.id, f"cancelled: removed {n} job(s) from the ComfyUI queue" if n else "cancelled: nothing left in the ComfyUI queue")
        except Exception as e:  # noqa: BLE001
            jobstore.append_log(job.id, f"cancelled, but could not clear the ComfyUI queue: {e}")

    @staticmethod
    def target_of(job):
        """Where the job's prompts are queued: the local ComfyUI it ran on, the pod (if it is still there), or Comfy Cloud."""
        if job.lane == "runpod":
            pod = runpod_pod.find()
            return targets.get("local", server=runpod_pod.url(pod["id"])) if pod else None
        if job.lane == "local":
            return targets.get("local", server=job.req.get("server") or None)
        return targets.get("cloud")

if __name__ == "__main__":
    Daemon().run()
