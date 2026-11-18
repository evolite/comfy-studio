"""Live updates for the web UI (GET /api/events, server-sent events): the UI refreshes a query when the server says it
changed, instead of polling for it.

    {"type": "jobs"}                 the job store changed; while a job runs, also every ACTIVE_TICK s (its progress bar moves)
    {"type": "batch"}                the "Start batch" queue changed
    {"type": "project", "name": n}   a file under data/pipelines/<n>/ changed (stills, clips, edits, project.json; not _debug/)
    {"type": "projects"}             a project folder appeared or went away, or a project.json changed (the project list)

One watcher thread (watchfiles) serves every open page; each stream checks it and the job store every STEP s.
"""
import asyncio
import json
import threading
import time
from collections import deque
from pathlib import Path

import watchfiles

import config
import jobstore

STEP = 0.5          # seconds between checks per open page
ACTIVE_TICK = 2.0   # a "jobs" refresh at least this often while a job is queued or running
PING = 15.0         # an SSE comment this often keeps proxies and the browser from dropping an idle stream


class ProjectWatch:
    """Changes under the pipelines folder, as numbered events: an open page asks for the ones after the last it saw."""

    def __init__(self, root: Path):
        self.root = root
        self.rev = 0
        self.log = deque(maxlen=1000)  # (rev, type, name)
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, daemon=True, name="project-watch")
                self.thread.start()

    def _run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for changes in watchfiles.watch(self.root, debounce=400, step=100):
            projects, listing = set(), False
            for _, path in changes:
                try:
                    parts = Path(path).relative_to(self.root).parts
                except ValueError:
                    continue
                if not parts or "_debug" in parts or parts[-1].endswith((".part", ".tmp")) or parts[-1].startswith("."):
                    continue
                projects.add(parts[0])
                listing |= len(parts) == 1 or parts[-1] == "project.json"
            with self.lock:
                for name in sorted(projects):
                    self.rev += 1
                    self.log.append((self.rev, "project", name))
                if listing:
                    self.rev += 1
                    self.log.append((self.rev, "projects", None))

    def since(self, rev: int) -> tuple[int, list]:
        with self.lock:
            return self.rev, [(t, n) for r, t, n in self.log if r > rev]


WATCH = ProjectWatch(config.PIPELINES_DIR)


async def stream(request, manager):
    """The SSE body for one open page."""
    WATCH.start()
    rev, batch = WATCH.rev, manager.batch_rev
    jobs_v, ticked, pinged = None, 0.0, time.time()
    yield "retry: 3000\n\n"
    while not await request.is_disconnected():
        out = []
        v, active = jobstore.version()
        now = time.time()
        if v != jobs_v or (active and now - ticked >= ACTIVE_TICK):
            out.append({"type": "jobs"})
            jobs_v, ticked = v, now
        if manager.batch_rev != batch:
            batch = manager.batch_rev
            out.append({"type": "batch"})
        rev, changes = WATCH.since(rev)
        for t, n in dict.fromkeys(changes):  # each project once per step
            out.append({"type": t, "name": n} if n else {"type": t})
        for e in out:
            yield f"data: {json.dumps(e)}\n\n"
        if not out and now - pinged >= PING:
            yield ": ping\n\n"
            pinged = now
        elif out:
            pinged = now
        await asyncio.sleep(STEP)
