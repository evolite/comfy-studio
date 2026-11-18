"""The durable job store, shared by the web server (submits, cancels, reads) and the queue daemon (runs the jobs).

SQLite at config.DATA_DIR/jobs.db in WAL mode, so both processes read while one writes. One row per job: its full state as
JSON in `data`, plus the columns the two sides change independently (`status`, `retry_at`, `cancel`), so neither
overwrites the other's change: the server only flips queued -> cancelled, sets `cancel` on a running job and clears
`retry_at`; the daemon claims queued -> running with a conditional update and owns the row from then on.
Each job's output goes to LOG_DIR/<id>.log, written by the job's own process (so it survives a daemon restart) and read
by both sides. `settings` holds the 🔔 notify state. Status `held` = a "Start batch" click waiting for the batch to fill
(the daemon never runs it; jobs.Manager releases them to `queued` together).
"""
import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: config.py
import config  # noqa: E402

DB = config.DATA_DIR / "jobs.db"
LOG_DIR = config.LOGS_DIR / "jobs"
ACTIVE = ("queued", "running")
_local = threading.local()


def conn() -> sqlite3.Connection:
    """This thread's connection (sqlite3 connections are per thread); autocommit, with explicit BEGIN IMMEDIATE where needed."""
    c = getattr(_local, "c", None)
    if c is None or getattr(_local, "path", None) != DB:
        DB.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(DB, timeout=15, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=15000")
        c.execute("CREATE TABLE IF NOT EXISTS jobs (seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, status TEXT NOT NULL,"
                  " lane TEXT NOT NULL, project TEXT NOT NULL, retry_at REAL, cancel INTEGER NOT NULL DEFAULT 0, data TEXT NOT NULL)")
        c.execute("CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status)")
        c.execute("CREATE TABLE IF NOT EXISTS settings (k TEXT PRIMARY KEY, v TEXT)")
        try:  # when the row last changed: version() is what the web UI's live updates watch
            c.execute("ALTER TABLE jobs ADD COLUMN updated REAL NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # already there
        _local.c, _local.path = c, DB
    return c


def log_path(jid: str):
    return LOG_DIR / f"{jid}.log"


def rc_path(jid: str):
    """Where the job's wrapper writes the exit code once project.py ends (read by a daemon that didn't start it)."""
    return LOG_DIR / f"{jid}.rc"


def _row(r) -> dict | None:
    if r is None:
        return None
    d = json.loads(r["data"])
    d.update(id=r["id"], seq=r["seq"], status=r["status"], retry_at=r["retry_at"], cancel=bool(r["cancel"]))
    return d


def _query(sql, args=()):
    c = conn()
    c.row_factory = sqlite3.Row
    return c.execute(sql, args)


def insert(data: dict) -> dict:
    """A new queued job: assigns seq and id (j<seq>) atomically; returns the stored state."""
    c = conn()
    c.execute("BEGIN IMMEDIATE")
    try:
        seq = (c.execute("SELECT COALESCE(MAX(seq), 0) FROM jobs").fetchone()[0]) + 1
        jid = f"j{seq}"
        data = {**data, "id": jid}
        c.execute("INSERT INTO jobs (seq, id, status, lane, project, retry_at, cancel, data, updated) VALUES (?, ?, ?, ?, ?, NULL, 0, ?, ?)",
                  (seq, jid, data.get("status", "queued"), data["lane"], data["project"], json.dumps(data), time.time()))
        c.execute("COMMIT")
    except BaseException:
        c.execute("ROLLBACK")
        raise
    return get(jid)


def save(data: dict) -> None:
    """The daemon's write of a job it owns: state + status (retry_at and cancel stay the other side's)."""
    keep = {k: v for k, v in data.items() if k not in ("seq", "retry_at", "cancel")}
    conn().execute("UPDATE jobs SET status = ?, data = ?, updated = ? WHERE id = ?", (data["status"], json.dumps(keep), time.time(), data["id"]))


def get(jid: str) -> dict | None:
    return _row(_query("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone())


def recent(limit: int = 200) -> list:
    """The newest jobs first (the jobs panel's list), and every active job even when older."""
    rows = _query("SELECT * FROM jobs WHERE seq > (SELECT COALESCE(MAX(seq), 0) FROM jobs) - ? OR status IN ('queued', 'running')"
                  " ORDER BY seq DESC", (limit,)).fetchall()
    return [_row(r) for r in rows]


def by_status(*statuses) -> list:
    marks = ",".join("?" * len(statuses))
    return [_row(r) for r in _query(f"SELECT * FROM jobs WHERE status IN ({marks}) ORDER BY seq", statuses).fetchall()]


def claim(jid: str) -> bool:
    """queued -> running, only if nobody cancelled it first."""
    return conn().execute("UPDATE jobs SET status = 'running', updated = ? WHERE id = ? AND status = 'queued'", (time.time(), jid)).rowcount == 1


def cancel_queued(jid: str) -> bool:
    return conn().execute("UPDATE jobs SET status = 'cancelled', updated = ? WHERE id = ? AND status = 'queued'", (time.time(), jid)).rowcount == 1


def release_held() -> None:
    """The "Start batch" jobs go into the queue, all at once."""
    conn().execute("UPDATE jobs SET status = 'queued', updated = ? WHERE status = 'held'", (time.time(),))


def cancel_held(jid: str | None = None) -> None:
    """Drop one held job, or every one."""
    sql, args = ("UPDATE jobs SET status = 'cancelled', updated = ? WHERE status = 'held'", (time.time(),))
    if jid:
        sql, args = sql + " AND id = ?", (*args, jid)
    conn().execute(sql, args)


def request_cancel(jid: str) -> None:
    conn().execute("UPDATE jobs SET cancel = 1, updated = ? WHERE id = ?", (time.time(), jid))


def set_retry(jid: str, at: float | None) -> None:
    conn().execute("UPDATE jobs SET retry_at = ?, updated = ? WHERE id = ?", (at, time.time(), jid))


def take_retry(jid: str) -> bool:
    """Clear a pending auto-retry; True when this call is the one that cleared it (so a retry happens once)."""
    return conn().execute("UPDATE jobs SET retry_at = NULL, updated = ? WHERE id = ? AND retry_at IS NOT NULL", (time.time(), jid)).rowcount == 1


def version() -> tuple:
    """(when any job last changed, how many are queued or running): the live updates send "jobs" when it moves."""
    r = conn().execute("SELECT COALESCE(MAX(updated), 0), SUM(status IN ('queued', 'running')) FROM jobs").fetchone()
    return r[0], r[1] or 0


def due_retries(now: float | None = None) -> list:
    return [_row(r) for r in _query("SELECT * FROM jobs WHERE retry_at IS NOT NULL AND retry_at <= ? ORDER BY seq",
                                    (now or time.time(),)).fetchall()]


def setting(k: str, default=None):
    r = conn().execute("SELECT v FROM settings WHERE k = ?", (k,)).fetchone()
    return json.loads(r[0]) if r else default


def set_setting(k: str, v) -> None:
    conn().execute("INSERT INTO settings (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", (k, json.dumps(v)))


def read_log(jid: str, start: int = 0) -> tuple[list, int]:
    """The log's complete lines from byte offset `start`, and the offset after them (a partial last line waits)."""
    try:
        with open(log_path(jid), "rb") as f:
            f.seek(start)
            chunk = f.read()
    except FileNotFoundError:
        return [], start
    end = chunk.rfind(b"\n") + 1
    return chunk[:end].decode(errors="replace").splitlines(), start + end


def append_log(jid: str, line: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path(jid), "a") as f:
        f.write(line + "\n")
