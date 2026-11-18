"""Machine-readable progress for the job queue (webui/server/jobs.py reads it): one `::event {json}` line printed next to
the human-readable one, so changing a log message can't break the board.

    stage(tag, status, **fields)   tag "stage_5" / "stage_5.t2" (a preview take) / 5; status queued | running |
                                   submitted | done | failed | skipped; fields: expected / processed (s), prompt_id, error
    note(kind, **fields)           job-level facts: memfree (reason), frames (n)
"""
import json

PREFIX = "::event "


def _emit(d: dict) -> None:
    print(PREFIX + json.dumps(d), flush=True)


def stage(tag, status: str, **fields) -> None:
    sid = str(tag).removeprefix("stage_")
    _emit({"stage": sid, "status": status, **{k: v for k, v in fields.items() if v is not None}})


def note(kind: str, **fields) -> None:
    _emit({"note": kind, **fields})
