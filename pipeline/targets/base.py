"""A target is where a graph runs: Comfy Cloud or a local ComfyUI. Steps (pipeline/steps.py) are written once, in cloud-canonical
graph form, against this interface; the runner (pipeline/runner.py) is the only loop that drives it.

A target turns a StageJob into its saved outputs (a still, a clip):
    graph = target.prepare(canonical_graph)      # identity for the cloud; model/node names for a local machine
    target.check(graph)                          # problems it would reject, [] if fine
    target.before_batch(jobs)                    # e.g. unload models the next batch doesn't need
    target.submit(job) for every job             # all queued up front, so the whole queue is visible
    for job in target.collect(jobs): ...         # yields each job as it finishes (job.final set, or job.error)
    target.cancel(prompt_ids)                    # pull them off the queue (interrupt the running one): the job queue's cancel

Targets: cloud (Comfy Cloud), local (a ComfyUI: this machine, or any URL), runpod (the pod: a local target at its URL).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

CANCELLED = "cancelled by you"  # a StageJob's error when the job queue cancelled just that stage


class TargetUnavailable(RuntimeError):
    """The target can't be reached (local ComfyUI down, network gone); the message is meant for the user."""


def cancelled(tag: str) -> bool:
    """A stage the job queue asked to skip: its tag is a line in $STUDIO_CANCEL_FILE (the web server appends to it; the server
    has also pulled the stage's prompt off the target's queue). Targets check it while they wait."""
    path = os.environ.get("STUDIO_CANCEL_FILE")
    try:
        return bool(path) and tag in Path(path).read_text().split()
    except OSError:
        return False


@dataclass
class StageJob:
    n: int                        # stage number
    graph: dict                   # prepared for the target
    label: str                    # shown in logs and the local dashboard
    debug_dir: Path               # <out>/_debug/stage_N: graph, raw job data, final.png
    final_node: str = "436"
    node_labels: dict = field(default_factory=dict)
    handle: object = None         # target-specific (prompt id, restart count, ...)
    final: Path | None = None     # debug_dir/final.<ext> once done (the final node's first file: a .png still, an .mp4 clip)
    tag: str = ""                 # "stage_N" (or "stage_N.tK"): the name in logs and ::event lines
    keep: tuple = ()              # output nodes besides final_node whose files are saved (e.g. animate's RIFE clip)
    outputs: list = field(default_factory=list)  # [(node_id, path)] of every saved output file
    error: str | None = None
    cold: bool = False            # the first job after models were unloaded also pays the load
    waited: float = 0.0
    seconds: float | None = None  # ComfyUI's own processing time, when known


class Target:
    name = ""
    lane = ""                     # "cloud" | "local": jobs of one lane run one batch at a time in the web UI
    params: dict = {}             # per-target values a step needs when building its graph (e.g. SeedVR model/colour)

    def available(self) -> bool:
        raise NotImplementedError

    def prepare(self, graph: dict) -> dict:
        return graph

    def check(self, _graph: dict) -> list:
        return []

    def upload(self, path: Path, name: str = None) -> str:
        raise NotImplementedError

    def before_batch(self, _jobs: list) -> bool:
        """Get ready for a batch; returns whether the first job will be cold (models were unloaded)."""
        return False

    def submit(self, job: StageJob) -> None:
        raise NotImplementedError

    def collect(self, jobs: list):
        raise NotImplementedError

    def pending(self, job: StageJob) -> bool:
        """Whether the job still waits in the target's queue (not started): the runner may then swap its graph for an edited one.
        False where it can't tell (the cloud)."""
        return False

    def cancel(self, prompt_ids: list) -> int:
        """Remove these prompts from the target's queue, interrupting the one running; returns how many it was asked to drop."""
        raise NotImplementedError
