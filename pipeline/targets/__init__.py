"""Where graphs run. `get(name)` -> a Target; steps and the runner never import a concrete target."""
from .base import StageJob, Target, TargetUnavailable
from .cloud import CloudTarget
from .local import LocalTarget
from .pod import PodTarget

NAMES = ("cloud", "local", "runpod")


def get(name: str, **kw) -> Target:
    if name == "cloud":
        return CloudTarget(**kw)
    if name == "local":
        return LocalTarget(**kw)
    if name == "runpod":
        return PodTarget(**kw)
    raise ValueError(f"unknown target {name!r}; choose from {NAMES}")
