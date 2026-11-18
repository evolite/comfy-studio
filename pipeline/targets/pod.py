"""The RunPod pod as a target: a ComfyUI at the pod's URL (so a local target there), started on demand by
pipeline/runpod_pod.py. What differs from this machine's ComfyUI: the pod starts when a run needs it (`start`), its H3
graphs load the distilled UNet (config.runpod_h3_unet()), the model check also covers UNet/CLIP/VAE names, nothing is
unloaded between batches, a clip may wait longer in its queue, and clip timings go into the pod's start record instead
of the baselines (another GPU than the cloud's)."""
import json

import config
import runpod_pod

from .base import StageJob, TargetUnavailable
from .local import NAME_FIELDS, LocalTarget, _call, check

CHECK_FIELDS = NAME_FIELDS + ("unet_name", "clip_name", "vae_name")  # model inputs checked against the pod


def pod_unet(graph: dict, unet: str) -> None:
    """On the pod, H3 graphs load `unet` (a distilled turbo finetune) in place of the stock UNet; Comfy Cloud keeps the stock one.
    It is distilled already: the turbo switch's LoRA branch is skipped (still 8 steps) and the sampler is euler (author's pick)."""
    for nid, node in graph.items():
        if node.get("class_type") == "UNETLoader" and node["inputs"].get("unet_name") == config.STOCK_H3_UNET:
            node["inputs"]["unet_name"] = unet
            for other in graph.values():  # the ComfySwitchNode picking UNet vs UNet + turbo LoRA
                if other.get("class_type") == "ComfySwitchNode" and other["inputs"].get("on_false") == [nid, 0]:
                    other["inputs"]["on_true"] = [nid, 0]
        elif node.get("class_type") == "KSamplerSelect":
            node["inputs"]["sampler_name"] = "euler"


class PodTarget(LocalTarget):
    name = "runpod"
    lane = "runpod"
    timeout = 3600.0

    def __init__(self, big: bool = False, start: bool = True):
        """big = a clip needs the 96 GB pod; start=False (a dry run) leaves the pod alone: graphs are built, not checked."""
        self.big, self.server, self._oi = big, None, None
        if start:
            try:
                self.server = runpod_pod.ensure_up(lambda m: print(m, flush=True), big).rstrip("/")
            except runpod_pod.PodError as e:
                raise TargetUnavailable(f"RunPod: {e}") from e

    def _unreachable(self, e, what="can't reach"):
        return TargetUnavailable(f"{what} the RunPod pod at {self.server} ({getattr(e, 'reason', e)})")

    def available(self) -> bool:
        return bool(self.server) and super().available()

    def prepare(self, graph: dict) -> dict:
        if unet := config.runpod_h3_unet():
            pod_unet(graph, unet)
        return graph

    def check(self, graph: dict) -> list:
        if not self.server:
            return []
        if self._oi is None:
            self._oi = json.loads(_call(self.server, "/object_info"))
        return check(graph, self.server, CHECK_FIELDS, self._oi)

    def before_batch(self, jobs: list) -> bool:
        return False

    def finished(self, job: StageJob) -> None:
        runpod_pod.record_clip(self.server, job.seconds)
