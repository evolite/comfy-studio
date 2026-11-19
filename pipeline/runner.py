"""The one loop that runs a step on a target: build each stage's graph, prepare it for the target, validate every stage before
queuing any, queue them all, collect as they finish, save into the step's folder. Returns the stages that failed."""
import json
import secrets
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import events  # noqa: E402
import history  # noqa: E402
import image_loras  # noqa: E402
import takes as takes_mod  # noqa: E402
from targets.base import CANCELLED, StageJob, TargetUnavailable  # noqa: E402


def _needs_of(step, ctx, numbers) -> dict:
    """The step's input folder per stage ({} when it starts from text); exits naming stages whose input file is missing."""
    needs = {n: step.needs(ctx, n) if callable(step.needs) else step.needs for n in numbers} if step.needs else {}
    missing = [n for n, f in needs.items() if not (ctx.d / f / step.input.format(n=n)).exists()]
    if missing:
        sys.exit(f"stages {missing}: no {'/'.join(sorted({needs[n] for n in missing}))}/{step.input.format(n='N')} yet -- "
                 + step.needs_hint.format(name=ctx.name, stages=" ".join(map(str, missing))))
    return needs


def _upload_needs(step, target, ctx, numbers, needs, dry_run) -> dict:
    """The step's input image per stage, uploaded to the target ("x" placeholders on a dry run)."""
    images = {}
    for n in numbers:
        if n in needs:
            path = ctx.d / needs[n] / step.input.format(n=n)
            images[n] = "x" if dry_run else target.upload(path, f"{step.name}_{ctx.name}_{n}{path.suffix}")
    return images


def _build_jobs(step, target, ctx, numbers, images, out, final_node, labels, takes) -> list:
    """One StageJob per stage (per take): fresh seed per take, graph prepared for the target."""
    return [_build_job(step, target, ctx, n, k if takes > 1 else None, images, out, final_node, labels)
            for k in range(1, takes + 1) for n in numbers]


def _build_job(step, target, ctx, n, take, images, out, final_node, labels) -> StageJob:
    """Stage n's job; take is its number when the run makes several (each with a fresh seed), else None."""
    stage = {**ctx.stages[n], "id": n, **({"seed": secrets.randbits(48)} if take else {})}
    graph = target.prepare(step.graph(ctx, stage, images.get(n), target))
    tag = f"stage_{n}" + (f".t{take}" if take else "")
    label = f"{ctx.name} stage {n}" + (f" {step.suffix}" if step.suffix else "") + (f" t{take}" if take else "")
    final = step.final_for(ctx, stage) if step.final_for else final_node
    j = StageJob(n, graph, label, out / "_debug" / (step.debug_prefix + tag), final, labels, tag=tag, keep=step.keep)
    j.stage = stage
    return j


def _report_dry_run(step, target, jobs, problems) -> None:
    for j in jobs:
        print(f"  {j.tag}: {step.name} graph on {target.name} " + ("OK" if j.n not in problems else f"PROBLEMS {problems[j.n]}") + f", {len(j.graph)} nodes")
    if problems:
        sys.exit(f"\nthe {target.name} target can't run stages {sorted(problems)}")


def _submit_all(step, target, ctx, numbers, reroll, out, jobs) -> None:
    """Archive the files a reroll replaces, take in existing previews, free models, then queue the whole batch."""
    if reroll and step.name == "main":
        for n in numbers:
            history.archive(ctx.d, ctx.d / f"stage_{n}.png")
    out.mkdir(parents=True, exist_ok=True)
    if step.takes:  # take in the existing preview (and its record from _debug) before this run overwrites _debug/stage_N
        for n in numbers:
            takes_mod.sync(ctx.d, n)
    target.before_batch(jobs)
    print(f"=== {step.name} on {target.name}: {len(jobs)} stage(s) -> {out} ===", flush=True)
    for j in jobs:
        target.submit(j)


def _collect_one(ctx, step, needs, out, j, got, target_name: str) -> None:
    """Save one finished job — the step's own save (a clip), as a take, or over the stage file with its history entry; a
    failure just logs."""
    dest = out / f"stage_{j.n}.png"
    if j.error == CANCELLED:
        print(f"  [{j.tag}] cancelled", flush=True)
        events.stage(j.tag, "cancelled")
        return
    if j.final and step.save:
        try:
            dest = step.save(ctx, j, out)
        except RuntimeError as e:
            j.final, j.error = None, str(e)
    if not j.final:
        print(f"  [{j.tag}] FAILED: {j.error}", flush=True)
        events.stage(j.tag, "failed", error=str(j.error))
        return
    if step.save:
        pass  # saved above
    elif step.takes:
        takes_mod.sync(ctx.d, j.n)
        rec = takes_mod.record(j.graph, j.stage, ctx.common, target_name, image_loras.resolve(j.stage.get("loras")))
        takes_mod.add(ctx.d, j.n, j.final, j.graph, rec, make_current=True)
        takes_mod.clear_done(ctx.d, j.n)
    else:
        history.archive(ctx.d, dest)  # the result it replaces stays restorable (web UI: ⋯ > Previous versions)
        shutil.copyfile(j.final, dest)
        _note_source(ctx, step, needs, j.n, out)
    got[j.n] += 1
    print(f"  [{j.tag}] done (waited {j.waited:.0f}s" + (f", processed {j.seconds:.0f}s" if j.seconds else "") + f") -> {dest}", flush=True)
    events.stage(j.tag, "done", processed=round(j.seconds) if j.seconds else None)


class _EditWatch:
    """Edits made while a run waits reach the stages still queued: after each finished stage, a changed project.json
    (ctx.reload) rebuilds the graphs of the jobs the target says haven't started; one that differs is pulled off the queue
    and queued again with the new graph (same seed). Targets that can't tell what has started (the cloud) keep theirs."""

    def __init__(self, step, target, ctx, images, jobs, takes):
        self.step, self.target, self.ctx, self.images, self.jobs = step, target, ctx, images, jobs
        self.on = takes == 1 and callable(getattr(ctx, "reload", None))
        self.mtime = self._mtime()

    def _mtime(self):
        try:
            return (self.ctx.d / "project.json").stat().st_mtime
        except OSError:
            return None

    def check(self, finished: set) -> None:
        if not self.on or self._mtime() == self.mtime:
            return
        self.mtime = self._mtime()
        self.ctx.stages = self.ctx.reload()
        for j in self.jobs:
            if j.tag in finished or j.n not in self.ctx.stages or not self.target.pending(j):
                continue
            stage = {**self.ctx.stages[j.n], "id": j.n}
            graph = _same_seeds(self.target.prepare(self.step.graph(self.ctx, stage, self.images.get(j.n), self.target)), j.graph)
            if graph == j.graph:
                continue
            self.target.cancel([j.handle])
            j.graph, j.stage = graph, stage
            print(f"  [{j.tag}] prompt edited while it waited: queued again with the new graph", flush=True)
            self.target.submit(j)


def _same_seeds(graph: dict, old: dict) -> dict:
    """The rebuilt graph with the old one's seeds (a rebuild rolls a fresh random seed; that alone isn't an edit)."""
    for nid, node in graph.items():
        for k in ("seed", "noise_seed"):
            if k in node.get("inputs", {}) and k in old.get(nid, {}).get("inputs", {}):
                node["inputs"][k] = old[nid]["inputs"][k]
    return graph


def _wrap_up(step, ctx, out, numbers, got, takes, dropped=()) -> list:
    """Run the step's after-hook and print the summary; returns the failed stage numbers (a stage cancelled on its own,
    `dropped`, isn't one)."""
    failed = [n for n in numbers if not got[n] and n not in dropped]
    ok = [n for n in numbers if got[n]]
    if step.after:
        step.after(ctx, ok, out)
    if takes > 1:
        print("\n" + ", ".join(f"stage {n}: {got[n]}/{takes} takes" for n in numbers))
    print(f"\n{len(ok)}/{len(numbers)} stages succeeded in {out}" + (f"; failed: {sorted(failed)} (rerun to retry)" if failed else "")
          + (f"; cancelled: {sorted(set(dropped))}" if dropped else ""))
    return failed


def run_step(step, target, ctx, numbers, dry_run=False, reroll=False, takes=1) -> list:
    """ctx: name, d (project dir), engine (module), engine_name, common (run-wide settings), stages ({id: stage}).
    A step with `takes` (preview) keeps every render as a take (pipeline/takes.py); `takes` > 1 renders that many per stage,
    round-robin (take 1 of every stage first), each with a fresh seed. Every take becomes the current take as it lands —
    the previous current stays in the takes folder for comparison — and ✓s the preview (displacing a different take
    un-✓s what was made from the old one)."""
    takes = max(1, int(takes or 1)) if step.takes else 1
    if not step.supports(target.name, ctx.engine_name):
        sys.exit(f"{step.name} doesn't run on the {target.name} target with the {ctx.engine_name} engine "
                 f"(targets: {', '.join(step.targets)}; engines: {', '.join(step.engines or ['any'])})")
    needs = _needs_of(step, ctx, numbers)
    folder = step.folder(ctx) if callable(step.folder) else step.folder
    out = ctx.d / folder if folder else ctx.d
    final_node, labels = step.final(ctx)
    try:
        if takes > 1 and (any("seed" in ctx.stages[n] for n in numbers) or "seed" in getattr(ctx, "cfg", {})):
            print(f"  note: the pinned seed is overridden for {takes} takes (each take gets a fresh seed)", flush=True)
        images = _upload_needs(step, target, ctx, numbers, needs, dry_run)
        jobs = _build_jobs(step, target, ctx, numbers, images, out, final_node, labels, takes)
        problems = {j.n: p for j in jobs if (p := target.check(j.graph))}
        if dry_run:
            _report_dry_run(step, target, jobs, problems)
            return []
        if problems:  # validate every stage before queuing any, so a missing file can't leave half a batch running
            sys.exit(f"the {target.name} target can't run these stages (nothing was queued):\n"
                     + "\n".join(f"  stage_{n}: {x}" for n, p in problems.items() for x in p))
        _submit_all(step, target, ctx, numbers, reroll, out, jobs)
        got, finished = dict.fromkeys(numbers, 0), set()
        watch = _EditWatch(step, target, ctx, images, jobs, takes)
        for j in target.collect(jobs):
            _collect_one(ctx, step, needs, out, j, got, target.name)
            finished.add(j.tag)
            watch.check(finished)
        dropped = {j.n for j in jobs if j.error == CANCELLED} - {n for n in numbers if got[n]}
    except TargetUnavailable as e:
        sys.exit(str(e))
    return _wrap_up(step, ctx, out, numbers, got, takes, dropped)


def _note_source(ctx, step, needs, n, out):
    """A step made from preview/ (IL, hires) records which preview take it started from, so the web UI can flag a result made from a
    take that is no longer current."""
    if needs.get(n) != takes_mod.FOLDER:
        return
    takes_mod.sync(ctx.d, n)
    tid = takes_mod.current_id(ctx.d, n)
    if tid:
        dd = out / "_debug" / f"stage_{n}"
        dd.mkdir(parents=True, exist_ok=True)
        (dd / "source.json").write_text(json.dumps({"take": tid}))
