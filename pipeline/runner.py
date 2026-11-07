"""The one loop that runs a step on a target: build each stage's graph, prepare it for the target, validate every stage before
queuing any, queue them all, collect as they finish, save into the step's folder. Returns the stages that failed."""
import json
import secrets
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import history  # noqa: E402
import image_loras  # noqa: E402
import takes as takes_mod  # noqa: E402
from targets.base import StageJob, TargetUnavailable  # noqa: E402


def _needs_of(step, ctx, numbers) -> dict:
    """The step's input folder per stage ({} when it starts from text); exits naming stages whose input file is missing."""
    needs = {n: step.needs(ctx, n) if callable(step.needs) else step.needs for n in numbers} if step.needs else {}
    missing = [n for n, f in needs.items() if not (ctx.d / f / f"stage_{n}.png").exists()]
    if missing:
        sys.exit(f"stages {missing}: no {'/'.join(sorted({needs[n] for n in missing}))}/stage_N.png yet -- "
                 + step.needs_hint.format(name=ctx.name, stages=" ".join(map(str, missing))))
    return needs


def _upload_needs(step, target, ctx, numbers, needs, dry_run) -> dict:
    """The step's input image per stage, uploaded to the target ("x" placeholders on a dry run)."""
    images = {}
    for n in numbers:
        if n in needs:
            images[n] = "x" if dry_run else target.upload(ctx.d / needs[n] / f"stage_{n}.png", f"{step.name}_{ctx.name}_{n}.png")
    return images


def _build_jobs(step, target, ctx, numbers, images, out, final_node, labels, takes) -> list:
    """One StageJob per stage (per take): fresh seed per take, graph prepared for the target."""
    jobs = []
    for k in range(1, takes + 1):
        for n in numbers:
            stage = ctx.stages[n] if takes == 1 else {**ctx.stages[n], "seed": secrets.randbits(48)}
            graph = target.prepare(step.graph(ctx, stage, images.get(n), target))
            tag = f"stage_{n}" + (f".t{k}" if takes > 1 else "")
            label = f"{ctx.name} stage {n}" + (f" {step.suffix}" if step.suffix else "") + (f" t{k}" if takes > 1 else "")
            j = StageJob(n, graph, label, out / "_debug" / tag, final_node, labels)
            j.tag, j.stage = tag, stage
            jobs.append(j)
    return jobs


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
    """Save one finished job — as a take, or over the stage file with its history entry; a failure just logs."""
    if not j.final:
        print(f"  [{j.tag}] FAILED: {j.error}", flush=True)
        return
    dest = out / f"stage_{j.n}.png"
    if step.takes:
        takes_mod.sync(ctx.d, j.n)
        rec = takes_mod.record(j.graph, j.stage, ctx.common, target_name, image_loras.resolve(j.stage.get("loras")))
        before = takes_mod.current_id(ctx.d, j.n)
        tid = takes_mod.add(ctx.d, j.n, j.final, j.graph, rec, make_current=True)
        takes_mod.approve(ctx.d, j.n, changed=before != tid)
    else:
        history.archive(ctx.d, dest)  # the result it replaces stays restorable (web UI: ⋯ > Previous versions)
        shutil.copyfile(j.final, dest)
        _note_source(ctx, step, needs, j.n, out)
    got[j.n] += 1
    print(f"  [{j.tag}] done (waited {j.waited:.0f}s" + (f", processed {j.seconds:.0f}s" if j.seconds else "") + f") -> {dest}", flush=True)


def _wrap_up(step, ctx, out, numbers, got, takes) -> list:
    """Run the step's after-hook and print the summary; returns the failed stage numbers."""
    failed = [n for n in numbers if not got[n]]
    ok = [n for n in numbers if n not in failed]
    if step.after:
        step.after(ctx, ok, out)
    if takes > 1:
        print("\n" + ", ".join(f"stage {n}: {got[n]}/{takes} takes" for n in numbers))
    print(f"\n{len(ok)}/{len(numbers)} stages succeeded in {out}" + (f"; failed: {sorted(failed)} (rerun to retry)" if failed else ""))
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
    out = ctx.d / step.folder if step.folder else ctx.d
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
        got = dict.fromkeys(numbers, 0)
        for j in target.collect(jobs):
            _collect_one(ctx, step, needs, out, j, got, target.name)
    except TargetUnavailable as e:
        sys.exit(str(e))
    return _wrap_up(step, ctx, out, numbers, got, takes)


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
