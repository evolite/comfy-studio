"""Helpers shared by every engine caller: stage-id validation and 'engine + run-wide settings + stage overrides -> graph'."""


def check_stage_ids(stages: list) -> None:
    """A stage's optional `id` names its stage_N.png / `motion` / `duration` files. Ids stay put when a stage
    is removed or reordered, so they only have to be unique positive integers."""
    seen = set()
    for pos, s in enumerate(stages, 1):
        sid = s.get("id", pos)
        if not isinstance(sid, int) or isinstance(sid, bool) or sid < 1 or sid in seen:
            raise SystemExit(f"stage at position {pos} has id {sid!r}; ids must be unique positive integers")
        seen.add(sid)


def stage_graph(engine, stage: dict, common: dict) -> dict:
    """A stage's settings override the run-wide ones; engines ignore keys they don't use."""
    settings = {k: v for k, v in common.items() if v is not None}
    settings.update({k: v for k, v in stage.items() if k not in ("id", "name")})
    return engine.build_graph(**settings)
