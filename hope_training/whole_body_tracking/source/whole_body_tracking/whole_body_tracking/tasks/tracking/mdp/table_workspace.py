"""Pure geometry helpers for the table-width strike-target curriculum."""

from __future__ import annotations


def table_side_lateral_bounds(
    table_width: float,
    edge_margin: float,
    side_overlap: float,
    swing_side: float,
) -> tuple[float, float]:
    """Return table-frame lateral bounds for one swing side.

    Forehand (positive swing sign) covers the negative-y half and backhand covers
    the positive-y half. The overlap around table centre avoids a discontinuity
    in side selection. The union of both ranges covers the complete playable
    table width after the configured edge margin.
    """
    width = float(table_width)
    margin = float(edge_margin)
    overlap = float(side_overlap)
    if width <= 0.0:
        raise ValueError(f"table_width must be positive, got {width}")
    half = 0.5 * width
    if not 0.0 <= margin < half:
        raise ValueError(f"edge_margin must be in [0, {half}), got {margin}")
    playable_half = half - margin
    if not 0.0 <= overlap <= playable_half:
        raise ValueError(f"side_overlap must be in [0, {playable_half}], got {overlap}")
    if float(swing_side) >= 0.0:
        return -playable_half, overlap
    return -overlap, playable_half


def interpolate_bounds(
    core: tuple[float, float],
    full: tuple[float, float],
    level: float,
) -> tuple[float, float]:
    """Linearly expand a core interval toward a full interval."""
    core_lo, core_hi = (float(v) for v in core)
    full_lo, full_hi = (float(v) for v in full)
    if core_hi < core_lo or full_hi < full_lo:
        raise ValueError(f"invalid interval: core={core}, full={full}")
    if core_lo < full_lo or core_hi > full_hi:
        raise ValueError(f"core interval {core} must be contained in full interval {full}")
    alpha = min(max(float(level), 0.0), 1.0)
    return (
        core_lo + alpha * (full_lo - core_lo),
        core_hi + alpha * (full_hi - core_hi),
    )


def motion_seed_blend(level: float, start_blend: float, end_level: float) -> float:
    """Fade a motion-aligned strike target out as measured task ability grows."""
    start = min(max(float(start_blend), 0.0), 1.0)
    if start == 0.0:
        return 0.0
    end = float(end_level)
    if end <= 0.0:
        raise ValueError("motion seed end_level must be positive when seeding is enabled")
    progress = min(max(float(level) / end, 0.0), 1.0)
    return start * (1.0 - progress)


def windowed_curriculum_level(
    level: float, start_level: float, full_level: float
) -> float:
    """Map a shared ability level onto one independently scheduled difficulty."""
    start = float(start_level)
    full = float(full_level)
    if not 0.0 <= start < full <= 1.0:
        raise ValueError(
            "curriculum window must satisfy 0 <= start_level < full_level <= 1"
        )
    return min(max((float(level) - start) / (full - start), 0.0), 1.0)


def ramped_curriculum_threshold(
    level: float,
    start_level: float,
    full_level: float,
    full_threshold: float,
) -> float:
    """Ramp an outcome requirement without a discontinuity at its first gate."""
    return float(full_threshold) * windowed_curriculum_level(
        level, start_level, full_level
    )


def hysteretic_curriculum_transition(
    *,
    level: int,
    max_level: int,
    good: bool,
    bad: bool,
    resolved_events: int,
    min_resolved_events: int,
    advance_streak: int,
    regress_streak: int,
    required_advance_checks: int,
    required_regress_checks: int,
) -> tuple[int, int, int, bool]:
    """Update one event-gated curriculum level with advance/regress hysteresis.

    Returns the next level, advance streak, regress streak, and whether the
    level changed. A regression takes priority if both guards are true.
    """
    if max_level < 0:
        raise ValueError("max_level must be non-negative")
    if not 0 <= level <= max_level:
        raise ValueError(f"level {level} must be in [0, {max_level}]")
    if min_resolved_events < 1:
        raise ValueError("min_resolved_events must be positive")
    if required_advance_checks < 1 or required_regress_checks < 1:
        raise ValueError("required curriculum checks must be positive")

    enough = int(resolved_events) >= int(min_resolved_events)
    next_advance = int(advance_streak) + 1 if bool(good) and enough else 0
    next_regress = int(regress_streak) + 1 if bool(bad) and level > 0 else 0

    if next_regress >= int(required_regress_checks):
        return level - 1, 0, 0, True
    if (
        next_advance >= int(required_advance_checks)
        and level < max_level
    ):
        return level + 1, 0, 0, True
    return level, next_advance, next_regress, False


def event_gated_scalar_curriculum_transition(
    *,
    level: float,
    good: bool,
    bad: bool,
    resolved_events: int,
    min_resolved_events: int,
    advance_streak: int,
    regress_streak: int,
    required_advance_checks: int,
    required_regress_checks: int,
    advance_rate: float,
    regress_rate: float,
) -> tuple[float, int, int, bool, bool]:
    """Update a continuous curriculum after a complete, disjoint event batch.

    The final boolean reports that a decision check was consumed. Callers reset
    their event counter after each consumed check, so a streak cannot reuse the
    same evidence window. Regression takes priority when both guards are true.
    """
    current = min(max(float(level), 0.0), 1.0)
    if min_resolved_events < 1:
        raise ValueError("min_resolved_events must be positive")
    if required_advance_checks < 1 or required_regress_checks < 1:
        raise ValueError("required curriculum checks must be positive")
    if advance_rate <= 0.0 or regress_rate <= 0.0:
        raise ValueError("curriculum advance/regress rates must be positive")

    if int(resolved_events) < int(min_resolved_events):
        return current, int(advance_streak), int(regress_streak), False, False

    next_advance = int(advance_streak) + 1 if bool(good) else 0
    next_regress = int(regress_streak) + 1 if bool(bad) and current > 0.0 else 0

    if next_regress >= int(required_regress_checks):
        next_level = max(0.0, current - float(regress_rate))
        return next_level, 0, 0, next_level != current, True
    if next_advance >= int(required_advance_checks) and current < 1.0:
        next_level = min(1.0, current + float(advance_rate))
        return next_level, 0, 0, next_level != current, True
    return current, next_advance, next_regress, False, True


def validate_table_workspace(
    table_width: float,
    edge_margin: float,
    side_overlap: float,
    forehand_core: tuple[float, float],
    backhand_core: tuple[float, float],
) -> None:
    """Validate that side cores are reachable subsets of full-table side ranges."""
    forehand_full = table_side_lateral_bounds(table_width, edge_margin, side_overlap, 1.0)
    backhand_full = table_side_lateral_bounds(table_width, edge_margin, side_overlap, -1.0)
    interpolate_bounds(forehand_core, forehand_full, 0.0)
    interpolate_bounds(backhand_core, backhand_full, 0.0)
