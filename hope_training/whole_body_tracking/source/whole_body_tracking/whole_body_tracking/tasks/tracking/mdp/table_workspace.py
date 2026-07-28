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
