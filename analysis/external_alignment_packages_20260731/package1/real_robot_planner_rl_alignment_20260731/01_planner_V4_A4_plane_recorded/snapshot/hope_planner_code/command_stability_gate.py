"""Stateful confidence and revision gate for live racket commands."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CommandStabilityConfig:
    initial_consecutive: int = 3
    initial_position_tolerance_m: float = 0.10
    max_position_jump_m: float = 0.15
    max_velocity_jump_mps: float = 1.50
    freeze_time_to_strike_s: float = 0.25
    max_strike_time_jump_s: float = 0.040


class CommandStabilityGate:
    """Accept converged commands and suppress unsafe late/large revisions.

    ``reset`` must be called at each physical incoming-ball boundary. The first
    command is withheld until ``initial_consecutive`` candidate positions have
    converged. Once published, revisions are accepted only while enough time
    remains and their position/velocity stay close to the last accepted target.
    """

    def __init__(self, config: CommandStabilityConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._previous_candidate = None
        self._consecutive = 0
        self._accepted_position = None
        self._accepted_velocity = None
        self._accepted_strike_time = None
        self._last_reason = "reset"

    @property
    def has_accepted(self) -> bool:
        return self._accepted_position is not None

    @property
    def last_reason(self) -> str:
        return self._last_reason

    def consider(
        self, position, velocity, time_to_strike: float,
        candidate_time_s: Optional[float] = None,
    ) -> bool:
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        if (
            position.shape != (3,) or velocity.shape != (3,)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(velocity))
            or not np.isfinite(time_to_strike)
            or (candidate_time_s is not None and not np.isfinite(candidate_time_s))
        ):
            self._last_reason = "invalid_candidate"
            return False

        if not self.has_accepted:
            if self._previous_candidate is None:
                self._consecutive = 1
            elif np.linalg.norm(position - self._previous_candidate) <= self.config.initial_position_tolerance_m:
                self._consecutive += 1
            else:
                self._consecutive = 1
            self._previous_candidate = position.copy()
            if self._consecutive < max(1, self.config.initial_consecutive):
                self._last_reason = "awaiting_convergence"
                return False
            if time_to_strike <= self.config.freeze_time_to_strike_s:
                self._last_reason = "late_initial"
                return False
        else:
            if time_to_strike <= self.config.freeze_time_to_strike_s:
                self._last_reason = "late_freeze"
                return False
            if np.linalg.norm(position - self._accepted_position) > self.config.max_position_jump_m:
                self._last_reason = "position_jump"
                return False
            if np.linalg.norm(velocity - self._accepted_velocity) > self.config.max_velocity_jump_mps:
                self._last_reason = "velocity_jump"
                return False
            strike_time = (
                None if candidate_time_s is None
                else float(candidate_time_s) + float(time_to_strike)
            )
            if (
                strike_time is not None
                and self._accepted_strike_time is not None
                and abs(strike_time - self._accepted_strike_time)
                    > self.config.max_strike_time_jump_s
            ):
                self._last_reason = "strike_time_jump"
                return False

        self._last_reason = "accepted_revision" if self.has_accepted else "accepted_initial"
        self._accepted_position = position.copy()
        self._accepted_velocity = velocity.copy()
        self._accepted_strike_time = (
            None if candidate_time_s is None
            else float(candidate_time_s) + float(time_to_strike)
        )
        return True
