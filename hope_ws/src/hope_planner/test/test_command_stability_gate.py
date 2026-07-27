import numpy as np

from hope_planner.command_stability_gate import (
    CommandStabilityConfig,
    CommandStabilityGate,
)


def command(gate, x, tts=0.8, velocity=(1.0, 0.0, 0.0), sample_time=None):
    return gate.consider(
        np.asarray(x), np.asarray(velocity), tts, candidate_time_s=sample_time
    )


def test_waits_for_consecutive_converged_candidates():
    gate = CommandStabilityGate(CommandStabilityConfig(initial_consecutive=3))
    assert not command(gate, (0.2, 0.0, 0.5))
    assert not command(gate, (0.2, 0.03, 0.52))
    assert command(gate, (0.2, 0.04, 0.53))


def test_large_candidate_jump_restarts_initial_convergence():
    gate = CommandStabilityGate(CommandStabilityConfig(initial_consecutive=2))
    assert not command(gate, (0.2, 0.0, 0.5))
    assert not command(gate, (0.2, 0.5, 1.0))
    assert command(gate, (0.2, 0.53, 1.02))


def test_rejects_large_revision_without_moving_reference():
    gate = CommandStabilityGate(CommandStabilityConfig(initial_consecutive=1))
    assert command(gate, (0.2, 0.0, 0.5))
    assert not command(gate, (0.2, 0.5, 0.5))
    assert command(gate, (0.2, 0.05, 0.5))


def test_freezes_near_strike_and_resets_for_next_ball():
    gate = CommandStabilityGate(CommandStabilityConfig(initial_consecutive=1))
    assert command(gate, (0.2, 0.0, 0.5), tts=0.8)
    assert not command(gate, (0.2, 0.01, 0.5), tts=0.2)
    gate.reset()
    assert command(gate, (0.2, -0.2, 0.6), tts=0.7)


def test_rejects_nonfinite_and_velocity_teleport():
    gate = CommandStabilityGate(CommandStabilityConfig(initial_consecutive=1))
    assert not command(gate, (0.2, np.nan, 0.5))
    assert command(gate, (0.2, 0.0, 0.5))
    assert not command(gate, (0.2, 0.01, 0.5), velocity=(4.0, 0.0, 0.0))


def test_rejects_absolute_strike_time_jump_without_moving_reference():
    gate = CommandStabilityGate(CommandStabilityConfig(initial_consecutive=1))
    assert command(gate, (0.2, 0.0, 0.5), tts=0.8, sample_time=10.0)
    assert not command(gate, (0.2, 0.01, 0.5), tts=0.86, sample_time=10.02)
    assert gate.last_reason == "strike_time_jump"
    assert command(gate, (0.2, 0.02, 0.5), tts=0.77, sample_time=10.03)
