"""Table-tennis environment that adds no-spin ball aerodynamics on top of the manager-based RL env.

Everything except aerodynamics is handled by the standard :class:`~isaaclab.envs.ManagerBasedRLEnv`
machinery configured in :mod:`.table_tennis_env_cfg`. PhysX does not model air drag, so this subclass
registers a **physics-step callback** that, every physics substep, reads the ball velocity, computes the
no-spin drag force (:func:`.ball.compute_drag_force`) and writes it to the ball as an external force.

The drag path uses a physics callback and applies the force at the full physics rate. Physical-outcome
tasks may additionally opt into a small ``step()`` extension that publishes command/FK/contact state
immediately before reward computation. Historical tasks retain the stock Isaac Lab step order.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_rotate_inverse

from . import geometry
from .ball import compute_drag_force
from .table_tennis_env_cfg import TableTennisEnvCfg


class TableTennisEnv(ManagerBasedRLEnv):
    """Manager-based table-tennis env with a per-substep no-spin ball drag force field."""

    cfg: TableTennisEnvCfg

    def __init__(self, cfg: TableTennisEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._aero_active = False
        self._physics_substep_observer_active = False
        self._setup_ball_aerodynamics()
        self._setup_physics_substep_observer()

    def _setup_ball_aerodynamics(self) -> None:
        self._ball = self.scene["ball"]
        self._aero_cfg = self.cfg.ball_aerodynamics
        self._ball_mass = float(geometry.BALL_MASS)
        # Reusable zeroed external-wrench buffers: (num_envs, num_bodies=1, 3). Torque stays zero (no spin).
        self._aero_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._aero_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        if not self._aero_cfg.enabled:
            return
        try:
            # isaaclab.sim.SimulationContext inherits add_physics_callback from the Isaac Sim core
            # SimulationContext; the callback fires once per physics step with the step size.
            self.sim.add_physics_callback("ball_aerodynamics", self._apply_ball_aerodynamics)
            self._aero_active = True
        except Exception as exc:  # pragma: no cover - defensive: never block the sim on aero setup
            import omni.log

            omni.log.warn(
                f"[TableTennisEnv] could not register the ball aerodynamics physics callback "
                f"({exc!r}); the ball will fly on PhysX gravity + contacts only."
            )

    def _apply_ball_aerodynamics(self, dt: float) -> None:
        """Physics-step callback: apply the drag force to the ball (world frame -> body frame)."""
        lin_vel_w = self._ball.data.root_lin_vel_w
        force_w = compute_drag_force(lin_vel_w, self._ball_mass, self._aero_cfg)

        # set_external_force_and_torque applies the wrench in the body frame; rotate the world-frame
        # force into the ball body frame so the net effect is the intended world-frame force. The
        # torque is always zero (no-spin model).
        quat_w = self._ball.data.root_quat_w
        self._aero_force[:, 0, :] = quat_rotate_inverse(quat_w, force_w)

        self._ball.set_external_force_and_torque(self._aero_force, self._aero_torque)
        self._ball.write_data_to_sim()

    def _setup_physics_substep_observer(self) -> None:
        """Register an optional reward-free physics-rate observer exposed by a command term."""
        try:
            term = self.command_manager.get_term("physical_shadow")
        except Exception:
            return
        callback = getattr(term, "capture_physics_substep", None)
        if callback is None:
            return
        try:
            self.sim.add_physics_callback(
                "physical_shadow_substep_observer", callback
            )
            term.physics_substep_capture_registered = True
            self._physics_substep_observer_active = True
        except Exception as exc:  # pragma: no cover - simulator integration guard
            import omni.log

            omni.log.warn(
                "[TableTennisEnv] could not register the physical-shadow "
                f"substep observer ({exc!r}); control-rate diagnostics remain available."
            )

    def _command_term_or_none(self, name: str):
        try:
            return self.command_manager.get_term(name)
        except Exception:
            return None

    def _prepare_pre_physics_commands(self) -> None:
        physical = self._command_term_or_none("physical_shadow")
        callback = getattr(physical, "prepare_pre_physics", None)
        if callback is not None:
            callback()

    def _prepare_reward_command_snapshot(self) -> None:
        # FK/health must be current before the physical contact latch records
        # contact-frame planner errors and before any reward reads either term.
        for name in ("racket_target", "physical_shadow"):
            term = self._command_term_or_none(name)
            callback = getattr(term, "prepare_reward_snapshot", None)
            if callback is not None:
                callback()

    def _refresh_post_reset_command_kinematics(self) -> None:
        target = self._command_term_or_none("racket_target")
        callback = getattr(target, "refresh_kinematic_snapshot", None)
        if callback is not None:
            callback()

    def step(self, action):
        """Advance one control step with optional same-frame physical settlement."""
        if not bool(getattr(self.cfg, "pre_reward_command_snapshot_enabled", False)):
            return super().step(action)

        self._prepare_pre_physics_commands()
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._prepare_reward_command_snapshot()

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self._refresh_post_reset_command_kinematics()
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute()
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )
