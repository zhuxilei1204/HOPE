from pathlib import Path

import numpy as np
import pytest


def test_euler_split_step_matches_mj_step() -> None:
    mujoco = pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[3]
    model_path = (
        root
        / "agibot/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg"
        / "model/a3_pingpong/a3_pingpong.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    direct = mujoco.MjData(model)
    split = mujoco.MjData(model)
    direct.qpos[:] = model.qpos0
    split.qpos[:] = model.qpos0
    rng = np.random.default_rng(7)

    for _ in range(1000):
        ctrl = rng.normal(0.0, 1.0, model.nu)
        direct.ctrl[:] = ctrl
        split.ctrl[:] = ctrl
        mujoco.mj_step(model, direct)
        mujoco.mj_step1(model, split)
        mujoco.mj_step2(model, split)

    np.testing.assert_array_equal(direct.qpos, split.qpos)
    np.testing.assert_array_equal(direct.qvel, split.qvel)
    assert direct.time == split.time
