"""Run a controller on the simulated arm and record what happened."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lac import arm
from lac.control import PDController, nearest_solution
from lac.mpc import MPCController


@dataclass
class Rollout:
    """One reach, sampled at every simulation step."""

    time: np.ndarray  # (N,) seconds
    q: np.ndarray  # (N, 2) rad
    dq: np.ndarray  # (N, 2) rad/s
    torque: np.ndarray  # (N, 2) N·m, after the torque limit
    tip: np.ndarray  # (N, 2) metres, (x, z)
    error: np.ndarray  # (N,) metres, tip-to-target distance
    q_goal: np.ndarray  # (2,) rad

    def final_error(self, window: float = 0.5) -> float:
        """Mean tip error over the last `window` seconds."""
        return float(self.error[self.time >= self.time[-1] - window].mean())


def run_reach(
    controller: PDController | MPCController,
    payload: float,
    q_start: np.ndarray,
    target_xz: np.ndarray,
    duration: float = 3.0,
    world: str = "W1",
) -> Rollout:
    """Start at rest at `q_start` with the given real payload and reach for `target_xz`.

    `world` selects how else the real arm differs from the controller's model
    (see lac.arm.WORLDS). The controller is told the target and sees joint
    angles and velocities. It is never told `payload` or `world`.
    """
    model, data = arm.load_arm()
    arm.set_world(model, world)
    arm.set_payload(model, payload)
    arm.set_target(model, data, target_xz)
    arm.set_joint_angles(model, data, q_start)

    q_goal = nearest_solution(target_xz, q_start)
    if q_goal is None:
        raise ValueError(f"target {target_xz} is out of reach")
    controller.start(target_xz, q_start)

    steps = round(duration / model.opt.timestep)
    log = {name: [] for name in ("time", "q", "dq", "torque", "tip", "error")}
    for _ in range(steps):
        tau = controller.torque(data.qpos.copy(), data.qvel.copy())
        data.ctrl[:] = tau
        # After mj_step the simulator's cached tip position still belongs to the previous
        # state, so compute the tip from the current joint angles instead.
        tip = arm.forward_kinematics(data.qpos)
        log["time"].append(data.time)
        log["q"].append(data.qpos.copy())
        log["dq"].append(data.qvel.copy())
        log["torque"].append(tau)
        log["tip"].append(tip)
        log["error"].append(np.linalg.norm(tip - target_xz))
        arm.step(model, data)
    return Rollout(**{k: np.array(v) for k, v in log.items()}, q_goal=q_goal)
