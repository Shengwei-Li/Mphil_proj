"""Controllers (milestone M1: joint-space PD).

Nothing in this module reads the simulated arm. The controller works from
its own model of the arm (lac.model) and an assumed payload. When the real
payload differs from the assumed one, the gravity compensation below is
wrong, which is the effect this project studies.
"""

from __future__ import annotations

import numpy as np

from lac.arm import LINK_LENGTHS
from lac.model import TORQUE_LIMITS
from lac.model import gravity as gravity_model


def inverse_kinematics(xz: np.ndarray) -> list[np.ndarray]:
    """Joint angles that put the tip at (x, z). Returns two solutions, or none if out of reach.

    The two solutions mirror the elbow to either side of the line from the
    shoulder to the target.
    """
    l1, l2 = LINK_LENGTHS
    x, z = xz
    cos_q2 = (x**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if abs(cos_q2) > 1.0:
        return []
    solutions = []
    for q2 in (np.arccos(cos_q2), -np.arccos(cos_q2)):
        q1 = np.arctan2(z, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        solutions.append(np.array([q1, q2]))
    return solutions


def nearest_solution(xz: np.ndarray, q_reference: np.ndarray) -> np.ndarray | None:
    """The inverse-kinematics solution closest to `q_reference` in joint space."""
    solutions = inverse_kinematics(xz)
    if not solutions:
        return None
    return min(solutions, key=lambda q: np.linalg.norm(q - q_reference))


class PDController:
    """Joint-space PD control towards a fixed joint-angle goal.

    torque = kp * (goal - q) - kd * dq  [+ gravity_model(q, assumed_payload)]

    `assumed_payload=None` switches gravity compensation off.
    """

    def __init__(self, kp: float, kd: float, assumed_payload: float | None = None):
        self.kp = kp
        self.kd = kd
        self.assumed_payload = assumed_payload
        self.goal: np.ndarray | None = None

    def set_goal(self, q_goal: np.ndarray) -> None:
        self.goal = np.asarray(q_goal, dtype=float)

    def start(self, target_xz: np.ndarray, q: np.ndarray) -> None:
        """Begin reaching for `target_xz`, keeping the elbow on the side nearest to pose `q`."""
        goal = nearest_solution(target_xz, q)
        if goal is None:
            raise ValueError(f"target {target_xz} is out of reach")
        self.set_goal(goal)

    def torque(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        if self.goal is None:
            raise RuntimeError("set_goal must be called before torque")
        tau = self.kp * (self.goal - q) - self.kd * dq
        if self.assumed_payload is not None:
            tau = tau + gravity_model(q, self.assumed_payload)
        return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
