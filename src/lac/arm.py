"""The simulated two-link arm: loading, payload changes and tip geometry.

This module wraps the MuJoCo model that plays the role of the *real* arm.
Controllers must not read the payload mass or the world settings from here;
they have their own model of the arm (lac.model).

Always advance the arm with `step`, not mujoco.mj_step directly: in worlds
with Coulomb friction, `step` is what applies that friction.
"""

from __future__ import annotations

from importlib import resources

import mujoco
import numpy as np

LINK_LENGTHS = (0.3, 0.3)  # metres, must match two_link_arm.xml
PAYLOAD_DENSITY = 2000.0  # kg/m^3, only sets how big the payload sphere is drawn
BASE_DAMPING = 0.1  # N·m·s/rad, the joint damping in two_link_arm.xml
FRICTION_SMOOTHING = 0.2  # rad/s; Coulomb friction ramps up over roughly this speed

# Ways in which the real arm can differ from the controller's model, besides the
# payload mass. Keys match the fields of lac.model.Belief.
#   payload_offset  payload centre of mass relative to the tip, in metres:
#                   (along link 2, perpendicular to it)
#   extra_damping   viscous friction on top of BASE_DAMPING, N·m·s/rad per joint
#   coulomb         Coulomb friction, N·m per joint
WORLDS = {
    "W1": {"payload_offset": (0.0, 0.0), "extra_damping": (0.0, 0.0), "coulomb": (0.0, 0.0)},
    "W2": {"payload_offset": (0.04, 0.06), "extra_damping": (0.0, 0.0), "coulomb": (0.0, 0.0)},
    "W3": {"payload_offset": (0.04, 0.06), "extra_damping": (0.3, 0.15), "coulomb": (0.6, 0.3)},
}


def load_arm() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the two-link arm with no payload and all joints at zero."""
    xml = resources.files("lac").joinpath("assets/two_link_arm.xml").read_text(encoding="utf-8")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def set_payload(model: mujoco.MjModel, mass: float) -> None:
    """Attach a payload of `mass` kg at the tip (0 removes it)."""
    if mass < 0:
        raise ValueError("payload mass must be non-negative")
    body = model.body("payload").id
    radius = (3.0 * mass / (4.0 * np.pi * PAYLOAD_DENSITY)) ** (1.0 / 3.0)
    model.body_mass[body] = mass  # point mass: the body's rotational inertia stays zero
    model.geom("payload_geom").size[0] = max(radius, 1e-3)
    # Refresh mass-dependent constants; a scratch MjData keeps the caller's state intact.
    mujoco.mj_setConst(model, mujoco.MjData(model))


def get_payload(model: mujoco.MjModel) -> float:
    return float(model.body("payload").mass[0])


def set_world(model: mujoco.MjModel, name: str) -> None:
    """Make the real arm differ from the controller's model as described in WORLDS[name]."""
    world = WORLDS[name]
    along, across = world["payload_offset"]
    model.body("payload").pos[:] = (LINK_LENGTHS[1] + along, 0.0, across)
    model.dof_damping[:] = BASE_DAMPING + np.asarray(world["extra_damping"])
    model.jnt_user[:, 0] = world["coulomb"]
    mujoco.mj_setConst(model, mujoco.MjData(model))


def friction_torque(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Coulomb friction at the current joint speeds (N·m), smoothed near zero speed."""
    return -model.jnt_user[:, 0] * np.tanh(data.qvel / FRICTION_SMOOTHING)


def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Advance the real arm by one simulator step, including Coulomb friction."""
    data.qfrc_applied[:] = friction_torque(model, data)
    mujoco.mj_step(model, data)


def set_joint_angles(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> None:
    """Place the arm at joint angles `q` (rad) at rest and update derived quantities."""
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def tip_position(data: mujoco.MjData) -> np.ndarray:
    """Tip position (x, z) in metres as computed by the simulator.

    Valid after mj_forward (or set_joint_angles). Right after mj_step it still
    describes the state before that step.
    """
    return data.site("tip").xpos[[0, 2]].copy()


def target_position(data: mujoco.MjData) -> np.ndarray:
    """Target position (x, z) in metres."""
    return data.mocap_pos[0, [0, 2]].copy()


def set_target(model: mujoco.MjModel, data: mujoco.MjData, xz: np.ndarray) -> None:
    data.mocap_pos[0] = (xz[0], 0.0, xz[1])
    mujoco.mj_forward(model, data)


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Tip position (x, z) from joint angles, by the two-link formula."""
    l1, l2 = LINK_LENGTHS
    return np.array(
        [
            l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]),
            l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1]),
        ]
    )


def gravity_torque(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Joint torques (N·m) that hold an arm at rest in its current pose against gravity.

    Only meaningful at zero velocity: MuJoCo's bias force also contains
    velocity-dependent terms once the arm moves.
    """
    return data.qfrc_bias.copy()
