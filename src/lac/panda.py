"""The Franka Panda platform (milestone M11): the real robot, the controller's model, and reaching.

The robot model is MuJoCo Menagerie's ``panda_nohand.xml`` (third_party/,
Apache-2.0, commit in MENAGERIE_COMMIT), loaded and edited with MjSpec:

    - no contacts, as on the two-link arm: the study is about payload, not collisions
    - Euler integration (see "Exactness" below)
    - the joint servos get the gains of Franka's own joint-impedance example
      (libfranka): stiffness 600, 600, 600, 600, 250, 150, 50 N·m/rad,
      damping 50, 50, 50, 20, 20, 20, 10 N·m·s/rad
    - a tool point ("tip") 0.1 m beyond the flange, and a payload body there

How the robot is driven. As a real Panda in joint impedance mode, the robot is
sent joint setpoints and itself applies

    tau = K (q_set - q) - D dq  +  g_load(q)

where g_load is the gravity of the arm plus the *configured* end-effector load
(on a real Panda this load is entered in its web interface). The configured
load is the controller's belief; the real payload may differ, and then the
arm sags. The spring part is limited to the Panda's torque limits; the gravity
compensation is added on top.

The controller's model (``Predictor``) is a second copy of the same MuJoCo
model in which the payload, friction and so on are what the controller
believes. MuJoCo computes its equation of motion; with seven joints it cannot
be written out by hand. The model compensates gravity itself (MuJoCo's
``gravcomp``), which is exactly g_load of the believed load.

Exactness. Told the truth, the model must predict the real robot exactly, as
lac.model does on the two-link arm (tests/test_panda.py). This is why both use
Euler integration: MuJoCo's default for the Panda (implicitfast) treats the
servo damping implicitly, and the real robot's gravity compensation is an
applied force that cannot be treated the same way.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout

MODEL_FILE = Path(__file__).resolve().parents[2] / "third_party" / "franka_emika_panda" / "panda_nohand.xml"
JOINTS = 7
STIFFNESS = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])  # N·m/rad
DAMPING = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0])  # N·m·s/rad
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])  # N·m
TOOL_LENGTH = 0.1  # m from the flange to the tip
TIMESTEP = 0.002  # s, Menagerie's
PERIOD = 0.02  # s between setpoint changes
HOLD = round(PERIOD / TIMESTEP)
HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785])  # rad, start pose of every task (before jitter)
PAYLOAD_DENSITY = 7800.0  # kg/m^3 (steel), only sets how big the payload sphere is drawn
_FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


@dataclass(frozen=True)
class Belief:
    """What is assumed about the parts of the robot that may be unknown.

    The real robot is built from the true values; the controller's model and
    the robot's gravity compensation from believed ones.
    """

    payload_mass: float = 0.0  # kg
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # m from the tip, in the flange frame
    extra_damping: tuple[float, ...] = (0.0,) * JOINTS  # N·m·s/rad on top of Menagerie's 1.0
    frictionloss: tuple[float, ...] = (0.0,) * JOINTS  # N·m, dry friction (MuJoCo's frictionloss)


# Ways in which the real robot differs from the controller's model besides the
# payload mass, mirroring lac.arm.WORLDS: W1 none, W2 the payload's centre of
# mass is off the tip, W3 also more friction in every joint.
WORLDS = {
    "W1": {},
    "W2": {"payload_offset": (0.03, 0.02, 0.04)},
    "W3": {"payload_offset": (0.03, 0.02, 0.04), "extra_damping": (2.0,) * 4 + (0.5,) * 3,
           "frictionloss": (1.0,) * 4 + (0.3,) * 3},
}


def world_belief(world: str, payload: float) -> Belief:
    """The truth about the real robot in this world, i.e. what a reference controller is told."""
    return Belief(payload_mass=payload, **WORLDS[world])


@lru_cache(maxsize=16)
def _compiled(offset, extra_damping, frictionloss, gravcomp: bool) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(MODEL_FILE))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    for geom in spec.geoms:
        geom.contype = geom.conaffinity = 0
    for joint, damping, friction in zip(spec.joints, extra_damping, frictionloss):
        joint.damping += damping
        joint.frictionloss = friction
    for actuator, k, d, limit in zip(spec.actuators, STIFFNESS, DAMPING, TORQUE_LIMITS):
        actuator.gainprm[0] = k
        actuator.biasprm[:3] = [0.0, -k, -d]
        actuator.forcerange = [-limit, limit]
    flange = spec.body("attachment")
    flange.add_geom(name="tool", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, TOOL_LENGTH],
                    size=[0.008] * 3, rgba=[0.3, 0.3, 0.3, 1], contype=0, conaffinity=0, mass=0.0)
    flange.add_site(name="tip", pos=[0.0, 0.0, TOOL_LENGTH], size=[0.006] * 3, rgba=[0.1, 0.1, 0.1, 1])
    # A point mass; its mass is set at runtime (set_payload), 0 means unloaded.
    payload = flange.add_body(name="payload", pos=np.r_[0.0, 0.0, TOOL_LENGTH] + offset)
    payload.add_geom(name="payload_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001] * 3,
                     rgba=[0.85, 0.35, 0.15, 1], contype=0, conaffinity=0, mass=0.0)
    if gravcomp:
        for body in spec.bodies[1:]:
            body.gravcomp = 1.0
    spec.add_sensor(name="tip_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="tip")
    # Scene for viewing: floor, light, target marker. None of it has mass or contacts.
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.85, 0.9, 0.95], rgb2=[0.45, 0.55, 0.65], width=256, height=256)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.85, 0.85, 0.85], rgb2=[0.75, 0.75, 0.75], width=256, height=256)
    floor = spec.add_material(name="floor", texrepeat=[8, 8])
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.5, 1.5, 0.01], material="floor",
                            contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0.5, -0.5, 2.0], dir=[-0.2, 0.2, -1.0], diffuse=[0.6, 0.6, 0.6])
    spec.worldbody.add_light(pos=[-0.5, 0.8, 1.5], dir=[0.3, -0.5, -1.0], diffuse=[0.3, 0.3, 0.3], castshadow=False)
    target = spec.worldbody.add_body(name="target", mocap=True, pos=[0.5, 0.0, 0.4])
    target.add_site(name="target_site", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02] * 3, rgba=[0.1, 0.7, 0.2, 0.6])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    return spec.compile()


def build(belief: Belief, gravcomp: bool) -> mujoco.MjModel:
    """A fresh model of the robot with this payload and friction; ``gravcomp`` makes it compensate its own gravity."""
    model = copy.deepcopy(_compiled(tuple(belief.payload_offset), tuple(belief.extra_damping),
                                    tuple(belief.frictionloss), gravcomp))
    set_payload(model, belief.payload_mass)
    return model


def set_payload(model: mujoco.MjModel, mass: float) -> None:
    if mass < 0:
        raise ValueError("payload mass must be non-negative")
    model.body_mass[model.body("payload").id] = mass  # point mass: its rotational inertia stays zero
    model.geom("payload_geom").size[0] = max((3.0 * mass / (4.0 * np.pi * PAYLOAD_DENSITY)) ** (1.0 / 3.0), 1e-3)
    mujoco.mj_setConst(model, mujoco.MjData(model))


def _same_structure(a: Belief, b: Belief) -> bool:
    return replace(a, payload_mass=0.0) == replace(b, payload_mass=0.0)


class Robot:
    """The simulated real Panda. Controllers must not read its payload or world from here."""

    def __init__(self, payload: float = 0.0, world: str = "W1", load: Belief = Belief()):
        self.model = build(world_belief(world, payload), gravcomp=False)
        self.data = mujoco.MjData(self.model)
        self.configure_load(load)

    def configure_load(self, load: Belief) -> None:
        """Set the load the gravity compensation assumes (the controller's belief, not the truth)."""
        if getattr(self, "load", None) == load:
            return
        if getattr(self, "load", None) is not None and _same_structure(self.load, load):
            set_payload(self._load_model, load.payload_mass)
        else:
            self._load_model = build(load, gravcomp=True)
            self._load_data = mujoco.MjData(self._load_model)
        self.load = load

    def place(self, q: np.ndarray) -> None:
        """Put the arm at rest at joint angles ``q``."""
        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def gravity_compensation(self) -> np.ndarray:
        """g_load at the current pose: what the robot adds to the spring torques."""
        self._load_data.qpos[:] = self.data.qpos
        mujoco.mj_forward(self._load_model, self._load_data)
        return self._load_data.qfrc_gravcomp.copy()

    def step(self, setpoint: np.ndarray) -> np.ndarray:
        """One simulator step towards ``setpoint``. Returns the joint torques applied during it."""
        self.data.qfrc_applied[:] = self.gravity_compensation()
        self.data.ctrl[:] = setpoint
        mujoco.mj_step(self.model, self.data)
        tau = self.data.actuator_force + self.data.qfrc_applied  # actuator_force is from this step's start
        mujoco.mj_kinematics(self.model, self.data)  # so that tip() describes the new state
        return tau.copy()

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos.copy()

    @property
    def dq(self) -> np.ndarray:
        return self.data.qvel.copy()

    def tip(self) -> np.ndarray:
        return self.data.site("tip").xpos.copy()

    def set_target(self, xyz: np.ndarray) -> None:
        self.data.mocap_pos[0] = xyz


class Predictor:
    """The controller's model: "if the robot is sent these setpoints, where will the tip be?"

    Rolls many setpoint sequences forward at once with MuJoCo's multithreaded
    rollout. The robot's spring torques are part of the model, so predicting
    setpoints predicts the closed loop, exactly as the robot runs it.
    """

    def __init__(self, belief: Belief = Belief(), threads: int = 1):
        self.belief = belief
        self.model = build(belief, gravcomp=True)
        self._datas = [mujoco.MjData(self.model) for _ in range(threads)]
        self._state = np.zeros(mujoco.mj_stateSize(self.model, _FULL))

    def set_belief(self, belief: Belief) -> None:
        if _same_structure(self.belief, belief):
            set_payload(self.model, belief.payload_mass)
            self.belief = belief
        else:
            self.__init__(belief, len(self._datas))

    def rollout(self, q: np.ndarray, dq: np.ndarray, setpoints: np.ndarray):
        """Predict from state (q, dq) under setpoint sequences of shape (n, periods, 7).

        Each setpoint is held for one period. Returns the tip (n, periods, 3) and
        the joint speeds (n, periods, 7) at the end of every period.
        """
        d = self._datas[0]
        d.time = 0.0
        d.qpos[:], d.qvel[:] = q, dq
        mujoco.mj_getState(self.model, d, self._state, _FULL)
        n = len(setpoints)
        # Sensors are read at the start of each step, so one extra step (holding the last
        # setpoint) is appended to read the tip at the end of the last period.
        control = np.concatenate([np.repeat(setpoints, HOLD, axis=1), setpoints[:, -1:]], axis=1)
        state, tip = mj_rollout.rollout(self.model, self._datas, np.tile(self._state, (n, 1)), control,
                                        persistent_pool=len(self._datas) > 1)
        return tip[:, HOLD::HOLD], state[:, HOLD - 1 : -1 : HOLD, 1 + JOINTS : 1 + 2 * JOINTS]


@dataclass
class Rollout:
    """One reach, sampled at every simulator step (the Panda version of lac.simulate.Rollout)."""

    time: np.ndarray  # (N,) s
    q: np.ndarray  # (N, 7) rad
    dq: np.ndarray  # (N, 7) rad/s
    setpoint: np.ndarray  # (N, 7) rad
    torque: np.ndarray  # (N, 7) N·m, spring plus gravity compensation
    tip: np.ndarray  # (N, 3) m
    error: np.ndarray  # (N,) m, tip-to-target distance

    def final_error(self, window: float = 0.5) -> float:
        """Mean tip error over the last ``window`` seconds."""
        return float(self.error[self.time >= self.time[-1] - window].mean())


class HoldController:
    """Sends the start pose as the setpoint throughout; the load it configures is ``load``."""

    def __init__(self, load: Belief = Belief()):
        self.load = load

    def start(self, target: np.ndarray, q: np.ndarray) -> None:
        self._q = np.asarray(q, dtype=float).copy()

    def setpoint(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        return self._q


def run_reach(controller, payload: float, q_start: np.ndarray, target: np.ndarray, duration: float = 3.0,
              world: str = "W1") -> Rollout:
    """Start at rest at ``q_start`` carrying ``payload`` kg and reach for ``target`` (x, y, z in m).

    Every simulator step the controller is asked for a setpoint (``setpoint(q, dq)``)
    and the robot is configured with the controller's ``load``. The controller
    is never told ``payload`` or ``world``.
    """
    robot = Robot(payload, world, controller.load)
    robot.place(q_start)
    robot.set_target(target)
    controller.start(np.asarray(target, dtype=float), robot.q)
    log = {k: [] for k in ("time", "q", "dq", "setpoint", "torque", "tip", "error")}
    for _ in range(round(duration / TIMESTEP)):
        q, dq, tip = robot.q, robot.dq, robot.tip()
        setpoint = np.asarray(controller.setpoint(q, dq), dtype=float)
        robot.configure_load(controller.load)
        log["time"].append(robot.data.time)
        log["q"].append(q)
        log["dq"].append(dq)
        log["setpoint"].append(setpoint)
        log["tip"].append(tip)
        log["error"].append(np.linalg.norm(tip - target))
        log["torque"].append(robot.step(setpoint))
    return Rollout(**{k: np.array(v) for k, v in log.items()})


# Target regions, in metres in the robot's base frame. "train" is in front of
# the robot; "outside" is off to either side, for M12's policies.
REGIONS = {"train": ((0.35, 0.65), (-0.25, 0.25), (0.2, 0.6)),
           "outside": ((0.0, 0.3), (0.4, 0.6), (0.2, 0.6))}


def make_tasks(n: int, seed: int, region: str = "train") -> list[tuple[np.ndarray, np.ndarray]]:
    """``n`` (start pose, target) pairs: HOME jittered by up to 0.2 rad per joint, a target in the region.

    Outside targets are on the left or right side at random.
    """
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1), (z0, z1) = REGIONS[region]
    tasks = []
    for _ in range(n):
        q = HOME + rng.uniform(-0.2, 0.2, JOINTS)
        y = rng.uniform(y0, y1) * (rng.choice([-1, 1]) if region == "outside" else 1)
        tasks.append((q, np.array([rng.uniform(x0, x1), y, rng.uniform(z0, z1)])))
    return tasks
