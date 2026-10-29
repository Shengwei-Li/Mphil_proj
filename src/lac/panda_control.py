"""Controllers for the Panda (milestone M11b): sampling MPC over joint setpoints, online payload estimate.

The MPC is the two-link arm's (lac.mpc) with one change: it plans joint
*setpoints*, which the robot's own joint impedance loop tracks (lac.panda),
instead of torques. Sampling seven torques per period would waste most
candidates; a setpoint already brings the robot near where it should go. Every
period it

    1. perturbs its current plan (setpoints for the coming periods), one random
       change per segment of the horizon, half of the candidates with small changes,
    2. predicts every candidate with its model (lac.panda.Predictor), which
       includes the robot's impedance loop and the gravity compensation of the
       configured load,
    3. scores the predicted tip positions and joint speeds, blends the
       candidates with weights exp(-cost / temperature), sends the first setpoint.

The load the robot is configured with is always the controller's belief: a
controller that believes in a payload also compensates for it.

Online payload estimate (Panda's C1). Rigid-body dynamics are linear in the mass
of a point payload at a known place: the torque needed for a motion is

    tau = ID_0(q, dq, ddq) + m * (ID_1(q, dq, ddq) - ID_0(q, dq, ddq))

where ID_m is MuJoCo's inverse dynamics of the robot carrying m kg at the tip.
From every simulator step (joint angles, speeds, the next speeds, and the torque
the robot reports having applied; a real Panda reports its joint torques) this
gives one equation per joint, and m is their least-squares solution. The sums
are kept running, so re-estimating costs nothing. The estimate assumes what C1
assumes on the two-link arm: a point mass exactly at the tip, no extra friction.
Unlike lac.estimate (M4, M10) the data are noise-free simulator steps; adding
sensor noise is left for M12.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from lac import panda


class PandaMPC:
    def __init__(
        self,
        belief: panda.Belief = panda.Belief(),
        horizon: float = 0.3,  # s looked ahead
        samples: int = 128,  # candidate setpoint sequences per round
        noise: float = 0.1,  # rad, spread of the candidates' changes
        segments: int = 3,  # the horizon is split into this many pieces, each changed as a whole
        fine_scale: float = 0.1,  # half of the candidates use noise shrunk by this factor
        temperature: float = 5.0,
        speed_weight: float = 8.0,  # cost of joint speed (rad/s)^2, relative to 1 cm^2 of tip distance
        final_weight: float = 10.0,  # extra weight on the last predicted period
        seed: int = 0,
        threads: int = 1,
    ):
        self.predictor = panda.Predictor(belief, threads)
        self.steps = round(horizon / panda.PERIOD)
        self.samples = samples
        self.noise = noise
        self.segments = min(segments, self.steps)
        self.segment_of_step = np.arange(self.steps) * self.segments // self.steps
        self.fine_scale = fine_scale
        self.temperature = temperature
        self.speed_weight = speed_weight
        self.final_weight = final_weight
        self.rng = np.random.default_rng(seed)
        model = self.predictor.model
        self.lower, self.upper = model.jnt_range[:, 0].copy(), model.jnt_range[:, 1].copy()
        self.target: np.ndarray | None = None
        self.solve_seconds: list[float] = []
        self._calls = 0

    @property
    def load(self) -> panda.Belief:
        return self.predictor.belief

    def set_belief(self, belief: panda.Belief) -> None:
        self.predictor.set_belief(belief)

    def start(self, target: np.ndarray, q: np.ndarray) -> None:
        """Begin reaching for ``target``; the first plan holds the current pose."""
        self.target = np.asarray(target, dtype=float)
        self.plan = np.tile(np.asarray(q, dtype=float), (self.steps, 1))
        self._calls = 0

    def retarget(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=float)

    def setpoint(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Called every simulator step; replans once per period and holds the setpoint otherwise."""
        if self._calls % panda.HOLD == 0:
            began = time.perf_counter()
            self._replan(np.asarray(q, dtype=float), np.asarray(dq, dtype=float))
            self.solve_seconds.append(time.perf_counter() - began)
            self._current = self.plan[0].copy()
            self.plan = np.vstack([self.plan[1:], self.plan[-1:]])
        self._calls += 1
        return self._current

    def _cost(self, tips: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        """Per candidate, in cm^2 summed over the horizon (lac.mpc's cost, in three dimensions)."""
        distance_cm = np.linalg.norm(tips - self.target, axis=-1) * 100.0
        step_cost = distance_cm**2 + self.speed_weight * np.sum(speeds**2, axis=-1)
        step_cost[:, -1] *= self.final_weight
        return step_cost.sum(axis=1)

    def _replan(self, q: np.ndarray, dq: np.ndarray) -> None:
        if self.target is None:
            raise RuntimeError("start must be called before planning")
        n = self.samples
        change = self.rng.normal(0.0, self.noise, size=(n, self.segments, panda.JOINTS))
        change[n // 2 :] *= self.fine_scale
        candidates = np.clip(self.plan + change[:, self.segment_of_step], self.lower, self.upper)
        candidates[0] = self.plan  # always keep the unperturbed plan
        tips, speeds = self.predictor.rollout(q, dq, candidates)
        cost = self._cost(tips, speeds)
        weights = np.exp(-(cost - cost.min()) / self.temperature)
        self.plan = np.einsum("n,nsk->sk", weights / weights.sum(), candidates)


class PayloadEstimator:
    """Least-squares payload mass from the robot's own steps (see the module docstring)."""

    def __init__(self):
        self._models = []
        for mass in (0.0, 1.0):
            m = panda.build(panda.Belief(mass), gravcomp=False)
            m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE  # the inverse of an Euler step, exactly
            self._models.append((m, mujoco.MjData(m)))
        self.reset()

    def reset(self) -> None:
        self._yy = self._yr = 0.0
        self._previous = None

    def _inverse(self, index: int, q, dq, ddq) -> np.ndarray:
        m, d = self._models[index]
        d.qpos[:], d.qvel[:], d.qacc[:] = q, dq, ddq
        mujoco.mj_inverse(m, d)
        return d.qfrc_inverse.copy()

    def add(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray, dq_next: np.ndarray) -> None:
        """One simulator step: state (q, dq), applied torque tau, speeds after the step."""
        ddq = (np.asarray(dq_next) - dq) / panda.TIMESTEP
        unloaded = self._inverse(0, q, dq, ddq)
        y = self._inverse(1, q, dq, ddq) - unloaded  # torque per kg of payload
        self._yy += float(y @ y)
        self._yr += float(y @ (np.asarray(tau) - unloaded))

    def mass(self) -> float:
        """Current estimate in kg, never negative; 0 before any informative step."""
        return max(self._yr / self._yy, 0.0) if self._yy > 0 else 0.0


class AdaptiveMPC(PandaMPC):
    """C1 on the Panda: the MPC starts believing 0 kg and re-estimates the payload every ``reestimate_every`` periods.

    Needs the torque the robot applied at every step (``observe``), which
    lac.panda.run_reach supplies.
    """

    def __init__(self, reestimate_every: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.reestimate_every = reestimate_every
        self.estimator = PayloadEstimator()

    def start(self, target: np.ndarray, q: np.ndarray) -> None:
        super().start(target, q)
        self.set_belief(panda.Belief(0.0))  # every reach starts knowing nothing, as M10's adaptive MPC
        self.estimator.reset()
        self.mass_estimates: list[tuple[int, float]] = []  # (period, kg)
        self._last = None

    def setpoint(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        if self._last is not None:
            self.estimator.add(*self._last, dq)
            self._last = None
        if self._calls % panda.HOLD == 0 and self._calls // panda.HOLD % self.reestimate_every == 0 and self._calls:
            mass = self.estimator.mass()
            self.mass_estimates.append((self._calls // panda.HOLD, mass))
            self.set_belief(panda.Belief(mass))
        self._q, self._dq = np.asarray(q, dtype=float).copy(), np.asarray(dq, dtype=float).copy()
        return super().setpoint(q, dq)

    def observe(self, tau: np.ndarray) -> None:
        """The torque the robot applied during the step just taken."""
        self._last = (self._q, self._dq, np.asarray(tau, dtype=float))


def make_controller(method: str, world: str, payload: float, seed: int = 0, threads: int = 1):
    """C0 (believes no payload), C1 (online estimate) or reference (told the truth about payload and world)."""
    if method == "C0":
        return PandaMPC(panda.Belief(0.0), seed=seed, threads=threads)
    if method == "C1":
        return AdaptiveMPC(seed=seed, threads=threads)
    if method == "reference":
        return PandaMPC(panda.world_belief(world, payload), seed=seed, threads=threads)
    raise ValueError(f"unknown method {method}")
