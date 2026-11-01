"""Monitors and handover on the Panda (milestones M12b and M12c).

The two-link arm's M9 and M10 in three dimensions, with the same recipe:

* What a monitor sees: joint angles and speeds at period boundaries, with the
  measurement noise of M9 (1e-4 rad, 0.01 rad/s), the setpoint sent in each
  period, and the policy's chunks. On the Panda the robot's own impedance loop
  turns setpoints into torques, so a model predicts from setpoints, as the MPC's
  model does (lac.panda.Predictor).
* Residual signal: a model of the robot as the policy believes it to be (no
  payload, gravity compensated for no load) predicts each period's end speeds
  from its measured start state and setpoint; the signal is the mean
  prediction error (rad/s, all seven joints) over the periods since the last
  check. The model is the experiment's variable:

      physics   the Panda's MuJoCo model without the world's friction
      learned   the same plus a residual network (M12b) that learned, from the
                unloaded demonstrations, how far the physics model's end speeds
                are off; M5's frozen network settings (lac.learned.fit_network)
      oracle    the MuJoCo model told the world's friction (an upper bound)

* Consistency signal, CUSUM accumulation, conformal thresholds: lac.monitor's,
  unchanged (they do not depend on the robot).
* ``PandaHandover`` (M10's HandoverController on the Panda): the policy acts;
  at every check its detectors look at what has happened so far; on the first
  alarm the sampling MPC takes over, its payload estimated by least squares
  from all the steps so far (lac.panda_control.PayloadEstimator, which keeps
  collecting from the first step) and re-estimated every 10 periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lac import data, learned, monitor, panda
from lac.handover import Detector
from lac.panda_control import PandaMPC, PayloadEstimator
from lac.panda_policy import PandaDemos, demo_world

NETS = data.CACHE / "panda_nets"
NET_SETTING = {"hidden": (64, 64), "learning_rate": 1e-3, "max_epochs": 10000, "patience": 1000}  # M5's frozen


@dataclass
class Periods:
    """A Panda run cut into control periods: what the robot measured and was sent."""

    q: np.ndarray  # (P + 1, 7) joint angles at each period boundary
    dq: np.ndarray  # (P + 1, 7)
    setpoint: np.ndarray  # (P, 7) the setpoint held during each period


def periods(rollout: panda.Rollout) -> Periods:
    n = len(rollout.q) // panda.HOLD
    edges = np.arange(n) * panda.HOLD
    return Periods(q=rollout.q[edges], dq=rollout.dq[edges], setpoint=rollout.setpoint[edges[:-1]])


def measured(p: Periods, rng: np.random.Generator, angle_noise: float = 1e-4, speed_noise: float = 0.01) -> Periods:
    return Periods(q=p.q + rng.normal(scale=angle_noise, size=p.q.shape),
                   dq=p.dq + rng.normal(scale=speed_noise, size=p.dq.shape), setpoint=p.setpoint)


def features(q: np.ndarray, dq: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
    """The monitor network's input: sin q, cos q, joint speeds, setpoint minus angles (28 numbers)."""
    return np.concatenate([np.sin(q), np.cos(q), dq, setpoint - q], axis=-1)


class MonitorModel:
    """End-of-period speeds of the robot as a model believes it to be; optionally plus a residual network."""

    def __init__(self, belief: panda.Belief = panda.Belief(), gains: str = "stiff", net: learned.ResidualNet = None):
        self.predictor = panda.Predictor(belief, gains=gains)
        self.net = net

    def predict_speeds(self, q: np.ndarray, dq: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
        if len(q) == 0:
            return np.zeros((0, panda.JOINTS))
        _, speeds = self.predictor.rollout(q, dq, setpoint[:, None, :])
        speeds = speeds[:, 0]
        return speeds + self.net(features(q, dq, setpoint)) if self.net is not None else speeds


def period_residuals(model: MonitorModel, p: Periods) -> np.ndarray:
    return np.linalg.norm(model.predict_speeds(p.q[:-1], p.dq[:-1], p.setpoint) - p.dq[1:], axis=1)


def residual_signal(model: MonitorModel, p: Periods, chunk_periods: list[int]) -> np.ndarray:
    """One number per check: mean period residual since the previous check (nan at the first), as lac.monitor."""
    r = period_residuals(model, p)
    out = np.full(len(chunk_periods), np.nan)
    for k in range(1, len(chunk_periods)):
        start, end = chunk_periods[k - 1], min(chunk_periods[k], len(r))
        if end > start:
            out[k] = r[start:end].mean()
    return out


def train_monitor_net(demos: PandaDemos, gains: str, validation_share: float = 0.2, seed: int = 0,
                      threads: int = 1) -> learned.ResidualNet:
    """Residual network of the physics monitor model, from the unloaded demonstrations' periods.

    The last ``validation_share`` of the episodes (both kinds) only decide when to stop.
    """
    physics = MonitorModel(gains=gains)
    episodes = np.unique(demos.episode)
    rng = np.random.default_rng(seed)
    validation = np.isin(demos.episode, rng.choice(episodes, int(len(episodes) * validation_share), replace=False))
    x = features(demos.q, demos.dq, demos.setpoint)
    y = demos.next_dq - physics.predict_speeds(demos.q, demos.dq, demos.setpoint)
    return learned.fit_network(x[~validation], y[~validation], x[validation], y[validation], seed=seed,
                               threads=threads, **NET_SETTING)


def load_or_train_net(world: str, gains: str, demos: PandaDemos) -> learned.ResidualNet:
    path = NETS / f"monitor_{demo_world(world)}_{gains}.npz"
    if path.exists():
        return learned.ResidualNet.load(path)
    net = train_monitor_net(demos, gains)
    NETS.mkdir(parents=True, exist_ok=True)
    net.save(path)
    return net


def monitor_models(world: str, gains: str, demos: PandaDemos) -> dict[str, MonitorModel]:
    return {"physics": MonitorModel(gains=gains),
            "learned": MonitorModel(gains=gains, net=load_or_train_net(world, gains, demos)),
            "oracle": MonitorModel(panda.world_belief(world, 0.0), gains=gains)}


def residual(model: MonitorModel):
    return lambda p, chunks, chunk_periods: residual_signal(model, p, chunk_periods)


def consistency():
    return lambda p, chunks, chunk_periods: monitor.chunk_disagreement(chunks, chunk_periods)


def fit_detector(name: str, signal, dev: list[dict], calibration: list[dict], alpha: float) -> Detector:
    """lac.handover.fit_detector for Panda runs (dicts with periods, chunks, chunk_periods, noise_seed)."""
    def measured_signal(r):
        return signal(measured(r["periods"], np.random.default_rng(r["noise_seed"])), r["chunks"], r["chunk_periods"])

    cusum = monitor.Cusum().fit([measured_signal(r) for r in dev])
    maxima = [monitor.episode_max(cusum.scores(measured_signal(r))) for r in calibration]
    return Detector(name, signal, cusum, monitor.conformal_threshold(maxima, alpha))


class PandaHandover:
    """Policy until a detector alarms, then the MPC with an estimated payload. The MPC's interface."""

    def __init__(self, policy_controller, detectors: tuple[Detector, ...] = (), mpc_seed: int = 0,
                 reestimate_every: int = 10, noise_seed: int = 0, angle_noise: float = 1e-4,
                 speed_noise: float = 0.01, gains: str = "stiff"):
        self.policy = policy_controller
        self.detectors = detectors
        self.mpc_seed = mpc_seed
        self.reestimate_every = reestimate_every
        self.noise_seed = noise_seed
        self.angle_noise, self.speed_noise = angle_noise, speed_noise
        self.gains = gains
        self.estimator = PayloadEstimator()
        self.mpc: PandaMPC | None = None

    @property
    def load(self) -> panda.Belief:
        return (self.mpc or self.policy).load

    def start(self, target: np.ndarray, q: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=float)
        self.rng = np.random.default_rng(self.noise_seed)
        self._q, self._dq, self._setpoints = [], [], []  # measured boundaries; setpoint of each period
        self._calls, self._last = 0, None
        self.estimator.reset()
        self.mpc = None
        self.handover_period: int | None = None
        self.alarmed_by: list[str] = []
        self.mass_estimates: list[tuple[int, float]] = []
        self.policy.start(target, q)

    def periods(self) -> Periods:
        n = len(self._q)
        return Periods(q=np.array(self._q), dq=np.array(self._dq), setpoint=np.array(self._setpoints[: n - 1]))

    def setpoint(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        q, dq = np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
        if self._last is not None:
            self.estimator.add(*self._last, dq)
            self._last = None
        boundary, period = self._calls % panda.HOLD == 0, self._calls // panda.HOLD
        if boundary:
            self._q.append(q + self.rng.normal(scale=self.angle_noise, size=panda.JOINTS))
            self._dq.append(dq + self.rng.normal(scale=self.speed_noise, size=panda.JOINTS))
        if self.mpc is None:
            setpoint = self.policy.setpoint(q, dq)
            if boundary and period > 0 and self.policy.chunk_periods[-1] == period:
                p = self.periods()  # the setpoints of all completed periods are recorded by now
                self.alarmed_by = [d.name for d in self.detectors
                                   if d.alarms(p, self.policy.chunks, self.policy.chunk_periods)]
                if self.alarmed_by:
                    self._hand_over(q, period)
                    setpoint = self.mpc.setpoint(q, dq)
        else:
            if boundary and period > self.handover_period and (period - self.handover_period) % self.reestimate_every == 0:
                self._estimate(period)
            setpoint = self.mpc.setpoint(q, dq)
        if boundary:
            self._setpoints.append(np.asarray(setpoint, dtype=float).copy())
        self._q_now, self._dq_now = q, dq
        self._calls += 1
        return setpoint

    def observe(self, tau: np.ndarray) -> None:
        self._last = (self._q_now, self._dq_now, np.asarray(tau, dtype=float))

    def _estimate(self, period: int) -> None:
        mass = self.estimator.mass()
        self.mass_estimates.append((period, mass))
        self.mpc.set_belief(panda.Belief(mass))

    def _hand_over(self, q: np.ndarray, period: int) -> None:
        self.handover_period = period
        self.mpc = PandaMPC(seed=self.mpc_seed, gains=self.gains)
        self._estimate(period)
        self.mpc.start(self.target, q)
