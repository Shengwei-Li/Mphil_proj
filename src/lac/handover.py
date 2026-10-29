"""Hand the arm from the policy to the MPC when a monitor alarms (milestone M10).

"The learned policy acts, a model watches, the physics model catches."

``HandoverController`` runs an imitation policy (lac.policy.PolicyController)
and, at every check (each time the policy generates a chunk), asks its
detectors whether to alarm, using only what has happened so far. On the first
alarm it hands over to the sampling MPC for the rest of the run. The MPC's
physics model gets the payload mass estimated from the run's own measured
periods (lac.estimate.payload_mass_from_periods), re-estimated every
``reestimate_every`` periods from all data so far, so the fallback knows no more
than what the robot has seen.

Without a policy the controller is that adaptive MPC from the start; it is the
baseline that tells whether the policy adds anything.

A detector computes exactly the score of lac.monitor offline: the same
measurement noise model, standardisation and threshold, checked at the same
instants, so thresholds calibrated offline (as in M9) hold online.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lac import estimate, model, monitor
from lac.mpc import MPCController
from lac.predictors import PhysicsPredictor

Signal = Callable[[monitor.Periods, list, list], np.ndarray]


def residual(predictor) -> Signal:
    return lambda p, chunks, chunk_periods: monitor.residual_signal(predictor, p, chunk_periods)


def consistency() -> Signal:
    return lambda p, chunks, chunk_periods: monitor.chunk_disagreement(chunks, chunk_periods)


@dataclass
class Detector:
    """One monitor, fitted and calibrated: alarm when the CUSUM score exceeds the threshold."""

    name: str
    signal: Signal
    cusum: monitor.Cusum
    threshold: float

    def scores(self, p: monitor.Periods, chunks, chunk_periods) -> np.ndarray:
        return self.cusum.scores(self.signal(p, chunks, chunk_periods))

    def alarms(self, p: monitor.Periods, chunks, chunk_periods) -> bool:
        """Whether the latest check's score exceeds the threshold."""
        s = self.scores(p, chunks, chunk_periods)
        return bool(len(s) and np.isfinite(s[-1]) and s[-1] > self.threshold)


def fit_detector(name: str, signal: Signal, dev: list[dict], calibration: list[dict], alpha: float) -> Detector:
    """Standardise on clean dev runs, threshold on clean calibration runs (Project B's recipe, as in M9).

    A run is a dict with the policy run's clean ``periods``, ``chunks``,
    ``chunk_periods`` and the ``noise_seed`` of its measurement noise.
    """
    def measured_signal(r):
        p = monitor.measured(r["periods"], np.random.default_rng(r["noise_seed"]))
        return signal(p, r["chunks"], r["chunk_periods"])

    cusum = monitor.Cusum().fit([measured_signal(r) for r in dev])
    maxima = [monitor.episode_max(cusum.scores(measured_signal(r))) for r in calibration]
    return Detector(name, signal, cusum, monitor.conformal_threshold(maxima, alpha))


class HandoverController:
    """Policy until a detector alarms, then the adaptive MPC. Same interface as the MPC (start, torque)."""

    def __init__(self, policy_controller=None, detectors: tuple[Detector, ...] = (), mpc_seed: int = 0,
                 reestimate_every: int = 10, noise_seed: int = 0, angle_noise: float = 1e-4,
                 speed_noise: float = 0.01):
        self.policy = policy_controller
        self.detectors = detectors
        self.mpc_seed = mpc_seed
        self.reestimate_every = reestimate_every
        self.noise_seed = noise_seed
        self.angle_noise, self.speed_noise = angle_noise, speed_noise
        self.hold = round(0.02 / model.TIMESTEP)
        self.mpc: MPCController | None = None

    def start(self, target_xz: np.ndarray, q: np.ndarray) -> None:
        self.target = np.asarray(target_xz, dtype=float)
        self.rng = np.random.default_rng(self.noise_seed)
        self._q, self._dq, self._tau = [], [], []  # measured boundaries; torque of every simulator step
        self._calls = 0
        self.mpc = None
        self.handover_period: int | None = None
        self.alarmed_by: list[str] = []
        self.mass_estimates: list[tuple[int, float]] = []  # (period, kg)
        if self.policy is None:
            self._hand_over(np.asarray(q, dtype=float), period=0)
        else:
            self.policy.start(target_xz, q)

    def retarget(self, target_xz: np.ndarray) -> None:
        self.target = np.asarray(target_xz, dtype=float)
        (self.mpc or self.policy).retarget(target_xz)

    def periods(self) -> monitor.Periods:
        """The measured periods completed so far."""
        n = len(self._q)
        tau = np.array(self._tau[: (n - 1) * self.hold]).reshape(n - 1, self.hold, 2)
        return monitor.Periods(q=np.array(self._q), dq=np.array(self._dq), tau=tau)

    def torque(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        q, dq = np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
        boundary = self._calls % self.hold == 0
        period = self._calls // self.hold
        if boundary:
            self._q.append(q + self.rng.normal(scale=self.angle_noise, size=2))
            self._dq.append(dq + self.rng.normal(scale=self.speed_noise, size=2))
        if self.mpc is None:
            tau = self.policy.torque(q, dq)  # generates the chunk when one is due
            if boundary and period > 0 and self.policy.chunk_periods and self.policy.chunk_periods[-1] == period:
                p = self.periods()
                self.alarmed_by = [d.name for d in self.detectors
                                   if d.alarms(p, self.policy.chunks, self.policy.chunk_periods)]
                if self.alarmed_by:
                    self._hand_over(q, period)
                    tau = self.mpc.torque(q, dq)
        else:
            if boundary and period > self.handover_period and (period - self.handover_period) % self.reestimate_every == 0:
                self._estimate(period)
            tau = self.mpc.torque(q, dq)
        self._tau.append(np.asarray(tau, dtype=float))
        self._calls += 1
        return tau

    def _estimate(self, period: int) -> None:
        mass = estimate.payload_mass_from_periods(self.periods()) if period > 0 else 0.0
        self.mass_estimates.append((period, mass))
        self.mpc.predictor = PhysicsPredictor(mass)

    def _hand_over(self, q: np.ndarray, period: int) -> None:
        self.handover_period = period
        self.mpc = MPCController(seed=self.mpc_seed, show_plan=False)
        self._estimate(period)
        self.mpc.start(self.target, q)
