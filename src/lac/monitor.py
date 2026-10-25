"""Runtime monitors for an imitation policy on the arm (milestone M9).

The question: to notice that the arm is no longer the arm the demonstrations were
recorded on (here: it carries an unknown payload), should a monitor watch the
policy itself, or compare what the arm does against a model of the normal arm,
and if so, a physics model or a learned one?

All monitors see only what the robot itself has: joint angles and speeds, the
torques it sent, and the chunks its policy generated. The angles and speeds a
monitor reads carry measurement noise (``periods``): without it a physics model
of a noise-free simulator is exact to rounding error, and the standardisation
below would blow rounding error up into a signal. None reads the payload or
the world. They produce one score per check, a check being every time the policy
generates a new chunk (every n_exec periods), so all are compared at the same
instants.

Signals (one number per check):

    residual    a model of the unloaded arm predicts each period's end speeds from
                its start state and the torques sent during it, simulator step by
                simulator step; the signal is the mean prediction error (rad/s)
                over the periods since the last check. (A first version used the
                period's mean torque; under the jerky motions of a policy outside
                its demonstrations that approximation alone raised false alarms,
                see PROGRESS M9.) A residual network's correction, trained for
                torques held over a period, is evaluated at the mean torque. Which model is the experiment's variable: physics
                without friction, physics plus a residual network trained on
                unloaded data, or (as an upper bound) physics told the world's
                friction.
    disagreement  Project B's overlapping-chunk disagreement: consecutive chunks
                both command the positions of the shared periods; the mean
                distance (rad) between the two commands. Watches the policy only.

Every signal is accumulated the same way, Project B's M3: standardise log(signal)
with the median and MAD of clean dev episodes, then CUSUM with allowance kappa.
So differences between monitors come from the signal, not the accumulation.

Thresholds are calibrated as in Project B: the maximum score of each clean
calibration episode, then the split-conformal quantile for the target
false-alarm rate (lac.monitor.conformal_threshold).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lac import data, learned, model


@dataclass
class Periods:
    """A run cut into control periods: what the robot measured and sent."""

    q: np.ndarray  # (P + 1, 2) joint angles at each period boundary
    dq: np.ndarray  # (P + 1, 2) joint speeds at each period boundary
    tau: np.ndarray  # (P, hold, 2) torque sent at each simulator step of each period


def periods(rollout, hold: int) -> Periods:
    """Period boundaries of a rollout (the log holds the state before each simulator step)."""
    n = len(rollout.q) // hold
    edges = np.arange(n) * hold
    return Periods(q=rollout.q[edges], dq=rollout.dq[edges],
                   tau=rollout.torque[: (n - 1) * hold].reshape(n - 1, hold, 2))


def measured(p: Periods, rng: np.random.Generator, angle_noise: float = 1e-4, speed_noise: float = 0.01) -> Periods:
    """The same periods as a monitor reads them: Gaussian noise on angles (rad) and speeds (rad/s).

    The defaults were fixed before any monitor was evaluated, as plausible for
    encoder-based sensing; they are not tuned.
    """
    return Periods(q=p.q + rng.normal(scale=angle_noise, size=p.q.shape),
                   dq=p.dq + rng.normal(scale=speed_noise, size=p.dq.shape), tau=p.tau)


def predict_period_speeds(predictor, p: Periods) -> np.ndarray:
    """Each period's end speeds as the predictor expects them, from the measured start state.

    Works for a PhysicsPredictor and for a ResidualPredictor built on one.
    """
    base = predictor.base if isinstance(predictor, learned.ResidualPredictor) else predictor
    q, dq = p.q[:-1], p.dq[:-1]
    dt = data.PERIOD / p.tau.shape[1]
    for i in range(p.tau.shape[1]):
        q, dq = model.predict(q, dq, p.tau[:, i], base.belief, dt)
    if isinstance(predictor, learned.ResidualPredictor):
        dq = dq + predictor.net(learned.features(p.q[:-1], p.dq[:-1], p.tau.mean(axis=1)))[:, 2:]
    return dq


def period_residuals(predictor, p: Periods) -> np.ndarray:
    """Per period, how far the predicted end speeds were from the measured ones (rad/s)."""
    return np.linalg.norm(predict_period_speeds(predictor, p) - p.dq[1:], axis=1)


def residual_signal(predictor, p: Periods, chunk_periods: list[int]) -> np.ndarray:
    """One number per check: the mean period residual since the previous check (nan at the first)."""
    r = period_residuals(predictor, p)
    out = np.full(len(chunk_periods), np.nan)
    for k in range(1, len(chunk_periods)):
        start, end = chunk_periods[k - 1], min(chunk_periods[k], len(r))
        if end > start:
            out[k] = r[start:end].mean()
    return out


def chunk_disagreement(chunks: list[np.ndarray], chunk_periods: list[int]) -> np.ndarray:
    """Project B's M2 signal on joint positions (rad); nan where consecutive chunks do not overlap."""
    out = np.full(len(chunks), np.nan)
    for k in range(1, len(chunks)):
        shift = chunk_periods[k] - chunk_periods[k - 1]
        overlap = len(chunks[k]) - shift
        if overlap > 0:
            out[k] = float(np.linalg.norm(chunks[k - 1][shift:] - chunks[k][:overlap], axis=1).mean())
    return out


@dataclass
class Cusum:
    """Project B's M3 accumulation: CUSUM over the log-standardised signal."""

    kappa: float = 0.5
    eps: float = 1e-6  # keeps the log finite when the signal is exactly zero
    mu: float = float("nan")
    sigma: float = float("nan")

    def fit(self, clean_signals: list[np.ndarray]) -> "Cusum":
        x = np.log(np.concatenate(clean_signals) + self.eps)
        x = x[np.isfinite(x)]
        self.mu = float(np.median(x))
        mad = float(np.median(np.abs(x - self.mu)))
        self.sigma = 1.4826 * mad if mad > 0 else float(x.std() or 1.0)
        return self

    def scores(self, signal: np.ndarray) -> np.ndarray:
        if not np.isfinite(self.mu):
            raise RuntimeError("fit on clean dev signals first")
        z = (np.log(signal + self.eps) - self.mu) / self.sigma
        out = np.full(len(z), np.nan)
        g = 0.0
        for k, zk in enumerate(z):
            if np.isfinite(zk):
                g = max(0.0, g + zk - self.kappa)
                out[k] = g
        return out


def conformal_threshold(episode_max: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold from clean episode maxima (Project B); alarm when score > threshold.

    +inf if there are too few episodes to support alpha.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    s = np.sort(np.asarray(episode_max, dtype=float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[k - 1]) if 1 <= k <= len(s) else float("inf")


def episode_max(scores: np.ndarray) -> float:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    return float(s.max()) if len(s) else float("-inf")


def first_alarm(scores: np.ndarray, chunk_periods: list[int], threshold: float) -> int | None:
    """Period of the first check whose score exceeds the threshold; None if the monitor stayed silent."""
    hit = np.flatnonzero(np.isfinite(scores) & (np.asarray(scores) > threshold))
    return int(chunk_periods[hit[0]]) if len(hit) else None
