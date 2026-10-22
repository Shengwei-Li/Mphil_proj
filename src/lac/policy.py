"""Imitation policies over the demonstrations: CFDP and its nearest-neighbour baseline (M8).

Ported from Project B (``pfd.policies``), where they drove the PushT pusher;
here they drive the arm through joint-position chunks. The algorithm is the Closed-Form
Diffusion Policy of Mishra and Manchester (arXiv:2606.01238):

* For a finite dataset the diffusion "denoising direction" (the score) can be
  written down directly: a softmax-weighted average of the demonstrated action
  chunks. No network is trained; the demonstrations are the model.
* The observation enters the same softmax through a Gaussian kernel, so the
  weights favour demonstrations that started in a similar situation.
* To avoid replaying demonstrations exactly, the action is jittered by
  ``tau`` and averaged over ``n_mc`` draws; only the ``k_nn`` nearest
  demonstrations are used, with their local covariance as the distance metric.
* Sampling starts from noise and integrates the reverse probability-flow ODE
  over ``n_steps`` steps.

What the policies see and output (chosen after a first attempt that imitated
raw torques from absolute targets failed even without payload; see PROGRESS):

    observation  sin q, cos q, joint speeds, target minus current tip     8 numbers per period
    action       the next n_act joint positions, relative to the current ones

After standardising, the two target-minus-tip numbers are multiplied by
``target_weight`` before the nearest demonstrations are looked up. Without it the
six numbers describing the arm's own state decide which demonstrations count as
similar, and the policy copied motions that started from a similar pose but
headed somewhere else, ending 1 to 5 cm short of the target (dev tasks, M8).

``PolicyController`` turns either policy into a controller with the same
interface as the MPC. Every ``n_exec`` periods it generates a chunk; within a
period, a low-level PD controller with gravity compensation tracks the chunk's
joint position at 500 Hz. As on real arms, that gravity compensation is computed
for the load the controller believes it carries (none), so an unknown payload
shows up as the arm not reaching the commanded positions. Consecutive chunks
overlap by ``n_act - n_exec`` periods, which Project B's consistency monitor
compares (M9).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from lac import data, model
from lac.arm import forward_kinematics
from lac.demos import Demos


def frame(q: np.ndarray, dq: np.ndarray, target: np.ndarray) -> np.ndarray:
    """One period's observation: sin and cos of the angles, the speeds and the target relative to the tip."""
    return np.concatenate([np.sin(q), np.cos(q), dq, target - forward_kinematics(q)])


def training_pairs(demos: Demos, n_obs: int, n_act: int) -> tuple[np.ndarray, np.ndarray]:
    """(observation window, action chunk) pairs, one per period of every demonstration episode.

    The window holds the last n_obs frames (the first repeated at an episode's
    start); the chunk holds the joint angles of the next n_act periods minus the
    current ones (the last repeated at an episode's end). Both flattened.
    """
    windows, chunks = [], []
    for e in np.unique(demos.episode):
        q, dq, target = demos.q[demos.episode == e], demos.dq[demos.episode == e], demos.target[demos.episode == e]
        frames = np.array([frame(q[t], dq[t], target[t]) for t in range(len(q))])
        n = len(q)
        for t in range(n):
            windows.append(frames[np.clip(np.arange(t - n_obs + 1, t + 1), 0, None)].ravel())
            chunks.append((q[np.clip(np.arange(t + 1, t + 1 + n_act), None, n - 1)] - q[t]).ravel())
    return np.array(windows), np.array(chunks)


@dataclass
class Normalizer:
    """Per-dimension standardisation (angles' sin/cos, speeds and metres have different scales).

    An optional per-dimension weight multiplies the standardised values.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-8) -> "Normalizer":
        scale = x.std(axis=0)
        scale = np.where(scale < eps, 1.0, scale)
        return cls(mean=x.mean(axis=0), scale=scale if weight is None else scale / weight)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.scale

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.scale + self.mean


@dataclass
class ChunkIndex:
    """The demonstrations as (normalised observation window, normalised action chunk) pairs."""

    n_obs: int = 1
    n_act: int = 8
    k_nn: int = 2000
    target_weight: float = 3.0

    obs: np.ndarray = field(init=False, repr=False)
    act: np.ndarray = field(init=False, repr=False)
    obs_norm: Normalizer = field(init=False, repr=False)
    act_norm: Normalizer = field(init=False, repr=False)

    def fit(self, demos: Demos) -> "ChunkIndex":
        obs, act = training_pairs(demos, self.n_obs, self.n_act)
        weight = np.tile(np.r_[np.ones(6), np.full(2, self.target_weight)], self.n_obs)
        self.obs_norm, self.act_norm = Normalizer.fit(obs, weight), Normalizer.fit(act)
        self.obs, self.act = self.obs_norm(obs), self.act_norm(act)
        return self

    def query(self, z: np.ndarray) -> np.ndarray:
        """Indices of the k_nn nearest demonstration windows (Euclidean, normalised)."""
        k = min(self.k_nn, len(self.obs))
        d2 = np.sum((self.obs - z) ** 2, axis=1)
        return np.argpartition(d2, k - 1)[:k] if k < len(d2) else np.arange(len(d2))


def local_mahalanobis_metric(neighbours: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Inverse covariance of a neighbourhood, used as a local distance metric (identity if degenerate)."""
    cov = np.atleast_2d(np.cov(neighbours, rowvar=False))
    scale = np.trace(cov) / len(cov)
    if not np.isfinite(scale) or scale <= 0.0:
        return np.eye(len(cov))
    return np.linalg.inv(cov + ridge * scale * np.eye(len(cov)))


@dataclass
class NearestNeighborPolicy:
    """Baseline: copy the action chunk of the closest demonstration (local Mahalanobis metric)."""

    n_obs: int = 1
    n_act: int = 8
    k_nn: int = 2000
    target_weight: float = 3.0
    index: ChunkIndex = field(init=False, repr=False)

    def fit(self, demos: Demos) -> "NearestNeighborPolicy":
        self.index = ChunkIndex(self.n_obs, self.n_act, self.k_nn, self.target_weight).fit(demos)
        return self

    def predict(self, window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z = self.index.obs_norm(window)
        idx = self.index.query(z)
        delta = self.index.obs[idx] - z
        d2 = np.einsum("ij,jk,ik->i", delta, local_mahalanobis_metric(self.index.obs[idx]), delta)
        return self.index.act_norm.inverse(self.index.act[idx[int(np.argmin(d2))]]).reshape(self.n_act, 2)


@dataclass
class CFDP:
    """Closed-Form Diffusion Policy. Defaults follow Project B's main setting except n_obs,
    target_weight and h_per_dim, chosen on dev tasks (M8, see PROGRESS)."""

    n_obs: int = 1
    n_act: int = 8
    k_nn: int = 2000
    n_steps: int = 20  # reverse-ODE steps; 100 (Project B) gave the same dev errors at 4x the time
    tau: float = 0.07  # action jitter; 0 replays demonstrations exactly
    h_per_dim: float = 0.045  # observation bandwidth per sqrt(observation dimension); see bandwidth
    n_mc: int = 8  # Monte-Carlo draws of the jitter
    target_weight: float = 3.0
    index: ChunkIndex = field(init=False, repr=False)

    def fit(self, demos: Demos) -> "CFDP":
        self.index = ChunkIndex(self.n_obs, self.n_act, self.k_nn, self.target_weight).fit(demos)
        return self

    @property
    def bandwidth(self) -> float:
        """Kernel bandwidth h = h_per_dim * sqrt(D_z), D_z the observation window's dimension.

        Project B found that reading the paper's 0.09 as a per-dimension normalised
        distance keeps CFDP from collapsing to the nearest neighbour.
        """
        return self.h_per_dim * np.sqrt(self.index.obs.shape[1])

    def predict(self, window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z = self.index.obs_norm(window)
        idx = self.index.query(z)
        actions = self.index.act[idx]
        delta = self.index.obs[idx] - z
        d2_obs = np.einsum("ij,jk,ik->i", delta, local_mahalanobis_metric(self.index.obs[idx]), delta)
        log_w_obs = -d2_obs / (2.0 * self.bandwidth**2)  # fixed over the denoising steps

        d_a = actions.shape[1]
        a = rng.normal(size=d_a)  # start from noise
        ds = 1.0 / self.n_steps
        for step in range(self.n_steps, 0, -1):
            s = step * ds
            gain, sigma = 1.0 - s, s
            scaled = gain * actions
            perturbed = a + self.tau * rng.normal(size=(self.n_mc, d_a))
            d2 = np.sum((perturbed[:, None, :] - scaled[None, :, :]) ** 2, axis=-1)
            logits = -d2 / (2.0 * sigma**2) + log_w_obs[None, :]
            logits -= logits.max(axis=1, keepdims=True)
            w = np.exp(logits)
            w /= w.sum(axis=1, keepdims=True)
            denoised = (w @ scaled).mean(axis=0)  # where the data says a should be, at this noise level
            score = (denoised - a) / sigma**2
            a = a - ds * (-(a + s * score) / max(1.0 - s, ds))
        return self.index.act_norm.inverse(a).reshape(self.n_act, 2)


class PolicyController:
    """Runs an imitation policy with the same interface as the MPC (start, torque).

    Every ``n_exec`` periods it generates a chunk of ``n_act`` joint positions from
    the last ``n_obs`` frames; a PD controller with gravity compensation (computed
    for ``assumed_payload``) tracks the current period's position at every simulator
    step. Chunks are stored as absolute joint positions, with the period at which
    each was generated, for the monitors.
    """

    def __init__(self, policy, n_exec: int = 4, seed: int = 0, kp=(300.0, 150.0), kd=(20.0, 6.0),
                 assumed_payload: float = 0.0, period: float = data.PERIOD):
        self.policy = policy
        self.n_exec = n_exec
        self.rng = np.random.default_rng(seed)
        self.kp, self.kd = np.asarray(kp), np.asarray(kd)
        self.assumed_payload = assumed_payload
        self.hold = round(period / model.TIMESTEP)
        self.target: np.ndarray | None = None
        self._history: list[np.ndarray] = []
        self._period, self._calls = 0, 0
        self.chunks: list[np.ndarray] = []  # each (n_act, 2), absolute joint positions
        self.chunk_periods: list[int] = []
        self.solve_seconds: list[float] = []
        self.setpoints: list[np.ndarray] = []  # the joint position commanded for each period
        self.next_tip_predictions: list[np.ndarray] = []  # a policy makes no predictions of its own

    def start(self, target_xz: np.ndarray, q: np.ndarray) -> None:
        self.target = np.asarray(target_xz, dtype=float)
        self._history: list[np.ndarray] = []
        self._period, self._calls = 0, 0
        self.chunks, self.chunk_periods, self.setpoints = [], [], []

    def retarget(self, target_xz: np.ndarray) -> None:
        self.target = np.asarray(target_xz, dtype=float)

    def torque(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        q, dq = np.asarray(q, float), np.asarray(dq, float)
        if self._calls % self.hold == 0:
            self._setpoint = self._next_setpoint(q, dq)
        self._calls += 1
        tau = self.kp * (self._setpoint - q) - self.kd * dq + model.gravity(q, self.assumed_payload)
        return np.clip(tau, -model.TORQUE_LIMITS, model.TORQUE_LIMITS)

    def _next_setpoint(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Start of a period: record the frame, regenerate the chunk when due, return this period's target."""
        if self.target is None:
            raise RuntimeError("start must be called before acting")
        self._history.append(frame(q, dq, self.target))
        position = self._period % self.n_exec
        if position == 0:
            n_obs = self.policy.n_obs
            window = np.concatenate([self._history[max(len(self._history) - n_obs + i, 0)] for i in range(n_obs)])
            began = time.perf_counter()
            relative = self.policy.predict(window, self.rng)
            self.solve_seconds.append(time.perf_counter() - began)
            self.chunks.append(q + relative)
            self.chunk_periods.append(self._period)
        self._period += 1
        self.setpoints.append(self.chunks[-1][position])
        return self.setpoints[-1]
