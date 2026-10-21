"""Sampling-based model predictive control (milestone M2b).

Every `period` seconds the controller:
  1. draws many candidate torque sequences around its current plan,
  2. predicts each one with the hand-written model (lac.model) and scores it,
  3. blends the candidates, giving the well-scoring ones more weight,
     executes only the first torque of the blended plan, and keeps the rest
     as the starting point for the next round.

This weighting scheme is known as MPPI (model predictive path integral
control). The controller never reads the simulator: everything it predicts
comes from its predictor (lac.predictors), which by default is the
hand-written model with an assumed payload.
"""

from __future__ import annotations

import time

import numpy as np

from lac import model
from lac.arm import LINK_LENGTHS
from lac.predictors import PhysicsPredictor


def tip_of(q):
    """Tip position (x, z) for arrays of joint angles (last axis = joints)."""
    l1, l2 = LINK_LENGTHS
    a, b = q[..., 0], q[..., 0] + q[..., 1]
    return np.stack([l1 * np.cos(a) + l2 * np.cos(b), l1 * np.sin(a) + l2 * np.sin(b)], axis=-1)


class MPCController:
    def __init__(
        self,
        assumed_payload: float = 0.0,  # used only when no predictor is given
        predictor=None,  # anything with step() and hold_torque(), see lac.predictors
        horizon: float = 0.2,  # seconds the controller looks ahead
        samples: int = 256,  # candidate torque sequences per round
        period: float = 0.02,  # seconds between rounds; the torque is held in between
        substeps: int = 10,  # model steps per period for the default predictor (10 = the simulator's own step)
        noise: tuple[float, float] = (3.0, 1.5),  # N·m, spread of the candidates per joint
        segments: int = 3,  # the horizon is split into this many pieces, each changed as a whole
        fine_scale: float = 0.05,  # half of the candidates use noise shrunk by this factor
        temperature: float = 5.0,  # how sharply good candidates are preferred (smaller = sharper)
        speed_weight: float = 8.0,  # cost of moving, relative to cost of being 1 cm from the target
        final_weight: float = 10.0,  # extra weight on the last predicted state
        seed: int = 0,
        show_plan: bool = True,  # also predict the whole chosen plan (for drawing); costs about half a round
        spread_weight: float = 0.0,  # M6: penalty on the predictor's disagreement (needs `last_spread`)
    ):
        self.predictor = predictor if predictor is not None else PhysicsPredictor(assumed_payload, substeps)
        self.steps = round(horizon / period)
        self.samples = samples
        self.period = period
        self.noise = np.asarray(noise)
        self.segments = min(segments, self.steps)
        self.fine_scale = fine_scale
        self.segment_of_step = np.arange(self.steps) * self.segments // self.steps
        self.temperature = temperature
        self.speed_weight = speed_weight
        self.final_weight = final_weight
        self.rng = np.random.default_rng(seed)
        self.show_plan = show_plan
        self.spread_weight = spread_weight
        self.hold = round(period / model.TIMESTEP)  # simulator steps per round
        self.target: np.ndarray | None = None
        self.plan = np.zeros((self.steps, 2))  # torques for the coming rounds
        self.planned_tip = np.zeros((self.steps + 1, 2))  # where the model expects the plan to lead
        self.solve_seconds: list[float] = []
        self.next_tip_predictions: list[np.ndarray] = []  # one per round, see _replan
        self._calls = 0

    def start(self, target_xz: np.ndarray, q: np.ndarray) -> None:
        """Begin reaching for `target_xz`. The initial plan just holds the current pose."""
        self.target = np.asarray(target_xz, dtype=float)
        self.plan[:] = self.predictor.hold_torque(np.asarray(q, dtype=float))
        self._calls = 0

    def retarget(self, target_xz: np.ndarray) -> None:
        self.target = np.asarray(target_xz, dtype=float)

    def torque(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Called every simulator step; replans once per period and holds the torque otherwise."""
        if self._calls % self.hold == 0:
            self._current = self.act(q, dq)
        self._calls += 1
        return self._current

    def act(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Plan one round now and return the torque to hold for the next period."""
        if self.target is None:
            raise RuntimeError("start must be called before planning")
        began = time.perf_counter()
        self._replan(np.asarray(q, dtype=float), np.asarray(dq, dtype=float))
        self.solve_seconds.append(time.perf_counter() - began)
        first = self.plan[0].copy()
        # Shift the plan one round forward; repeat the last torque to fill the end.
        self.plan = np.vstack([self.plan[1:], self.plan[-1:]])
        return first

    # --- one round ---------------------------------------------------------

    def _predict(self, q, dq, torques):
        """Roll torque sequences of shape (steps, n, 2) through the predictor.

        Returns tip positions, joint speeds and the predictor's disagreement after
        every round, each of shape (steps, n, ...). The disagreement is None
        unless a penalty on it is switched on.
        """
        tips, speeds, spreads = [], [], []
        for tau in torques:
            q, dq = self.predictor.step(q, dq, tau, self.period)
            tips.append(tip_of(q))
            speeds.append(dq)
            if self.spread_weight:
                spreads.append(self.predictor.last_spread)
        return np.array(tips), np.array(speeds), (np.array(spreads) if self.spread_weight else None)

    def _cost(self, tips, speeds, spreads=None):
        """Score of each candidate (lower is better), in cm^2 summed over the horizon.

        With a disagreement penalty, each round also costs spread_weight times the
        squared disagreement (cm^2): the MPC is steered away from states where the
        ensemble's members disagree, i.e. where the training data was thin.
        """
        distance_cm = np.linalg.norm(tips - self.target, axis=-1) * 100.0
        step_cost = distance_cm**2 + self.speed_weight * np.sum(speeds**2, axis=-1)
        step_cost[-1] *= self.final_weight
        cost = step_cost.sum(axis=0)
        if spreads is not None:
            cost = cost + self.spread_weight * np.sum((spreads * 100.0) ** 2, axis=0)
        return cost

    def _replan(self, q, dq) -> None:
        n = self.samples
        # One random change per segment of the horizon (not per step), so a few hundred
        # candidates can cover the possibilities. Half of the candidates get full-size
        # changes for big moves; the other half get small ones for fine adjustment.
        change = self.rng.normal(0.0, self.noise, size=(self.segments, n, 2))
        change[:, n // 2 :, :] *= self.fine_scale
        candidates = self.plan[:, None, :] + change[self.segment_of_step]
        candidates[:, 0, :] = self.plan  # always keep the unperturbed plan as one candidate
        candidates = np.clip(candidates, -model.TORQUE_LIMITS, model.TORQUE_LIMITS)

        tips, speeds, spreads = self._predict(np.tile(q, (n, 1)), np.tile(dq, (n, 1)), candidates)
        cost = self._cost(tips, speeds, spreads)
        weights = np.exp(-(cost - cost.min()) / self.temperature)
        weights /= weights.sum()
        self.plan = np.einsum("n,snk->sk", weights, candidates)

        # The first torque of the plan is applied unchanged for one period, so the tip
        # predicted one period ahead can later be compared with where the arm really went.
        if self.show_plan:
            planned, _, _ = self._predict(q[None], dq[None], self.plan[:, None, :])
            self.planned_tip = np.vstack([tip_of(q)[None], planned[:, 0]])
            self.next_tip_predictions.append(self.planned_tip[1])
        else:
            q_next, _ = self.predictor.step(q[None], dq[None], self.plan[:1], self.period)
            self.next_tip_predictions.append(tip_of(q_next)[0])
