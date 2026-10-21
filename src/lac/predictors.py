"""Predictors: "given this state and this torque held for a while, where will the arm be?"

The MPC asks a predictor this question hundreds of times per round. Making
the predictor a separate, swappable part means every method in the study
uses the same MPC and differs only in the predictor it hands over:

    C0, C1, reference   PhysicsPredictor with different beliefs
    C2, C3 (M5)         a physics predictor plus a learned correction

A predictor needs two things: `step` (batched prediction over a duration)
and `hold_torque` (the torque it thinks would hold a pose still, used to
start the MPC's first plan).
"""

from __future__ import annotations

import numpy as np

from lac import model


class PhysicsPredictor:
    """Integrates the hand-written equation of motion under a given belief."""

    def __init__(self, belief: model.Belief | float = 0.0, substeps: int = 10):
        self.belief = model.as_belief(belief)
        # Integration steps per call. 10 steps per 0.02 s period is the simulator's own
        # 0.002 s step, so a correct belief predicts the real arm exactly; 2 would be
        # five times cheaper but leaves about 0.5 cm of error after 0.2 s (measured in M4).
        self.substeps = substeps

    def step(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray, duration: float):
        """State after `duration` seconds with `tau` held constant. Works on batches."""
        dt = duration / self.substeps
        for _ in range(self.substeps):
            q, dq = model.predict(q, dq, tau, self.belief, dt)
        return q, dq

    def hold_torque(self, q: np.ndarray) -> np.ndarray:
        return model.gravity(q, self.belief)
