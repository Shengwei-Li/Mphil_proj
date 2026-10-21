"""C1: estimate the payload mass from recorded motion (milestone M4).

The idea: try a payload mass in the controller's model, predict every
recorded transition, and measure how far the predicted joint speeds are
from the recorded ones. The best estimate is the mass that makes this
mismatch smallest in the least-squares sense.

The payload mass enters the equation of motion almost linearly, so the
mismatch changes almost linearly with the guessed mass. One least-squares
step along that line (Gauss-Newton) then lands very close to the answer, and
a few repeats remove what is left. Only the mass is estimated: C1 believes
the payload sits exactly at the tip and that there is no extra friction.
"""

from __future__ import annotations

import numpy as np

from lac.data import PERIOD, Transitions
from lac.predictors import PhysicsPredictor

MASS_RANGE = (0.0, 5.0)  # kg; estimates are kept inside this physically sensible range


def _mismatch(mass: float, data: Transitions) -> np.ndarray:
    """Predicted minus recorded joint speeds after one period, for every transition, flattened."""
    _, dq_pred = PhysicsPredictor(mass).step(data.q, data.dq, data.tau, PERIOD)
    return (dq_pred - data.dq_next).ravel()


def payload_mass(data: Transitions, iterations: int = 5, initial: float = 0.0) -> float:
    """Least-squares estimate of the payload mass (kg) from transitions."""
    mass, h = initial, 1e-3
    for _ in range(iterations):
        r = _mismatch(mass, data)
        slope = (_mismatch(mass + h, data) - r) / h  # how the mismatch changes per kg
        mass = float(np.clip(mass - slope @ r / (slope @ slope), *MASS_RANGE))
    return mass
