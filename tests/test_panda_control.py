import os

import numpy as np
import pytest

from lac import panda
from lac.panda_control import AdaptiveMPC, PandaMPC, PayloadEstimator, make_controller

THREADS = os.cpu_count() or 1
Q, TARGET = panda.make_tasks(1, seed=11)[0]


@pytest.fixture(scope="module")
def reaches():
    """2.5 kg in W1, 2 s, for each method."""
    out = {}
    for method in ("C0", "C1", "reference"):
        c = make_controller(method, "W1", 2.5, threads=THREADS)
        out[method] = (c, panda.run_reach(c, 2.5, Q, TARGET, 2.0, "W1"))
    return out


def test_reference_reaches_and_c0_stays_off_by_the_sag(reaches):
    assert reaches["reference"][1].final_error() < 0.003
    assert reaches["C0"][1].final_error() > 0.015


def test_c1_estimates_the_payload_every_ten_periods_and_reaches(reaches):
    c, r = reaches["C1"]
    assert [p for p, _ in c.mass_estimates] == list(range(10, 100, 10))
    assert abs(c.mass_estimates[-1][1] - 2.5) < 1e-6  # W1: point mass at the tip, as C1 assumes
    assert r.final_error() < 0.003


def test_the_robot_is_configured_with_the_controllers_belief(reaches):
    assert reaches["C0"][0].load == panda.Belief(0.0)
    assert reaches["reference"][0].load == panda.world_belief("W1", 2.5)


def test_estimator_is_exact_in_w1_and_ignores_steps_without_payload_information():
    est = PayloadEstimator()
    assert est.mass() == 0.0
    robot = panda.Robot(1.3, "W1", panda.Belief(0.0))
    robot.place(Q)
    rng = np.random.default_rng(0)
    for period in range(20):
        setpoint = Q + rng.normal(0.0, 0.1, 7)
        for _ in range(panda.HOLD):
            q, dq = robot.q, robot.dq
            tau = robot.step(setpoint)
            est.add(q, dq, tau, robot.dq)
    assert est.mass() == pytest.approx(1.3, abs=1e-6)


def test_setpoints_stay_within_joint_limits_and_change_once_per_period():
    c = PandaMPC(threads=THREADS, noise=2.0)  # wild candidates, so clipping matters
    r = panda.run_reach(c, 0.0, Q, TARGET, 0.2, "W1")
    lower, upper = c.lower, c.upper
    assert np.all(r.setpoint >= lower) and np.all(r.setpoint <= upper)
    changes = np.nonzero(np.any(np.diff(r.setpoint, axis=0) != 0, axis=1))[0] + 1
    assert set(changes) <= set(range(0, 100, panda.HOLD)) and len(c.solve_seconds) == 10


def test_same_seed_same_run_regardless_of_threads():
    a = panda.run_reach(PandaMPC(seed=3, threads=1), 1.0, Q, TARGET, 0.2, "W2")
    b = panda.run_reach(PandaMPC(seed=3, threads=THREADS), 1.0, Q, TARGET, 0.2, "W2")
    np.testing.assert_array_equal(a.q, b.q)


def test_planning_before_start_is_an_error():
    with pytest.raises(RuntimeError):
        PandaMPC()._replan(Q, np.zeros(7))
    with pytest.raises(ValueError):
        make_controller("C9", "W1", 0.0)


def test_adaptive_mpc_restarts_its_estimate_on_a_new_reach():
    c = AdaptiveMPC(threads=THREADS)
    panda.run_reach(c, 2.0, Q, TARGET, 0.5, "W1")
    panda.run_reach(c, 2.0, Q, TARGET, 0.1, "W1")
    assert c.mass_estimates == [] and c.load == panda.Belief(0.0)
