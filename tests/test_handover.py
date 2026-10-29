import numpy as np
import pytest

from lac import arm, demos, estimate, handover, monitor
from lac.policy import NearestNeighborPolicy, PolicyController
from lac.predictors import PhysicsPredictor
from lac.simulate import run_reach

Q_START = np.array([0.2, 1.2])
TARGET = arm.forward_kinematics(np.array([0.6, 0.7]))


@pytest.fixture(scope="module")
def policy():
    return NearestNeighborPolicy().fit(demos.record("W1", episodes=6, settle_episodes=6, seed=1))


def clean_run(policy, payload):
    c = PolicyController(policy, seed=0)
    r = run_reach(c, payload, Q_START, TARGET, 1.0, "W1")
    return {"periods": monitor.periods(r, c.hold), "chunks": c.chunks, "chunk_periods": c.chunk_periods,
            "noise_seed": 0}


def test_online_scores_equal_offline_scores(policy):
    detector = handover.Detector("physics", handover.residual(PhysicsPredictor(0.0)),
                                 monitor.Cusum().fit([np.exp(np.random.default_rng(0).normal(size=50))]), np.inf)
    c = handover.HandoverController(PolicyController(policy, seed=0), (detector,), angle_noise=0.0, speed_noise=0.0)
    run_reach(c, 0.5, Q_START, TARGET, 1.0, "W1")
    offline = clean_run(policy, 0.5)
    online = detector.scores(c.periods(), c.policy.chunks, c.policy.chunk_periods)
    expected = detector.scores(offline["periods"], offline["chunks"], offline["chunk_periods"])
    np.testing.assert_array_equal(online, expected[: len(online)])
    assert c.handover_period is None  # an infinite threshold never alarms


def test_alarm_hands_over_to_the_mpc_with_an_estimated_payload(policy):
    detector = handover.Detector("physics", handover.residual(PhysicsPredictor(0.0)),
                                 monitor.Cusum().fit([np.full(30, 0.01)]), threshold=5.0)
    c = handover.HandoverController(PolicyController(policy, seed=0), (detector,), angle_noise=0.0, speed_noise=0.0)
    rollout = run_reach(c, 1.0, Q_START, TARGET, 2.0, "W1")
    assert c.handover_period is not None and c.handover_period <= 8
    assert c.alarmed_by == ["physics"]
    assert abs(c.mass_estimates[-1][1] - 1.0) < 0.02
    assert rollout.final_error() < 0.005


def test_without_a_policy_it_is_the_adaptive_mpc_from_the_start():
    c = handover.HandoverController(None, noise_seed=3)
    rollout = run_reach(c, 1.0, Q_START, TARGET, 2.0, "W1")
    assert c.handover_period == 0 and c.mass_estimates[0] == (0, 0.0)
    assert [p for p, _ in c.mass_estimates][:3] == [0, 10, 20]
    assert abs(c.mass_estimates[-1][1] - 1.0) < 0.1
    assert rollout.final_error() < 0.01


def test_mass_estimate_from_policy_periods(policy):
    p = clean_run(policy, 0.7)["periods"]
    assert estimate.payload_mass_from_periods(p) == pytest.approx(0.7, abs=1e-3)


def test_calibrated_threshold_is_finite_with_enough_clean_runs(policy):
    runs = [clean_run(policy, 0.0) | {"noise_seed": s} for s in range(20)]
    d = handover.fit_detector("physics", handover.residual(PhysicsPredictor(0.0)), runs[:5], runs[5:], 0.10)
    assert np.isfinite(d.threshold)
    assert handover.fit_detector("physics", handover.residual(PhysicsPredictor(0.0)), runs[:5], runs[5:10],
                                 0.05).threshold == np.inf
