import numpy as np
import pytest

from lac import arm, learned, model, monitor
from lac.control import PDController
from lac.predictors import PhysicsPredictor
from lac.simulate import run_reach


def pd_periods(world: str, payload: float) -> monitor.Periods:
    q_start = np.array([0.3, 0.9])
    rollout = run_reach(PDController(kp=(60.0, 30.0), kd=(6.0, 2.0)), payload, q_start,
                        arm.forward_kinematics(q_start) + np.array([0.05, -0.08]), 1.0, world)
    return monitor.periods(rollout, 10)


def test_a_model_told_the_truth_predicts_every_period_exactly():
    p = pd_periods("W3", 0.0)
    truth = PhysicsPredictor(model.Belief(0.0, **arm.WORLDS["W3"]))
    assert monitor.period_residuals(truth, p).max() < 1e-9
    assert monitor.period_residuals(PhysicsPredictor(0.0), p).max() > 1e-3  # friction it does not know about


def test_an_unknown_payload_shows_up_in_the_residual():
    clean = monitor.period_residuals(PhysicsPredictor(0.0), pd_periods("W1", 0.0))
    loaded = monitor.period_residuals(PhysicsPredictor(0.0), pd_periods("W1", 0.1))
    assert clean.max() < 1e-9 and np.median(loaded) > 0.01


def test_a_network_that_outputs_zero_leaves_the_physics_residual_unchanged():
    p = pd_periods("W3", 0.0)
    zero = learned.ResidualNet([np.zeros((8, 4))], [np.zeros(4)], np.zeros(8), np.ones(8), np.zeros(4), np.ones(4))
    np.testing.assert_array_equal(monitor.period_residuals(learned.ResidualPredictor(PhysicsPredictor(0.0), zero), p),
                                  monitor.period_residuals(PhysicsPredictor(0.0), p))


def test_residual_signal_averages_the_periods_between_checks():
    p = pd_periods("W1", 0.1)
    r = monitor.period_residuals(PhysicsPredictor(0.0), p)
    signal = monitor.residual_signal(PhysicsPredictor(0.0), p, [0, 4, 8])
    assert np.isnan(signal[0])
    np.testing.assert_allclose(signal[1:], [r[0:4].mean(), r[4:8].mean()])


def test_measurement_noise_has_the_requested_size_and_is_reproducible():
    p = pd_periods("W1", 0.0)
    a = monitor.measured(p, np.random.default_rng(1), angle_noise=0.0, speed_noise=0.5)
    b = monitor.measured(p, np.random.default_rng(1), angle_noise=0.0, speed_noise=0.5)
    np.testing.assert_array_equal(a.dq, b.dq)
    np.testing.assert_array_equal(a.q, p.q)
    assert 0.4 < np.std(a.dq - p.dq) < 0.6


def test_chunk_disagreement_compares_only_the_shared_periods():
    first = np.arange(16.0).reshape(8, 2)
    agreeing = np.vstack([first[4:], np.full((4, 2), 99.0)])  # same plan for the 4 shared periods
    disagreeing = agreeing.copy()
    disagreeing[:4] += np.array([0.3, 0.4])
    d = monitor.chunk_disagreement([first, agreeing], [0, 4])
    assert np.isnan(d[0]) and d[1] == 0.0
    assert monitor.chunk_disagreement([first, disagreeing], [0, 4])[1] == pytest.approx(0.5)
    assert np.isnan(monitor.chunk_disagreement([first, agreeing], [0, 8])[1])  # no overlap


def test_cusum_stays_at_zero_on_clean_signals_and_grows_after_a_change():
    rng = np.random.default_rng(0)
    clean = [np.exp(rng.normal(size=40)) for _ in range(20)]
    cusum = monitor.Cusum().fit(clean)
    assert np.nanmedian(cusum.scores(clean[0])) < 1.0
    shifted = np.r_[clean[1][:20], 10 * clean[1][20:]]
    s = cusum.scores(shifted)
    assert s[-1] > 20 and s[-1] > s[19] + 20


def test_conformal_threshold_and_first_alarm():
    maxima = np.arange(1.0, 20.0)  # 19 clean episodes
    assert monitor.conformal_threshold(maxima, 0.05) == 19.0
    assert monitor.conformal_threshold(maxima, 0.10) == 18.0
    assert monitor.conformal_threshold(maxima[:18], 0.05) == float("inf")
    assert monitor.first_alarm(np.array([np.nan, 1.0, 3.0, 5.0]), [0, 4, 8, 12], 2.0) == 8
    assert monitor.first_alarm(np.array([np.nan, 1.0]), [0, 4], 2.0) is None
