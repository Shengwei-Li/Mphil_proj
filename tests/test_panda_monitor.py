import numpy as np
import pytest

from lac import learned, monitor, panda
from lac import panda_monitor as pm
from lac.panda_policy import PandaDemos, PandaPolicyController, fit_policy, tip_of
from lac.policy import NearestNeighborPolicy

Q, TARGET = panda.make_tasks(1, seed=11)[0]


class Jitter:
    """Random setpoints around the start pose, one per period: cheap motion for checks."""

    def __init__(self, seed=0, gains="stiff"):
        self.rng, self.gains, self.load = np.random.default_rng(seed), gains, panda.Belief(0.0)

    def start(self, target, q):
        self.q0, self.calls = np.asarray(q).copy(), 0

    def setpoint(self, q, dq):
        if self.calls % panda.HOLD == 0:
            self.current = self.q0 + self.rng.normal(0, 0.1, 7)
        self.calls += 1
        return self.current


def jitter_periods(world, payload, gains="stiff", seconds=0.6, seed=0):
    return pm.periods(panda.run_reach(Jitter(seed, gains), payload, Q, TARGET, seconds, world))


def synthetic_demos(world, gains="stiff", episodes=6):
    parts = []
    for e in range(episodes):
        r = panda.run_reach(Jitter(e, gains), 0.0, Q, TARGET, 0.6, world)
        edges = np.arange(0, len(r.q), panda.HOLD)
        parts.append(dict(q=r.q[edges[:-1]], dq=r.dq[edges[:-1]], setpoint=r.setpoint[edges[:-1]],
                          target=np.tile(TARGET, (len(edges) - 1, 1)), next_dq=r.dq[edges[1:]],
                          episode=np.full(len(edges) - 1, e)))
    return PandaDemos(**{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})


def test_a_model_told_the_truth_predicts_every_period_exactly():
    p = jitter_periods("W3", 0.0, "soft")
    assert pm.period_residuals(pm.MonitorModel(panda.world_belief("W3", 0.0), "soft"), p).max() < 1e-9
    assert pm.period_residuals(pm.MonitorModel(gains="soft"), p).max() > 1e-3  # friction it does not know


def test_an_unknown_payload_shows_up_in_the_physics_residual():
    clean = pm.period_residuals(pm.MonitorModel(), jitter_periods("W1", 0.0))
    loaded = pm.period_residuals(pm.MonitorModel(), jitter_periods("W1", 0.5))
    assert clean.max() < 1e-9 and np.median(loaded) > 0.01


def test_the_monitor_network_learns_the_friction_residual():
    demos = synthetic_demos("W3")
    net = pm.train_monitor_net(demos, "stiff")
    p = jitter_periods("W3", 0.0, seed=99)  # motion not in the training data
    physics = pm.period_residuals(pm.MonitorModel(), p)
    learned_ = pm.period_residuals(pm.MonitorModel(net=net), p)
    assert np.median(learned_) < 0.5 * np.median(physics)


def test_a_network_that_outputs_zero_leaves_the_physics_residual_unchanged():
    p = jitter_periods("W3", 0.0)
    zero = learned.ResidualNet([np.zeros((28, 7))], [np.zeros(7)], np.zeros(28), np.ones(28), np.zeros(7), np.ones(7))
    np.testing.assert_array_equal(pm.period_residuals(pm.MonitorModel(net=zero), p),
                                  pm.period_residuals(pm.MonitorModel(), p))


def test_policy_controller_sends_chunks_and_keeps_them():
    policy = fit_policy(NearestNeighborPolicy(), synthetic_demos("W1", episodes=3))
    c = PandaPolicyController(policy, gains="soft")
    r = panda.run_reach(c, 0.0, Q, TARGET, 0.3, "W1")
    assert c.chunk_periods == [0, 4, 8, 12] and c.chunks[0].shape == (8, 7)
    np.testing.assert_allclose(r.setpoint[0], c.chunks[0][0])
    np.testing.assert_allclose(tip_of(Q), r.tip[0], atol=1e-12)


@pytest.fixture(scope="module")
def nn_policy():
    return fit_policy(NearestNeighborPolicy(), synthetic_demos("W1", episodes=3))


def test_handover_online_scores_equal_offline_and_alarm_hands_over(nn_policy):
    physics = pm.residual(pm.MonitorModel())
    silent = pm.Detector("physics", physics, monitor.Cusum().fit([np.exp(np.random.default_rng(0).normal(size=50))]),
                         np.inf)
    c = pm.PandaHandover(PandaPolicyController(nn_policy), (silent,), angle_noise=0.0, speed_noise=0.0)
    panda.run_reach(c, 1.0, Q, TARGET, 0.6, "W1")
    offline_policy = PandaPolicyController(nn_policy)
    p = pm.periods(panda.run_reach(offline_policy, 1.0, Q, TARGET, 0.6, "W1"))
    online = silent.scores(c.periods(), c.policy.chunks, c.policy.chunk_periods)
    np.testing.assert_allclose(online, silent.scores(p, offline_policy.chunks, offline_policy.chunk_periods)[: len(online)])
    assert c.handover_period is None

    eager = pm.Detector("physics", physics, monitor.Cusum().fit([np.full(30, 0.01)]), threshold=5.0)
    c = pm.PandaHandover(PandaPolicyController(nn_policy), (eager,), angle_noise=0.0, speed_noise=0.0)
    r = panda.run_reach(c, 2.0, Q, TARGET, 2.0, "W1")
    assert c.handover_period is not None and c.handover_period <= 16 and c.alarmed_by == ["physics"]
    assert abs(c.mass_estimates[-1][1] - 2.0) < 1e-3  # the estimate used every step, policy phase included
    assert r.final_error() < 0.005 and c.load == panda.Belief(c.mass_estimates[-1][1])


def test_fit_detector_threshold_is_finite_with_enough_calibration_runs(nn_policy):
    runs = []
    for s in range(25):
        c = PandaPolicyController(nn_policy, seed=s)
        r = panda.run_reach(c, 0.0, Q, TARGET, 0.4, "W3")
        runs.append({"periods": pm.periods(r), "chunks": c.chunks, "chunk_periods": c.chunk_periods, "noise_seed": s})
    d = pm.fit_detector("physics", pm.residual(pm.MonitorModel()), runs[:5], runs[5:], 0.10)
    assert np.isfinite(d.threshold)
