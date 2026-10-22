import numpy as np
import pytest

from lac import data, demos, model, policy
from lac.arm import forward_kinematics
from lac.demos import Demos
from lac.simulate import run_reach


def synthetic_demos(episodes: int = 6, length: int = 30, seed: int = 0) -> Demos:
    """Straight joint-space moves at constant speed; enough structure for the policies' bookkeeping."""
    rng = np.random.default_rng(seed)
    q, dq, target, episode = [], [], [], []
    for e in range(episodes):
        start, end = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
        path = start + np.linspace(0, 1, length)[:, None] * (end - start)
        q.append(path)
        dq.append(np.tile((end - start) / (length * data.PERIOD), (length, 1)))
        target.append(np.tile(forward_kinematics(end), (length, 1)))
        episode.append(np.full(length, e))
    q = np.concatenate(q)
    return Demos(q=q, dq=np.concatenate(dq), target=np.concatenate(target), torque=np.zeros_like(q),
                 episode=np.concatenate(episode))


def test_training_pairs_hold_future_joint_positions_relative_to_now():
    d = synthetic_demos(episodes=2, length=10)
    windows, chunks = policy.training_pairs(d, n_obs=2, n_act=3)
    assert windows.shape == (20, 16) and chunks.shape == (20, 6)
    q = d.q[:10]
    np.testing.assert_allclose(chunks[4].reshape(3, 2), q[5:8] - q[4])
    np.testing.assert_allclose(chunks[9].reshape(3, 2), np.zeros((3, 2)))  # last period repeats the end
    np.testing.assert_allclose(windows[0][:8], windows[0][8:])  # first frame repeated at the start


def test_target_weight_scales_only_the_target_minus_tip_numbers():
    d = synthetic_demos()
    plain = policy.ChunkIndex(n_obs=1, target_weight=1.0).fit(d)
    weighted = policy.ChunkIndex(n_obs=1, target_weight=3.0).fit(d)
    np.testing.assert_allclose(weighted.obs[:, :6], plain.obs[:, :6])
    np.testing.assert_allclose(weighted.obs[:, 6:], 3.0 * plain.obs[:, 6:])


def test_nearest_neighbour_returns_the_chunk_of_a_demonstrated_situation():
    d = synthetic_demos()
    nn = policy.NearestNeighborPolicy(n_act=4).fit(d)
    windows, chunks = policy.training_pairs(d, nn.n_obs, nn.n_act)
    got = nn.predict(windows[37], np.random.default_rng(0))
    np.testing.assert_allclose(got, chunks[37].reshape(4, 2), atol=1e-9)


def test_cfdp_is_reproducible_and_near_the_demonstration_with_little_noise():
    d = synthetic_demos()
    cfdp = policy.CFDP(n_act=4, tau=0.0, h_per_dim=0.01, n_steps=50).fit(d)
    windows, chunks = policy.training_pairs(d, cfdp.n_obs, cfdp.n_act)
    a = cfdp.predict(windows[37], np.random.default_rng(3))
    np.testing.assert_array_equal(a, cfdp.predict(windows[37], np.random.default_rng(3)))
    assert np.abs(a - chunks[37].reshape(4, 2)).max() < 0.01  # rad


class Still:
    """A policy that always asks to stay where the arm is."""

    n_obs, n_act = 1, 8

    def predict(self, window, rng):
        return np.zeros((self.n_act, 2))


def test_controller_holds_an_unloaded_arm_still_and_records_each_period():
    controller = policy.PolicyController(Still(), n_exec=4)
    q_start = np.array([0.4, 0.8])
    rollout = run_reach(controller, 0.0, q_start, forward_kinematics(q_start) - 0.05, 1.0, "W1")
    assert np.abs(rollout.q - q_start).max() < np.radians(0.5)  # gravity compensation does the holding
    assert len(controller.setpoints) == 50
    assert controller.chunk_periods == list(range(0, 50, 4))


def test_controller_with_wrong_gravity_compensation_sags_under_payload():
    controller = policy.PolicyController(Still())
    q_start = np.array([0.0, 0.3])
    rollout = run_reach(controller, 1.0, q_start, forward_kinematics(q_start), 1.0, "W1")
    gap = np.linalg.norm(np.array(controller.setpoints[-1]) - rollout.q[-1])
    assert gap > np.radians(0.5)
    fixed = policy.PolicyController(Still(), assumed_payload=1.0)
    rollout = run_reach(fixed, 1.0, q_start, forward_kinematics(q_start), 1.0, "W1")
    assert np.linalg.norm(np.array(fixed.setpoints[-1]) - rollout.q[-1]) < gap / 10


def test_recorded_demonstrations_have_both_kinds_of_episode():
    d = demos.record("W1", episodes=1, settle_episodes=1, seed=3)
    lengths = np.bincount(d.episode)
    assert list(lengths) == [150, 75]
    start = forward_kinematics(d.q[150])
    assert demos.SETTLE_OFFSET[0] - 1e-9 <= np.linalg.norm(d.target[150] - start) <= demos.SETTLE_OFFSET[1] + 1e-9
    assert np.all(model.TORQUE_LIMITS >= np.abs(d.torque).max(axis=0) - 1e-9)


def test_controller_refuses_to_act_before_start():
    with pytest.raises(RuntimeError):
        policy.PolicyController(Still()).torque(np.zeros(2), np.zeros(2))
