import numpy as np
import pytest

from lac import arm, data, estimate, model
from lac.predictors import PhysicsPredictor


@pytest.fixture(scope="module")
def w1_data():
    return data.collect("W1", 0.8, seconds=6.0, seed=0)


def test_collect_records_whole_episodes_of_consecutive_transitions(w1_data):
    per_episode = round(data.EPISODE_SECONDS / data.PERIOD)
    assert len(w1_data) == 2 * per_episode
    assert w1_data.seconds == pytest.approx(6.0)
    same_episode = w1_data.episode[1:] == w1_data.episode[:-1]
    np.testing.assert_array_equal(w1_data.q[1:][same_episode], w1_data.q_next[:-1][same_episode])
    assert np.all(np.abs(w1_data.tau) <= model.TORQUE_LIMITS + 1e-12)


def test_collect_is_repeatable_for_a_seed():
    a = data.collect("W2", 0.5, seconds=3.0, seed=7)
    b = data.collect("W2", 0.5, seconds=3.0, seed=7)
    np.testing.assert_array_equal(a.q_next, b.q_next)


def test_split_keeps_episodes_whole_and_apart():
    d = data.collect("W1", 0.5, seconds=15.0, seed=1)
    train, val = data.split_by_episode(d, 0.2, seed=0)
    assert len(train) + len(val) == len(d)
    assert not set(train.episode) & set(val.episode)
    assert len(set(val.episode)) == 1


@pytest.mark.parametrize("region", ["train", "outside"])
def test_random_targets_are_reachable_and_in_their_region(region):
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, z = data.random_target(rng, region)
        assert data.REGIONS[region](x, z)
        assert 0.15 <= np.hypot(x, z) <= 0.55


def test_correct_belief_explains_recorded_data_exactly(w1_data):
    reference = PhysicsPredictor(model.Belief(0.8, **arm.WORLDS["W1"]))
    assert data.one_period_velocity_error(reference, w1_data) < 1e-9
    assert np.max(data.horizon_tip_errors(reference, w1_data)) < 1e-9


def test_c1_recovers_the_mass_when_only_the_mass_is_unknown(w1_data):
    assert estimate.payload_mass(w1_data) == pytest.approx(0.8, abs=1e-4)


def test_c1_cannot_fully_explain_a_world_with_an_offset_payload():
    d = data.collect("W2", 1.0, seconds=6.0, seed=2)
    c1 = PhysicsPredictor(estimate.payload_mass(d))
    reference = PhysicsPredictor(model.Belief(1.0, **arm.WORLDS["W2"]))
    assert data.one_period_velocity_error(c1, d) > 100 * max(data.one_period_velocity_error(reference, d), 1e-9)
