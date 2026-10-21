import numpy as np
import pytest
import torch

from lac import data, learned
from lac.predictors import PhysicsPredictor


@pytest.fixture(scope="module")
def w2_split():
    recorded = data.collect("W2", 1.0, seconds=15.0, seed=11)
    return data.split_by_episode(recorded, 0.2, seed=0)


@pytest.fixture(scope="module")
def trained(w2_split):
    train, validation = w2_split
    base = PhysicsPredictor(0.0)
    return base, learned.train_residual(base, train, validation, hidden=(16, 16), max_epochs=400, patience=400)


def test_features_do_not_change_when_an_angle_goes_round_a_full_turn():
    q, dq, tau = np.array([0.3, -2.0]), np.array([1.0, 2.0]), np.array([3.0, -1.0])
    np.testing.assert_allclose(learned.features(q, dq, tau), learned.features(q + 2 * np.pi, dq, tau), atol=1e-12)
    assert learned.features(np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2))).shape == (5, 8)


def test_a_network_that_outputs_zero_leaves_the_physics_prediction_unchanged():
    zero = learned.ResidualNet([np.zeros((8, 4))], [np.zeros(4)], np.zeros(8), np.ones(8), np.zeros(4), np.ones(4))
    base = PhysicsPredictor(0.5)
    q, dq, tau = np.array([[0.2, 0.4]]), np.array([[1.0, -1.0]]), np.array([[2.0, 1.0]])
    expected = base.step(q, dq, tau, data.PERIOD)
    for got, want in zip(learned.ResidualPredictor(base, zero).step(q, dq, tau, data.PERIOD), expected):
        np.testing.assert_array_equal(got, want)


def test_predictor_refuses_a_step_length_it_was_not_trained_for(trained):
    base, net = trained
    with pytest.raises(ValueError):
        learned.ResidualPredictor(base, net).step(np.zeros(2), np.zeros(2), np.zeros(2), 0.05)


def test_numpy_network_matches_the_same_weights_in_pytorch(trained):
    _, net = trained
    layers = []
    for i, (w, b) in enumerate(zip(net.weights, net.biases)):
        linear = torch.nn.Linear(*w.shape).double()
        with torch.no_grad():
            linear.weight.copy_(torch.as_tensor(w.T))
            linear.bias.copy_(torch.as_tensor(b))
        layers += [linear] + ([torch.nn.Tanh()] if i < len(net.weights) - 1 else [])
    x = np.random.default_rng(0).normal(size=(20, 8))
    with torch.no_grad():
        expected = torch.nn.Sequential(*layers)(torch.as_tensor((x - net.input_mean) / net.input_std)).numpy()
    np.testing.assert_allclose(net(x), expected * net.output_std + net.output_mean, rtol=1e-9, atol=1e-12)


def test_training_reduces_the_error_on_held_out_episodes(w2_split, trained):
    _, validation = w2_split
    base, net = trained
    before = data.one_period_velocity_error(base, validation)
    after = data.one_period_velocity_error(learned.ResidualPredictor(base, net), validation)
    assert after < 0.5 * before


def test_saved_network_loads_back_identically(trained, tmp_path):
    _, net = trained
    net.save(tmp_path / "net.npz")
    loaded = learned.ResidualNet.load(tmp_path / "net.npz")
    x = np.random.default_rng(1).normal(size=(5, 8))
    np.testing.assert_array_equal(loaded(x), net(x))


def test_ensemble_of_identical_members_equals_the_single_network(trained):
    base, net = trained
    ensemble = learned.EnsemblePredictor(base, [net] * 3)
    q, dq, tau = np.array([[0.2, 0.4]]), np.array([[1.0, -1.0]]), np.array([[2.0, 1.0]])
    single = learned.ResidualPredictor(base, net).step(q, dq, tau, data.PERIOD)
    for got, want in zip(ensemble.step(q, dq, tau, data.PERIOD), single):
        np.testing.assert_allclose(got, want, atol=1e-12)
    np.testing.assert_allclose(ensemble.last_spread, 0.0, atol=1e-12)


def test_members_that_differ_disagree_and_are_averaged(w2_split):
    train, validation = w2_split
    base = PhysicsPredictor(0.0)
    nets = [learned.train_residual(base, train, validation, hidden=(8,), max_epochs=50, patience=50, seed=s) for s in (0, 1)]
    ensemble = learned.EnsemblePredictor(base, nets)
    q, dq, tau = np.array([[0.2, 0.4], [1.0, -0.5]]), np.array([[1.0, -1.0], [0.0, 2.0]]), np.array([[2.0, 1.0], [0.0, 0.0]])
    _, dq_mean = ensemble.step(q, dq, tau, data.PERIOD)
    _, dq_a = learned.ResidualPredictor(base, nets[0]).step(q, dq, tau, data.PERIOD)
    _, dq_b = learned.ResidualPredictor(base, nets[1]).step(q, dq, tau, data.PERIOD)
    np.testing.assert_allclose(dq_mean, (dq_a + dq_b) / 2, atol=1e-12)
    assert np.all(ensemble.last_spread > 0)
