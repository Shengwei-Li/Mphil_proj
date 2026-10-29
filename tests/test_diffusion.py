import numpy as np
import pytest
import torch

from lac import demos
from lac.diffusion import ConditionalUNet1D, DiffusionPolicy, Scaler, cosine_alpha_bar
from lac.policy import PolicyController, training_pairs


@pytest.fixture(scope="module")
def few_demos():
    return demos.record("W1", episodes=4, settle_episodes=4, seed=1)


def test_noise_schedule_goes_from_almost_clean_to_almost_pure_noise():
    ab = cosine_alpha_bar(100)
    assert ab[0] > 0.999 and ab[-1] < 1e-3 and np.all(np.diff(ab) < 0)


def test_scaler_maps_the_data_range_onto_minus_one_to_one_and_back():
    x = np.random.default_rng(0).normal(size=(50, 3)) * [1.0, 10.0, 0.01]
    s = Scaler.fit(x)
    z = s(x)
    np.testing.assert_allclose(z.min(axis=0), -1.0)
    np.testing.assert_allclose(z.max(axis=0), 1.0)
    np.testing.assert_allclose(s.inverse(z), x)


def test_unet_keeps_the_chunk_shape_and_uses_the_observation():
    torch.manual_seed(0)
    net = ConditionalUNet1D(action_dim=2, obs_dim=8)
    x, k = torch.randn(4, 8, 2), torch.tensor([0, 10, 50, 99])
    a, b = net(x, k, torch.zeros(4, 8)), net(x, k, torch.ones(4, 8))
    assert a.shape == (4, 8, 2) and not torch.allclose(a, b)


def test_training_lowers_the_loss_and_the_cache_reproduces_the_policy(few_demos, tmp_path, monkeypatch):
    monkeypatch.setattr("lac.diffusion.CACHE", tmp_path)
    p = DiffusionPolicy(epochs=6, batch=64, warmup=10, threads=2).fit(few_demos)
    assert p.losses[-1] < p.losses[0]
    window = training_pairs(few_demos, 1, 8)[0][5]
    again = DiffusionPolicy(epochs=6, batch=64, warmup=10).fit(few_demos)  # loaded from tmp_path
    np.testing.assert_array_equal(p.predict(window, np.random.default_rng(3)),
                                  again.predict(window, np.random.default_rng(3)))
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_predicted_chunks_stay_within_the_demonstrated_range(few_demos, tmp_path, monkeypatch):
    monkeypatch.setattr("lac.diffusion.CACHE", tmp_path)
    p = DiffusionPolicy(epochs=2, batch=64, warmup=10, threads=2).fit(few_demos)
    _, act = training_pairs(few_demos, 1, 8)
    rng = np.random.default_rng(0)
    for window in training_pairs(few_demos, 1, 8)[0][::40]:
        chunk = p.predict(window, rng).ravel()
        assert chunk.shape == (16,)
        assert np.all(chunk >= act.min(axis=0) - 1e-9) and np.all(chunk <= act.max(axis=0) + 1e-9)  # x0 is clipped


def test_runs_inside_the_policy_controller(few_demos, tmp_path, monkeypatch):
    monkeypatch.setattr("lac.diffusion.CACHE", tmp_path)
    from lac import arm
    from lac.simulate import run_reach

    c = PolicyController(DiffusionPolicy(epochs=1, batch=64, warmup=10, threads=2).fit(few_demos), seed=0)
    q = np.array([0.2, 1.2])
    r = run_reach(c, 0.0, q, arm.forward_kinematics(np.array([0.4, 1.0])), 0.4, "W1")
    assert len(c.chunks) == 5 and np.all(np.isfinite(r.q))
