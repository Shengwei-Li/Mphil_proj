import mujoco
import numpy as np
import pytest

from lac import arm, model


def random_states(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-np.pi, np.pi, size=(n, 2))
    dq = rng.uniform(-5.0, 5.0, size=(n, 2))
    tau = rng.uniform(-model.TORQUE_LIMITS, model.TORQUE_LIMITS, size=(n, 2))
    return q, dq, tau


def test_beliefs_match_the_unloaded_simulated_arm():
    sim, _ = arm.load_arm()
    for i, name in enumerate(("link1", "link2")):
        body = sim.body(name)
        assert body.mass[0] == pytest.approx(model.LINK_MASSES[i])
        assert body.ipos[0] == pytest.approx(model.LINK_COM[i])
        assert body.inertia[1] == pytest.approx(model.LINK_INERTIAS[i])
    np.testing.assert_allclose(sim.dof_damping, model.JOINT_DAMPING)
    assert sim.opt.timestep == pytest.approx(model.TIMESTEP)


@pytest.mark.parametrize("payload", [0.0, 0.5, 1.5])
def test_each_term_matches_the_simulator(payload):
    sim, data = arm.load_arm()
    arm.set_payload(sim, payload)
    dense = np.zeros((2, 2))
    for q, dq, _ in zip(*random_states(20)):
        data.qpos[:], data.qvel[:] = q, dq
        mujoco.mj_forward(sim, data)
        for j, unit in enumerate(np.eye(2)):  # column j of M is M times the j-th unit vector
            column = np.zeros(2)
            mujoco.mj_mulM(sim, data, column, unit)
            dense[:, j] = column
        m11, m12, m22 = model.mass_matrix(q, payload)
        np.testing.assert_allclose([[m11, m12], [m12, m22]], dense, atol=1e-9)
        bias = model.velocity_terms(q, dq, payload) + model.gravity(q, payload)
        np.testing.assert_allclose(bias, data.qfrc_bias, atol=1e-8)


@pytest.mark.parametrize("payload", [0.0, 1.0])
def test_acceleration_matches_the_simulator(payload):
    sim, data = arm.load_arm()
    arm.set_payload(sim, payload)
    for q, dq, tau in zip(*random_states(20, seed=1)):
        data.qpos[:], data.qvel[:], data.ctrl[:] = q, dq, tau
        mujoco.mj_forward(sim, data)
        np.testing.assert_allclose(model.acceleration(q, dq, tau, payload), data.qacc, atol=1e-6)


def test_one_step_prediction_is_close_to_one_simulator_step():
    sim, data = arm.load_arm()
    arm.set_payload(sim, 0.7)
    for q, dq, tau in zip(*random_states(20, seed=2)):
        data.qpos[:], data.qvel[:], data.ctrl[:] = q, dq, tau
        mujoco.mj_step(sim, data)
        q_pred, dq_pred = model.predict(q, dq, tau, 0.7)
        np.testing.assert_allclose(q_pred, data.qpos, atol=1e-9)
        np.testing.assert_allclose(dq_pred, data.qvel, atol=1e-7)


def test_wrong_payload_gives_a_wrong_acceleration():
    q, dq, tau = np.array([0.3, 0.8]), np.zeros(2), np.zeros(2)
    assert np.linalg.norm(model.acceleration(q, dq, tau, 0.0) - model.acceleration(q, dq, tau, 1.0)) > 1.0


def test_functions_accept_batches():
    q, dq, tau = random_states(7, seed=3)
    batch = model.acceleration(q, dq, tau, 0.5)
    assert batch.shape == (7, 2)
    np.testing.assert_allclose(batch[3], model.acceleration(q[3], dq[3], tau[3], 0.5))
    qs, dqs = model.rollout(q, dq, np.zeros((10, 7, 2)), 0.5)
    assert qs.shape == dqs.shape == (11, 7, 2)


def oracle_belief(world: str, payload: float) -> model.Belief:
    return model.Belief(payload_mass=payload, **arm.WORLDS[world])


@pytest.mark.parametrize("world", ["W2", "W3"])
def test_generalised_terms_match_the_simulator(world):
    sim, data = arm.load_arm()
    arm.set_world(sim, world)
    arm.set_payload(sim, 1.2)
    belief = oracle_belief(world, 1.2)
    dense = np.zeros((2, 2))
    for q, dq, _ in zip(*random_states(20, seed=4)):
        data.qpos[:], data.qvel[:] = q, dq
        mujoco.mj_forward(sim, data)
        for j, unit in enumerate(np.eye(2)):
            column = np.zeros(2)
            mujoco.mj_mulM(sim, data, column, unit)
            dense[:, j] = column
        m11, m12, m22 = model.mass_matrix(q, belief)
        np.testing.assert_allclose([[m11, m12], [m12, m22]], dense, atol=1e-9)
        bias = model.velocity_terms(q, dq, belief) + model.gravity(q, belief)
        np.testing.assert_allclose(bias, data.qfrc_bias, atol=1e-8)


@pytest.mark.parametrize("world", ["W1", "W2", "W3"])
def test_reference_belief_predicts_each_world_step_for_step(world):
    sim, data = arm.load_arm()
    arm.set_world(sim, world)
    arm.set_payload(sim, 0.8)
    belief = oracle_belief(world, 0.8)
    for q, dq, tau in zip(*random_states(20, seed=5)):
        data.qpos[:], data.qvel[:], data.ctrl[:] = q, dq, tau
        arm.step(sim, data)
        q_pred, dq_pred = model.predict(q, dq, tau, belief)
        np.testing.assert_allclose(q_pred, data.qpos, atol=1e-9)
        np.testing.assert_allclose(dq_pred, data.qvel, atol=1e-7)


def test_mass_only_belief_cannot_describe_worlds_two_and_three():
    q, dq, tau = np.array([0.4, 0.9]), np.array([1.0, -2.0]), np.zeros(2)
    exact = model.acceleration(q, dq, tau, 1.0)
    for world in ("W2", "W3"):
        assert np.linalg.norm(model.acceleration(q, dq, tau, oracle_belief(world, 1.0)) - exact) > 0.5


def test_a_plain_number_still_means_a_point_mass_at_the_tip():
    q, dq, tau = np.array([0.4, 0.9]), np.array([1.0, -2.0]), np.array([3.0, 1.0])
    np.testing.assert_array_equal(model.acceleration(q, dq, tau, 0.7), model.acceleration(q, dq, tau, model.Belief(0.7)))
