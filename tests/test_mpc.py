import numpy as np
import pytest

from lac import arm, model
from lac.mpc import MPCController, tip_of
from lac.simulate import run_reach

Q_START = np.radians([-60.0, 60.0])
TARGET = np.array([0.35, 0.25])


def test_tip_of_matches_forward_kinematics_for_batches():
    q = np.random.default_rng(0).uniform(-np.pi, np.pi, size=(4, 5, 2))
    assert tip_of(q).shape == (4, 5, 2)
    np.testing.assert_allclose(tip_of(q)[2, 3], arm.forward_kinematics(q[2, 3]))


@pytest.fixture(scope="module")
def unloaded_reach():
    controller = MPCController(seed=0)
    return controller, run_reach(controller, 0.0, Q_START, TARGET, duration=2.0)


def test_unloaded_mpc_reaches_the_target(unloaded_reach):
    _, rollout = unloaded_reach
    assert rollout.final_error() < 0.005
    assert rollout.error[0] > 0.3


def test_torque_stays_within_limits_and_is_held_between_rounds(unloaded_reach):
    controller, rollout = unloaded_reach
    assert np.all(np.abs(rollout.torque) <= model.TORQUE_LIMITS + 1e-12)
    held = rollout.torque.reshape(-1, controller.hold, 2)
    np.testing.assert_array_equal(held, np.repeat(held[:, :1], controller.hold, axis=1))
    assert len(controller.solve_seconds) == len(held)


def test_plan_prediction_starts_at_the_current_tip():
    controller = MPCController(seed=0)
    controller.start(TARGET, Q_START)
    controller.torque(Q_START, np.zeros(2))
    assert controller.planned_tip.shape == (controller.steps + 1, 2)
    np.testing.assert_allclose(controller.planned_tip[0], arm.forward_kinematics(Q_START))


def test_same_seed_gives_the_same_reach():
    a = run_reach(MPCController(seed=3), 0.0, Q_START, TARGET, duration=0.4)
    b = run_reach(MPCController(seed=3), 0.0, Q_START, TARGET, duration=0.4)
    c = run_reach(MPCController(seed=4), 0.0, Q_START, TARGET, duration=0.4)
    np.testing.assert_array_equal(a.tip, b.tip)
    assert not np.array_equal(a.tip, c.tip)


def test_skipping_the_plan_drawing_changes_nothing_else():
    a, b = MPCController(seed=2), MPCController(seed=2, show_plan=False)
    ra = run_reach(a, 0.5, Q_START, TARGET, duration=0.4)
    rb = run_reach(b, 0.5, Q_START, TARGET, duration=0.4)
    np.testing.assert_array_equal(ra.tip, rb.tip)
    np.testing.assert_array_equal(np.array(a.next_tip_predictions), np.array(b.next_tip_predictions))


def test_disagreement_penalty_adds_weight_times_squared_spread_in_cm():
    controller = MPCController(seed=0, spread_weight=2.0)
    controller.start(TARGET, Q_START)
    tips = np.tile(TARGET, (controller.steps, 3, 1))
    speeds = np.zeros((controller.steps, 3, 2))
    spreads = np.full((controller.steps, 3), 0.01)  # 1 cm at every step
    without = controller._cost(tips, speeds)
    with_penalty = controller._cost(tips, speeds, spreads)
    np.testing.assert_allclose(with_penalty - without, 2.0 * controller.steps * 1.0)
