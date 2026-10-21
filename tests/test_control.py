import numpy as np
import pytest

from lac import arm
from lac.control import PDController, gravity_model, inverse_kinematics, nearest_solution
from lac.model import LINK_MASSES, TORQUE_LIMITS
from lac.simulate import run_reach

Q_START = np.radians([-60.0, 60.0])
TARGET = np.array([0.35, 0.25])


def test_controller_beliefs_match_the_unloaded_simulated_arm():
    model, _ = arm.load_arm()
    assert model.body("link1").mass[0] == pytest.approx(LINK_MASSES[0])
    assert model.body("link2").mass[0] == pytest.approx(LINK_MASSES[1])
    np.testing.assert_allclose(model.actuator_ctrlrange[:, 1], TORQUE_LIMITS)


@pytest.mark.parametrize("xz", [(0.35, 0.25), (-0.2, 0.4), (0.1, -0.5), (0.45, 0.0)])
def test_both_inverse_kinematics_solutions_reach_the_point(xz):
    solutions = inverse_kinematics(np.array(xz))
    assert len(solutions) == 2
    for q in solutions:
        np.testing.assert_allclose(arm.forward_kinematics(q), xz, atol=1e-9)
    assert solutions[0][1] == pytest.approx(-solutions[1][1])


def test_out_of_reach_point_has_no_solution():
    assert inverse_kinematics(np.array([0.5, 0.5])) == []
    assert nearest_solution(np.array([0.5, 0.5]), np.zeros(2)) is None


@pytest.mark.parametrize("payload", [0.0, 0.5, 1.5])
def test_gravity_model_matches_the_simulator(payload):
    model, data = arm.load_arm()
    arm.set_payload(model, payload)
    rng = np.random.default_rng(0)
    for q in rng.uniform(-np.pi, np.pi, size=(20, 2)):
        arm.set_joint_angles(model, data, q)
        np.testing.assert_allclose(gravity_model(q, payload), arm.gravity_torque(model, data), atol=1e-6)


def test_torque_respects_the_limits():
    controller = PDController(kp=500.0, kd=0.0)
    controller.set_goal(np.array([3.0, -3.0]))
    np.testing.assert_allclose(controller.torque(np.zeros(2), np.zeros(2)), [20.0, -10.0])


def test_pd_alone_leaves_a_steady_offset_that_grows_with_payload():
    errors = [run_reach(PDController(60.0, 6.0), p, Q_START, TARGET).final_error() for p in (0.0, 1.0)]
    assert errors[0] > 0.01
    assert errors[1] > 1.5 * errors[0]


def test_correct_gravity_compensation_removes_the_offset():
    rollout = run_reach(PDController(60.0, 6.0, assumed_payload=1.0), 1.0, Q_START, TARGET)
    assert rollout.final_error() < 1e-3


def test_unloaded_gravity_compensation_is_exact_only_without_payload():
    def final(payload: float) -> float:
        return run_reach(PDController(60.0, 6.0, assumed_payload=0.0), payload, Q_START, TARGET).final_error()

    assert final(0.0) < 1e-3
    assert final(1.0) > 0.01


def test_logged_tip_belongs_to_the_logged_joint_angles():
    rollout = run_reach(PDController(60.0, 6.0), 0.5, Q_START, TARGET, duration=0.3)
    for q, tip in zip(rollout.q[::25], rollout.tip[::25]):
        np.testing.assert_allclose(tip, arm.forward_kinematics(q), atol=1e-12)


def test_rollout_is_repeatable():
    a = run_reach(PDController(60.0, 6.0), 0.5, Q_START, TARGET, duration=1.0)
    b = run_reach(PDController(60.0, 6.0), 0.5, Q_START, TARGET, duration=1.0)
    np.testing.assert_array_equal(a.tip, b.tip)
