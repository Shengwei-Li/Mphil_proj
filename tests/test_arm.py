import mujoco
import numpy as np
import pytest

from lac import arm

G = 9.81
L1, L2 = arm.LINK_LENGTHS
M1 = M2 = 1.0  # link masses in two_link_arm.xml


@pytest.mark.parametrize("q_deg", [(0, 0), (30, 45), (90, -90), (-120, 60), (170, 140)])
def test_simulator_tip_matches_two_link_formula(q_deg):
    model, data = arm.load_arm()
    q = np.radians(q_deg)
    arm.set_joint_angles(model, data, q)
    np.testing.assert_allclose(arm.tip_position(data), arm.forward_kinematics(q), atol=1e-9)


def test_positive_shoulder_angle_lifts_the_arm():
    model, data = arm.load_arm()
    arm.set_joint_angles(model, data, np.radians([90, 0]))
    np.testing.assert_allclose(arm.tip_position(data), [0.0, L1 + L2], atol=1e-9)


@pytest.mark.parametrize("payload", [0.0, 0.5, 1.5])
def test_holding_torque_of_horizontal_arm(payload):
    # Arm stretched out horizontally: each joint carries the weight beyond it times its lever arm.
    model, data = arm.load_arm()
    arm.set_payload(model, payload)
    arm.set_joint_angles(model, data, np.zeros(2))
    shoulder = G * (M1 * L1 / 2 + M2 * (L1 + L2 / 2) + payload * (L1 + L2))
    elbow = G * (M2 * L2 / 2 + payload * L2)
    np.testing.assert_allclose(arm.gravity_torque(model, data), [shoulder, elbow], rtol=1e-6)


def test_holding_torque_keeps_the_arm_still():
    model, data = arm.load_arm()
    arm.set_payload(model, 0.8)
    q = np.radians([30, 45])
    arm.set_joint_angles(model, data, q)
    data.ctrl[:] = arm.gravity_torque(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos, q, atol=1e-6)


def test_released_arm_falls_and_payload_changes_the_motion():
    def tip_after_release(payload: float) -> np.ndarray:
        model, data = arm.load_arm()
        arm.set_payload(model, payload)
        arm.set_joint_angles(model, data, np.radians([30, 45]))
        for _ in range(150):
            mujoco.mj_step(model, data)
        return arm.tip_position(data)

    start = arm.forward_kinematics(np.radians([30, 45]))
    empty, loaded = tip_after_release(0.0), tip_after_release(1.0)
    assert empty[1] < start[1] and loaded[1] < start[1]
    assert np.linalg.norm(empty - loaded) > 0.01


def test_payload_round_trip_and_validation():
    model, _ = arm.load_arm()
    arm.set_payload(model, 0.7)
    assert arm.get_payload(model) == pytest.approx(0.7)
    arm.set_payload(model, 0.0)
    assert arm.get_payload(model) == 0.0
    with pytest.raises(ValueError):
        arm.set_payload(model, -0.1)


def test_changing_payload_keeps_the_current_pose():
    model, data = arm.load_arm()
    q = np.radians([40, -20])
    arm.set_joint_angles(model, data, q)
    arm.set_payload(model, 1.0)
    np.testing.assert_allclose(data.qpos, q)


def test_worlds_change_only_what_they_say():
    model, _ = arm.load_arm()
    arm.set_world(model, "W3")
    np.testing.assert_allclose(model.body("payload").pos, [0.34, 0.0, 0.06])
    np.testing.assert_allclose(model.dof_damping, [0.4, 0.25])
    np.testing.assert_allclose(model.jnt_user[:, 0], [0.6, 0.3])
    arm.set_world(model, "W1")
    np.testing.assert_allclose(model.body("payload").pos, [0.3, 0.0, 0.0])
    np.testing.assert_allclose(model.dof_damping, [0.1, 0.1])
    np.testing.assert_allclose(model.jnt_user[:, 0], [0.0, 0.0])


def test_coulomb_friction_opposes_motion():
    model, data = arm.load_arm()
    arm.set_world(model, "W3")
    data.qvel[:] = [2.0, -2.0]
    tau = arm.friction_torque(model, data)
    assert tau[0] < -0.5 and tau[1] > 0.25
