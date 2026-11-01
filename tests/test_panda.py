from importlib import resources

import mujoco
import numpy as np
import pytest

from lac import arm, model, panda

RNG_SETPOINTS = np.random.default_rng(0).normal(0.0, 0.2, size=(10, 7))  # 10 periods of jittered setpoints


def drive(robot: panda.Robot, setpoints: np.ndarray) -> np.ndarray:
    """Each setpoint for one period; returns the joint angles at the end of every period."""
    out = []
    for s in setpoints:
        for _ in range(panda.HOLD):
            robot.step(s)
        out.append(robot.q)
    return np.array(out)


def test_the_bridge_on_the_two_link_arm_mujoco_gravcomp_is_the_hand_written_gravity():
    # On the Panda the controller's gravity compensation comes from MuJoCo's gravcomp.
    # On the two-link arm the same quantity is lac.model.gravity, written by hand.
    spec = mujoco.MjSpec.from_string(resources.files("lac").joinpath("assets/two_link_arm.xml").read_text("utf-8"))
    for body in spec.bodies[1:]:
        body.gravcomp = 1.0  # must be set before compiling: MuJoCo enables gravcomp at compile time
    m = spec.compile()
    d = mujoco.MjData(m)
    arm.set_payload(m, 0.7)
    q = np.radians([30.0, 45.0])
    arm.set_joint_angles(m, d, q)
    np.testing.assert_allclose(d.qfrc_gravcomp, model.gravity(q, 0.7), atol=1e-12)


@pytest.mark.parametrize("payload", [0.0, 1.0, 2.5])
def test_correct_load_holds_any_pose_still(payload):
    robot = panda.Robot(payload, "W1", panda.Belief(payload))
    robot.place(panda.HOME + 0.3)
    drive(robot, np.tile(panda.HOME + 0.3, (25, 1)))
    np.testing.assert_allclose(robot.q, panda.HOME + 0.3, atol=1e-9)


def test_wrong_load_makes_the_arm_sag_more_with_more_payload():
    sag = []
    for payload in (1.0, 2.5):
        robot = panda.Robot(payload, "W1", panda.Belief(0.0))
        robot.place(panda.HOME)
        tip = robot.tip()
        drive(robot, np.tile(panda.HOME, (100, 1)))
        sag.append(tip[2] - robot.tip()[2])
    assert 0.002 < sag[0] < sag[1] < 0.05


@pytest.mark.parametrize("world", ["W1", "W2", "W3"])
def test_a_model_told_the_truth_predicts_the_robot_exactly(world):
    truth = panda.world_belief(world, 1.5)
    robot = panda.Robot(1.5, world, truth)
    robot.place(panda.HOME)
    setpoints = panda.HOME + RNG_SETPOINTS
    q_real = drive(robot, setpoints)
    tips, _ = panda.Predictor(truth).rollout(panda.HOME, np.zeros(7), setpoints[None])
    robot.place(q_real[-1])
    np.testing.assert_allclose(tips[0, -1], robot.tip(), atol=1e-12)
    wrong = panda.Predictor(panda.Belief(0.0)).rollout(panda.HOME, np.zeros(7), setpoints[None])[0]
    assert np.linalg.norm(wrong[0, -1] - robot.tip()) > 1e-3


def test_predictor_batch_and_threads_agree_with_one_at_a_time():
    setpoints = panda.HOME + np.stack([RNG_SETPOINTS, -RNG_SETPOINTS, 0 * RNG_SETPOINTS])
    one = [panda.Predictor().rollout(panda.HOME, np.zeros(7), s[None])[0][0] for s in setpoints]
    batch, speeds = panda.Predictor(threads=3).rollout(panda.HOME, np.zeros(7), setpoints)
    np.testing.assert_array_equal(batch, np.array(one))
    assert batch.shape == (3, 10, 3) and speeds.shape == (3, 10, 7)


def test_belief_changes_reuse_or_rebuild_the_model():
    p = panda.Predictor()
    before = p.model
    p.set_belief(panda.Belief(2.0))
    assert p.model is before and p.model.body("payload").mass[0] == 2.0
    p.set_belief(panda.world_belief("W3", 2.0))
    assert p.model is not before and p.model.dof_frictionloss[0] == 1.0


def test_run_reach_logs_every_step_and_never_moves_with_a_hold_controller_and_true_load():
    q, target = panda.make_tasks(1, seed=0)[0]
    r = panda.run_reach(panda.HoldController(panda.Belief(1.0)), 1.0, q, target, duration=0.2)
    assert len(r.time) == 100 and r.q.shape == (100, 7)
    np.testing.assert_allclose(r.q, np.tile(q, (100, 1)), atol=1e-9)
    np.testing.assert_allclose(r.error, np.linalg.norm(r.tip - target, axis=1))


def test_tasks_are_reproducible_and_inside_their_regions():
    a, b = panda.make_tasks(20, seed=3), panda.make_tasks(20, seed=3)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
    for region in panda.REGIONS:
        lo, hi = np.array(panda.REGIONS[region]).T
        for _, t in panda.make_tasks(20, seed=1, region=region):
            t = t if region == "train" else t * [1, np.sign(t[1]), 1]
            assert np.all(t >= lo) and np.all(t <= hi)


def test_tip_is_tool_length_beyond_the_flange():
    robot = panda.Robot()
    robot.place(panda.HOME)
    d = robot.data
    flange = d.site("attachment_site")
    axis = d.site_xmat[flange.id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(robot.tip(), flange.xpos + panda.TOOL_LENGTH * axis, atol=1e-12)
    assert robot.model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER


def test_soft_gains_quarter_the_stiffness_and_the_model_still_predicts_exactly():
    soft = panda.build(panda.Belief(), gravcomp=False, gains="soft")
    np.testing.assert_allclose(soft.actuator_gainprm[:, 0], panda.STIFFNESS / 4)
    np.testing.assert_allclose(-soft.actuator_biasprm[:, 2], panda.DAMPING / 2)
    truth = panda.world_belief("W3", 1.5)
    robot = panda.Robot(1.5, "W3", truth, gains="soft")
    robot.place(panda.HOME)
    setpoints = panda.HOME + RNG_SETPOINTS
    q_real = drive(robot, setpoints)
    tips, _ = panda.Predictor(truth, gains="soft").rollout(panda.HOME, np.zeros(7), setpoints[None])
    robot.place(q_real[-1])
    np.testing.assert_allclose(tips[0, -1], robot.tip(), atol=1e-12)


def test_soft_gains_sag_about_four_times_more():
    sag = {}
    for gains in ("stiff", "soft"):
        robot = panda.Robot(1.0, "W1", panda.Belief(0.0), gains=gains)
        robot.place(panda.HOME)
        tip = robot.tip()
        drive(robot, np.tile(panda.HOME, (150, 1)))
        sag[gains] = tip[2] - robot.tip()[2]
    assert 3.0 < sag["soft"] / sag["stiff"] < 5.0


def test_rollout_from_a_different_start_for_each_sequence_equals_one_at_a_time():
    p = panda.Predictor(panda.world_belief("W3", 0.5), gains="soft")
    rng = np.random.default_rng(5)
    qs, dqs = panda.HOME + rng.normal(0, 0.2, (4, 7)), rng.normal(0, 0.5, (4, 7))
    setpoints = qs[:, None, :] + rng.normal(0, 0.1, (4, 3, 7))
    tips, speeds = p.rollout(qs, dqs, setpoints)
    for i in range(4):
        t, s = p.rollout(qs[i], dqs[i], setpoints[i : i + 1])
        np.testing.assert_array_equal(tips[i], t[0])
        np.testing.assert_array_equal(speeds[i], s[0])
