import numpy as np
import pytest

from lac import arm, data, methods


@pytest.mark.parametrize("region", ["train", "outside"])
def test_tasks_start_and_end_in_their_region(region):
    tasks = methods.make_tasks(10, seed=1, region=region)
    assert len(tasks) == 10
    for q_start, target in tasks:
        start_tip = arm.forward_kinematics(q_start)
        assert data.REGIONS[region](*start_tip) and data.REGIONS[region](*target)
        assert np.linalg.norm(target - start_tip) >= 0.15


def test_tasks_are_repeatable_for_a_seed():
    a, b = methods.make_tasks(3, seed=2, region="train"), methods.make_tasks(3, seed=2, region="train")
    for (qa, ta), (qb, tb) in zip(a, b):
        np.testing.assert_array_equal(qa, qb)
        np.testing.assert_array_equal(ta, tb)


def test_missing_network_is_an_error_unless_training_is_allowed():
    # 3 s is not one of the budgets M5a trains, so no network exists for it.
    with pytest.raises(FileNotFoundError):
        methods.build_predictors("W1", 1.0, 3.0, 1)
