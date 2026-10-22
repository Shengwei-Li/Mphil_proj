"""Expert demonstrations on the unloaded arm, for the imitation policy (milestone M8).

The scenario: demonstrations were recorded before the arm picked anything up.
The expert is the reference MPC, which knows the real arm exactly (no payload,
and the world's friction if there is any), so the demonstrations show good
reaching motions for an empty hand.

Two kinds of episode, both starting at rest in the training region:

* reaching: a random start and a random target, 3 s;
* settling: the start tip a few centimetres (0.5 to 8 cm) from the target, 1.5 s.

Settling episodes were added because reaching episodes alone almost never show
the arm at rest a few centimetres short of the target (the expert passes there
at speed), so a policy that stopped there found only "already arrived, hold
still" demonstrations and stalled 1 to 5 cm away. They show how to close the
last few centimetres, the same idea as recovery data in DAgger or DART.

Every control period (0.02 s) we record the joint angles and speeds, the target
and the torque the expert held. How these become a policy's observations and
actions is decided in lac.policy.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from lac import arm, data, model
from lac.control import inverse_kinematics
from lac.mpc import MPCController
from lac.predictors import PhysicsPredictor

EPISODE_SECONDS = {"reach": 3.0, "settle": 1.5}
SETTLE_OFFSET = (0.005, 0.08)  # metres, start tip to target in settling episodes


@dataclass
class Demos:
    q: np.ndarray  # (N, 2) joint angles at the start of each period
    dq: np.ndarray  # (N, 2) joint speeds
    target: np.ndarray  # (N, 2) the episode's target (x, z)
    torque: np.ndarray  # (N, 2) torque the expert held during the period
    episode: np.ndarray  # (N,) episode number; periods of one episode are consecutive


def _episode(job) -> tuple[np.ndarray, ...]:
    world, kind, seed = job
    rng = np.random.default_rng(seed)
    sim, sim_data = arm.load_arm()
    arm.set_world(sim, world)
    arm.set_payload(sim, 0.0)
    expert = MPCController(predictor=PhysicsPredictor(model.Belief(0.0, **arm.WORLDS[world])),
                           seed=int(rng.integers(1 << 31)), show_plan=False)
    target = data.random_target(rng, "train")
    if kind == "reach":
        start = inverse_kinematics(data.random_target(rng, "train"))[rng.integers(2)]
    else:
        while True:
            angle = rng.uniform(-np.pi, np.pi)
            tip = target + rng.uniform(*SETTLE_OFFSET) * np.array([np.cos(angle), np.sin(angle)])
            solutions = inverse_kinematics(tip)
            if solutions and data.REGIONS["train"](*tip):
                start = solutions[rng.integers(2)]
                break
    arm.set_joint_angles(sim, sim_data, start)
    expert.start(target, sim_data.qpos)
    hold = round(data.PERIOD / sim.opt.timestep)
    q, dq, torque = [], [], []
    for _ in range(round(EPISODE_SECONDS[kind] / data.PERIOD)):
        tau = expert.act(sim_data.qpos.copy(), sim_data.qvel.copy())
        q.append(sim_data.qpos.copy())
        dq.append(sim_data.qvel.copy())
        torque.append(tau)
        sim_data.ctrl[:] = tau
        for _ in range(hold):
            arm.step(sim, sim_data)
    q = np.array(q)
    return q, np.array(dq), np.tile(target, (len(q), 1)), np.array(torque)


def record(world: str, episodes: int, settle_episodes: int, seed: int) -> Demos:
    """Record reaching then settling episodes in parallel; deterministic for a given seed."""
    seeds = np.random.default_rng(seed).integers(1 << 31, size=episodes + settle_episodes)
    kinds = ["reach"] * episodes + ["settle"] * settle_episodes
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_episode, [(world, k, int(s)) for k, s in zip(kinds, seeds)]))
    return Demos(q=np.concatenate([r[0] for r in results]), dq=np.concatenate([r[1] for r in results]),
                 target=np.concatenate([r[2] for r in results]), torque=np.concatenate([r[3] for r in results]),
                 episode=np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(results)]))


def load_or_record(world: str, episodes: int = 200, settle_episodes: int = 200, seed: int = 7000) -> Demos:
    """Same as record, but saved to and reused from the cache folder."""
    path = data.CACHE / f"demos_v3_{world}_{episodes}+{settle_episodes}ep_seed{seed}.npz"
    if path.exists():
        with np.load(path) as f:
            return Demos(**{k: f[k] for k in f.files})
    demos = record(world, episodes, settle_episodes, seed)
    data.CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, **asdict(demos))
    return demos
