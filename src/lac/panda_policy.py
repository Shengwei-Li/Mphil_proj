"""Imitation policy on the Panda (milestone M12c): demonstrations, observations, controller.

The same set-up as on the two-link arm (lac.demos, lac.policy), in three
dimensions:

* Demonstrations: the expert is the reference MPC told the real robot exactly
  (no payload, and the world's friction), on the unloaded robot, in the training
  region. Reaching episodes (3 s, a start and target as in lac.panda.make_tasks)
  and settling episodes (1.5 s, starting at rest with the tip 0.5 to 8 cm from
  the target). A settling episode's start pose is found by letting the expert
  reach the offset point first (2 s, not recorded). Every period records the
  joint angles, speeds, the expert's setpoint and the target.
* Observation: sin q, cos q, joint speeds, target minus tip (24 numbers); action:
  the expert's setpoints for this and the next 7 periods, relative to the current
  joint angles (8 x 7). On the two-link arm the action was the next joint angles,
  which a stiff PD loop tracks closely; the Panda's impedance loop lags its
  setpoint (more so at soft stiffness), so a policy sending demonstrated *angles*
  as setpoints moved too slowly and stopped short: unloaded on the 20 dev tasks,
  median 1.5 cm (stiff) and 5.8 cm (soft) from the target, against 0.19 and 0.18
  cm with the expert's setpoints (PROGRESS, M12c). CFDP and
  the nearest-neighbour policy take these through ``fit_pairs``; the target
  weighting of lac.policy applies to the 3 target numbers.
* ``PandaPolicyController`` sends the chunk's setpoints to the robot's joint
  impedance loop, configured for no load (the policy believes the hand is
  empty), and keeps every chunk for the monitors, as lac.policy.PolicyController.

The same recordings also train the learned monitor's network (lac.panda_monitor).
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from lac import data, panda
from lac.panda_control import PandaMPC

EPISODE_SECONDS = {"reach": 3.0, "settle": 1.5}
SETTLE_OFFSET = (0.005, 0.08)  # m, start tip to target in settling episodes
_KINEMATICS: dict = {}


def tip_of(q: np.ndarray) -> np.ndarray:
    """Forward kinematics of the tip; the controller's own (the geometry is known exactly)."""
    if "model" not in _KINEMATICS:
        _KINEMATICS["model"] = panda.build(panda.Belief(), gravcomp=False)
        _KINEMATICS["data"] = mujoco.MjData(_KINEMATICS["model"])
    m, d = _KINEMATICS["model"], _KINEMATICS["data"]
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return d.site("tip").xpos.copy()


def frame(q: np.ndarray, dq: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.concatenate([np.sin(q), np.cos(q), dq, target - tip_of(q)])


@dataclass
class PandaDemos:
    q: np.ndarray  # (N, 7) joint angles at the start of each period
    dq: np.ndarray  # (N, 7)
    setpoint: np.ndarray  # (N, 7) the expert's setpoint during the period
    target: np.ndarray  # (N, 3)
    episode: np.ndarray  # (N,)
    next_dq: np.ndarray  # (N, 7) joint speeds at the end of the period (for the monitor's network)


def demo_world(world: str) -> str:
    """W2 differs from W1 only in where a payload sits, so unloaded they are the same robot."""
    return "W1" if world == "W2" else world


def _episode(job) -> dict:
    world, gains, kind, seed = job
    rng = np.random.default_rng(seed)
    q_start, target = panda.make_tasks(1, seed=int(rng.integers(2**31)), region="train")[0]
    expert = PandaMPC(panda.world_belief(world, 0.0), seed=seed, gains=gains)
    if kind == "settle":
        direction = rng.normal(size=3)
        offset = direction / np.linalg.norm(direction) * rng.uniform(*SETTLE_OFFSET)
        q_start = panda.run_reach(expert, 0.0, q_start, target + offset, 2.0, world).q[-1]
    rollout = panda.run_reach(expert, 0.0, q_start, target, EPISODE_SECONDS[kind], world)
    edges = np.arange(0, len(rollout.q), panda.HOLD)
    return {"q": rollout.q[edges[:-1]], "dq": rollout.dq[edges[:-1]], "setpoint": rollout.setpoint[edges[:-1]],
            "target": np.tile(target, (len(edges) - 1, 1)), "next_dq": rollout.dq[edges[1:]]}


def record(world: str, gains: str, episodes: int = 200, settle_episodes: int = 200, seed: int = 8000) -> PandaDemos:
    jobs = [(world, gains, "reach", seed + i) for i in range(episodes)]
    jobs += [(world, gains, "settle", seed + 100_000 + i) for i in range(settle_episodes)]
    with ProcessPoolExecutor() as pool:
        parts = list(pool.map(_episode, jobs, chunksize=4))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["episode"] = np.concatenate([np.full(len(p["q"]), i) for i, p in enumerate(parts)])
    return PandaDemos(**out)


def load_or_record(world: str, gains: str = "stiff", episodes: int = 200, settle_episodes: int = 200,
                   seed: int = 8000) -> PandaDemos:
    path = data.CACHE / f"panda_demos_{demo_world(world)}_{gains}_{episodes}+{settle_episodes}ep_seed{seed}.npz"
    if path.exists():
        with np.load(path) as f:
            return PandaDemos(**{k: f[k] for k in f.files})
    demos = record(demo_world(world), gains, episodes, settle_episodes, seed)
    data.CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, **asdict(demos))
    return demos


def training_pairs(demos: PandaDemos, n_obs: int = 1, n_act: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """As lac.policy.training_pairs, but the chunk is the expert's next n_act setpoints minus the current angles."""
    windows, chunks = [], []
    for e in np.unique(demos.episode):
        sel = demos.episode == e
        q, dq, target, setpoint = demos.q[sel], demos.dq[sel], demos.target[sel], demos.setpoint[sel]
        frames = np.array([frame(q[t], dq[t], target[t]) for t in range(len(q))])
        n = len(q)
        for t in range(n):
            windows.append(frames[np.clip(np.arange(t - n_obs + 1, t + 1), 0, None)].ravel())
            chunks.append((setpoint[np.clip(np.arange(t, t + n_act), None, n - 1)] - q[t]).ravel())
    return np.array(windows), np.array(chunks)


def fit_policy(policy, demos: PandaDemos):
    """A lac.policy policy (CFDP or NearestNeighborPolicy) fitted to Panda demonstrations."""
    obs, act = training_pairs(demos, policy.n_obs, policy.n_act)
    return policy.fit_pairs(obs, act, target_dims=3)


class PandaPolicyController:
    """Runs an imitation policy on the Panda with the MPC's interface (start, setpoint, load, gains)."""

    def __init__(self, policy, n_exec: int = 4, seed: int = 0, gains: str = "stiff"):
        self.policy = policy
        self.n_exec = n_exec
        self.rng = np.random.default_rng(seed)
        self.gains = gains
        self.load = panda.Belief(0.0)  # the policy believes the hand is empty
        self.target: np.ndarray | None = None
        self._history: list[np.ndarray] = []
        self._period, self._calls = 0, 0
        self.chunks: list[np.ndarray] = []  # each (n_act, 7), absolute setpoints
        self.chunk_periods: list[int] = []
        self.solve_seconds: list[float] = []

    def start(self, target: np.ndarray, q: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=float)
        self._history = []
        self._period, self._calls = 0, 0
        self.chunks, self.chunk_periods = [], []

    def retarget(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype=float)

    def setpoint(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        if self._calls % panda.HOLD == 0:
            self._current = self._next_setpoint(np.asarray(q, float), np.asarray(dq, float))
        self._calls += 1
        return self._current

    def _next_setpoint(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        if self.target is None:
            raise RuntimeError("start must be called before acting")
        self._history.append(frame(q, dq, self.target))
        position = self._period % self.n_exec
        if position == 0:
            n_obs = self.policy.n_obs
            window = np.concatenate([self._history[max(len(self._history) - n_obs + i, 0)] for i in range(n_obs)])
            began = time.perf_counter()
            relative = self.policy.predict(window, self.rng)
            self.solve_seconds.append(time.perf_counter() - began)
            self.chunks.append(q + relative)
            self.chunk_periods.append(self._period)
        self._period += 1
        return self.chunks[-1][position]
