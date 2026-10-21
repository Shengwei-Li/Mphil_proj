"""Motion data that every correction method learns from (milestone M4).

The arm, carrying its real payload in a chosen world, is driven by the
uncorrected MPC (C0) between random targets, with small random torque changes
added so the data covers more states. Every control period (0.02 s) we record
one transition:

    state at the start (q, dq)  +  torque held for the period  ->  state at the end

Data is measured in seconds of motion and always split by whole episodes,
never by single transitions: neighbouring transitions are nearly identical, so
splitting them would let a method be tested on what it was trained on.

The functions at the bottom measure how well a predictor explains recorded
data; they are how M4 and M5 compare models before any control is run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lac import arm, model
from lac.control import inverse_kinematics
from lac.mpc import MPCController, tip_of

PERIOD = 0.02  # s, one transition; equals the MPC's control period
CACHE = Path(__file__).resolve().parents[2] / "cache"  # recorded data, so experiments can share it
EPISODE_SECONDS = 3.0
RETARGET_SECONDS = 0.6  # a new random target this often within an episode
EXPLORATION_NOISE = (1.5, 0.75)  # N·m added to the MPC's torque, per joint

# Where targets are drawn from. "train" is the part of the workspace the data
# covers; "outside" is kept away from all training data for out-of-distribution tests.
REGIONS = {
    "train": lambda x, z: x >= 0.05,
    "outside": lambda x, z: x <= -0.1,
}


@dataclass
class Transitions:
    q: np.ndarray  # (N, 2) rad, at the start of the period
    dq: np.ndarray  # (N, 2) rad/s
    tau: np.ndarray  # (N, 2) N·m, held for the whole period
    q_next: np.ndarray  # (N, 2) at the end of the period
    dq_next: np.ndarray  # (N, 2)
    episode: np.ndarray  # (N,) episode number; transitions of one episode are consecutive

    def __len__(self) -> int:
        return len(self.q)

    @property
    def seconds(self) -> float:
        return len(self) * PERIOD

    def select(self, mask: np.ndarray) -> "Transitions":
        return Transitions(self.q[mask], self.dq[mask], self.tau[mask], self.q_next[mask], self.dq_next[mask],
                           self.episode[mask])

    def episodes(self, ids) -> "Transitions":
        return self.select(np.isin(self.episode, list(ids)))


def random_target(rng: np.random.Generator, region: str) -> np.ndarray:
    """A reachable target in the given region, not too close to the shoulder or the edge of reach."""
    inside = REGIONS[region]
    while True:
        radius, angle = rng.uniform(0.15, 0.55), rng.uniform(-np.pi, np.pi)
        x, z = radius * np.cos(angle), radius * np.sin(angle)
        if inside(x, z):
            return np.array([x, z])


def collect(world: str, payload: float, seconds: float, seed: int, region: str = "train") -> Transitions:
    """Record `seconds` of motion (rounded up to whole episodes) on the real arm."""
    rng = np.random.default_rng(seed)
    sim, data = arm.load_arm()
    arm.set_world(sim, world)
    arm.set_payload(sim, payload)
    hold = round(PERIOD / sim.opt.timestep)
    periods_per_episode = round(EPISODE_SECONDS / PERIOD)
    retarget_every = round(RETARGET_SECONDS / PERIOD)

    log = {name: [] for name in ("q", "dq", "tau", "q_next", "dq_next", "episode")}
    for episode in range(int(np.ceil(seconds / EPISODE_SECONDS))):
        solutions = inverse_kinematics(random_target(rng, region))
        arm.set_joint_angles(sim, data, solutions[rng.integers(2)])
        # The exploring controller only has to move the arm around, not predict exactly,
        # so it uses the cheaper 2-step integration (about four times faster).
        controller = MPCController(assumed_payload=0.0, substeps=2, seed=int(rng.integers(1 << 31)))
        controller.start(random_target(rng, region), data.qpos)
        for k in range(periods_per_episode):
            if k > 0 and k % retarget_every == 0:
                controller.retarget(random_target(rng, region))
            q, dq = data.qpos.copy(), data.qvel.copy()
            tau = controller.act(q, dq) + rng.normal(0.0, EXPLORATION_NOISE)
            tau = np.clip(tau, -model.TORQUE_LIMITS, model.TORQUE_LIMITS)
            data.ctrl[:] = tau
            for _ in range(hold):
                arm.step(sim, data)
            for name, value in (("q", q), ("dq", dq), ("tau", tau), ("q_next", data.qpos.copy()),
                                ("dq_next", data.qvel.copy()), ("episode", episode)):
                log[name].append(value)
    return Transitions(**{k: np.array(v) for k, v in log.items()})


def load_or_collect(world: str, payload: float, seconds: float, seed: int, region: str = "train") -> Transitions:
    """Same as collect, but saved to and reused from the cache folder.

    Collection is deterministic for given arguments, so a cached file is
    exactly what collect would return. Delete the folder after changing how
    data is collected.
    """
    path = CACHE / f"{world}_{payload:g}kg_{seconds:g}s_seed{seed}_{region}.npz"
    if path.exists():
        with np.load(path) as f:
            return Transitions(**{k: f[k] for k in f.files})
    recorded = collect(world, payload, seconds, seed, region)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(path, **asdict(recorded))
    return recorded


def split_by_episode(data: Transitions, validation_fraction: float, seed: int) -> tuple[Transitions, Transitions]:
    """Training and validation sets made of whole, disjoint episodes."""
    ids = np.unique(data.episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * validation_fraction))
    return data.episodes(ids[n_val:]), data.episodes(ids[:n_val])


# --- how well does a predictor explain recorded data? -------------------------


def one_period_velocity_error(predictor, data: Transitions) -> float:
    """Root-mean-square error of the predicted joint speeds after one period (rad/s)."""
    _, dq_pred = predictor.step(data.q, data.dq, data.tau, PERIOD)
    return float(np.sqrt(np.mean((dq_pred - data.dq_next) ** 2)))


def horizon_tip_errors(predictor, data: Transitions, periods: int = 10, every: int = 5) -> np.ndarray:
    """Tip errors (m) after predicting `periods` periods ahead with the recorded torques.

    Start points are taken every `every` periods inside each episode. With the
    default 10 periods this is the MPC's own look-ahead of 0.2 s.
    """
    starts = []
    for episode in np.unique(data.episode):
        index = np.nonzero(data.episode == episode)[0]
        starts.extend(index[: len(index) - periods + 1 : every])
    starts = np.array(starts)
    q, dq = data.q[starts], data.dq[starts]
    for k in range(periods):
        q, dq = predictor.step(q, dq, data.tau[starts + k], PERIOD)
    return np.linalg.norm(tip_of(q) - tip_of(data.q_next[starts + periods - 1]), axis=-1)
