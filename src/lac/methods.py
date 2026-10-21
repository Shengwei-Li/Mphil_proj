"""The five methods of the study, assembled from recorded data (milestones M5 and later).

Every method is a predictor handed to the same MPC:

    C0          physics model assuming no payload
    C1          physics model with the payload mass estimated from data
    C2          C0's model plus a residual network
    C3          C1's model plus a residual network
    reference   physics model told the real payload and world (an upper bound, not a method)

Networks are trained once (in M5a) with the settings frozen in
results/m5/dev_choice.json and stored in cache/nets/, so every later
experiment uses exactly the same trained models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lac import arm, data, estimate, learned, model
from lac.control import inverse_kinematics
from lac.predictors import PhysicsPredictor

ROOT = Path(__file__).resolve().parents[2]
NETS = ROOT / "cache" / "nets"
DEV_CHOICE = ROOT / "results" / "m5" / "dev_choice.json"
METHODS = ("C0", "C1", "C2", "C3", "reference")
RECORDED_SECONDS = 90.0  # every data budget is the first episodes of one 90 s recording


def frozen_setting() -> dict:
    setting = json.loads(DEV_CHOICE.read_text(encoding="utf-8"))["setting"]
    setting["hidden"] = tuple(setting["hidden"])
    return setting


def net_path(world: str, payload: float, budget: float, seed: int, method: str) -> Path:
    return NETS / f"{world}_{payload:g}kg_{budget:g}s_seed{seed}_{method}.npz"


def build_predictors(world: str, payload: float, budget: float, seed: int, train_missing: bool = False) -> dict:
    """All five predictors for one world, payload, data budget and data seed.

    A network missing from the cache is trained only if `train_missing` is set;
    otherwise that is an error, so later experiments cannot silently retrain.
    """
    recorded = data.load_or_collect(world, payload, RECORDED_SECONDS, seed)
    used = recorded.episodes(range(round(budget / data.EPISODE_SECONDS)))
    mass = estimate.payload_mass(used)
    predictors = {
        "C0": PhysicsPredictor(0.0),
        "C1": PhysicsPredictor(mass),
        "reference": PhysicsPredictor(model.Belief(payload_mass=payload, **arm.WORLDS[world])),
    }
    for method, base in (("C2", PhysicsPredictor(0.0)), ("C3", PhysicsPredictor(mass))):
        path = net_path(world, payload, budget, seed, method)
        if not path.exists():
            if not train_missing:
                raise FileNotFoundError(f"{path} is missing; run experiments/m5a_prediction.py first")
            train, validation = data.split_by_episode(used, 0.2, seed=0)
            path.parent.mkdir(parents=True, exist_ok=True)
            learned.train_residual(base, train, validation, seed=seed, **frozen_setting()).save(path)
        predictors[method] = learned.ResidualPredictor(base, learned.ResidualNet.load(path))
    return predictors


def make_tasks(n: int, seed: int, region: str, min_distance: float = 0.15) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reaching tasks (starting joint angles, target) with start tip and target both in `region`.

    The start is a random pose whose tip lies in the region (random elbow side);
    the target is at least `min_distance` metres away from the starting tip.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    while len(tasks) < n:
        start_tip, target = data.random_target(rng, region), data.random_target(rng, region)
        if np.linalg.norm(target - start_tip) < min_distance:
            continue
        q_start = inverse_kinematics(start_tip)[rng.integers(2)]
        tasks.append((q_start, target))
    return tasks


# --- M6: ensembles ------------------------------------------------------------

ENSEMBLE_SIZE = 5


def member_path(world: str, payload: float, budget: float, seed: int, method: str, member: int) -> Path:
    """Member 0 is the network M5 already trained; members 1-4 differ only in their random start."""
    if member == 0:
        return net_path(world, payload, budget, seed, method)
    return NETS / f"{world}_{payload:g}kg_{budget:g}s_seed{seed}_{method}_member{member}.npz"


def build_ensembles(world: str, payload: float, budget: float, seed: int, train_missing: bool = False) -> dict:
    """Ensemble predictors "C2-E" and "C3-E" on the same physics models and data as C2 and C3."""
    recorded = data.load_or_collect(world, payload, RECORDED_SECONDS, seed)
    used = recorded.episodes(range(round(budget / data.EPISODE_SECONDS)))
    mass = estimate.payload_mass(used)
    train, validation = data.split_by_episode(used, 0.2, seed=0)
    ensembles = {}
    for method, base in (("C2", PhysicsPredictor(0.0)), ("C3", PhysicsPredictor(mass))):
        nets = []
        for member in range(ENSEMBLE_SIZE):
            path = member_path(world, payload, budget, seed, method, member)
            if not path.exists():
                if not train_missing:
                    raise FileNotFoundError(f"{path} is missing; run the M6b experiment first")
                path.parent.mkdir(parents=True, exist_ok=True)
                learned.train_residual(base, train, validation, seed=seed + 100 * member, **frozen_setting()).save(path)
            nets.append(learned.ResidualNet.load(path))
        ensembles[f"{method}-E"] = learned.EnsemblePredictor(base, nets)
    return ensembles
