"""M5 development: choose the network's size and training settings before the main comparison.

Run:  python experiments/m5_dev_choices.py     (uses all CPU cores, a few minutes)

Uses only development data that the main comparison never touches:
  training data   data seed 4 (the main comparison uses seeds 1-3)
  judged on       validation seed 500, inside the training region (the main comparison reports seed 1000)

For every setting, C2 and C3 are trained in all three worlds at 9, 30 and 90 s
of data. The setting with the lowest average log tip error after 0.2 s wins.
The winner is written to results/m5/dev_choice.json and then frozen.

Writes results/m5/dev_choices.csv and dev_choice.json.
"""

from __future__ import annotations

import csv
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lac import data, estimate, learned
from lac.predictors import PhysicsPredictor

WORLDS = ("W1", "W2", "W3")
PAYLOAD = 1.0
BUDGETS = (9.0, 30.0, 90.0)
DEV_DATA_SEED = 4
DEV_VALIDATION_SEED = 500
SETTINGS = [
    {"hidden": hidden, "learning_rate": lr, "max_epochs": 10000, "patience": patience}
    for hidden, lr, patience in itertools.product(((32, 32), (64, 64), (128, 128)), (1e-3, 3e-3), (300, 1000))
]
OUT = Path(__file__).resolve().parents[1] / "results" / "m5"


def evaluate(job) -> dict:
    index, setting, world, budget, method = job
    recorded = data.load_or_collect(world, PAYLOAD, max(BUDGETS), DEV_DATA_SEED)
    judged_on = data.load_or_collect(world, PAYLOAD, 30.0, DEV_VALIDATION_SEED)
    used = recorded.episodes(range(round(budget / data.EPISODE_SECONDS)))
    train, validation = data.split_by_episode(used, 0.2, seed=0)
    base = PhysicsPredictor(0.0 if method == "C2" else estimate.payload_mass(used))
    net = learned.train_residual(base, train, validation, seed=0, **setting)
    error = np.median(data.horizon_tip_errors(learned.ResidualPredictor(base, net), judged_on))
    return {"setting": index, "hidden": "x".join(map(str, setting["hidden"])), "learning_rate": setting["learning_rate"],
            "patience": setting["patience"], "world": world, "budget_s": budget, "method": method,
            "epochs": len(net.history["train"]), "median_tip_error_at_0.2s_cm": round(error * 100, 5)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for world in WORLDS:  # record the data once, before the workers start
        data.load_or_collect(world, PAYLOAD, max(BUDGETS), DEV_DATA_SEED)
        data.load_or_collect(world, PAYLOAD, 30.0, DEV_VALIDATION_SEED)
    jobs = [(i, s, w, b, m) for i, s in enumerate(SETTINGS) for w in WORLDS for b in BUDGETS for m in ("C2", "C3")]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(evaluate, jobs))
    with open(OUT / "dev_choices.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scores = []
    for i, setting in enumerate(SETTINGS):
        errors = [r["median_tip_error_at_0.2s_cm"] for r in rows if r["setting"] == i]
        hit_cap = sum(r["epochs"] >= setting["max_epochs"] for r in rows if r["setting"] == i)
        score = float(np.mean(np.log10(np.maximum(errors, 1e-4))))  # errors below 1e-4 cm count as 1e-4
        scores.append(score)
        print(f"setting {i:2d} {setting}: mean log10 error {score:+.3f}, median {np.median(errors):.3f} cm, "
              f"hit the epoch cap {hit_cap}/{len(errors)}")
    best = int(np.argmin(scores))
    with open(OUT / "dev_choice.json", "w", encoding="utf-8") as f:
        json.dump({"setting": SETTINGS[best], "mean_log10_error": scores[best]}, f, indent=2)
    print("chosen:", SETTINGS[best])


if __name__ == "__main__":
    main()
