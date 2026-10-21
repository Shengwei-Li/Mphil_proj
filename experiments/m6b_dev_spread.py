"""M6b development: choose how strongly the MPC avoids ensemble disagreement.

Run:  python experiments/m6b_dev_spread.py     (uses all CPU cores, several minutes)

Development material only, never used for the reported results:
  data      data seed 4 (reported results use seeds 1-3)
  tasks     4 tasks inside the training region (seed 5500) and 4 outside (seed 6500);
            the reported tasks use seeds 5000 and 6000
  worlds    W2 and W3, where the networks have something to learn

For each penalty weight, C2-E and C3-E reach every development task at 9, 30
and 90 s of data. The weight with the lowest average log final error (inside
and outside together) is written to results/m6/spread_choice.json and frozen.
A weight of 0 means "average the five networks, no penalty".

Writes results/m6/spread_dev.csv and spread_choice.json.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lac import methods
from lac.mpc import MPCController
from lac.simulate import run_reach

WORLDS = ("W2", "W3")
PAYLOAD = 1.0
BUDGETS = (9.0, 30.0, 90.0)
DEV_DATA_SEED = 4
# The first run tried 0 to 300; 300 was best but at the edge and the differences were small,
# so the range was extended upwards (log of that run: spread_dev_log_first_range.txt).
WEIGHTS = (0.0, 0.3, 3.0, 30.0, 300.0, 3000.0, 30000.0)
TASKS = {"inside": methods.make_tasks(4, seed=5500, region="train"),
         "outside": methods.make_tasks(4, seed=6500, region="outside")}
OUT = Path(__file__).resolve().parents[1] / "results" / "m6"


def train(job) -> None:
    world, budget = job
    methods.build_ensembles(world, PAYLOAD, budget, DEV_DATA_SEED, train_missing=True)


def evaluate(job) -> list[dict]:
    world, budget, weight = job
    ensembles = methods.build_ensembles(world, PAYLOAD, budget, DEV_DATA_SEED)
    rows = []
    for name, predictor in ensembles.items():
        for region, tasks in TASKS.items():
            for task, (q_start, target) in enumerate(tasks):
                controller = MPCController(predictor=predictor, seed=0, show_plan=False, spread_weight=weight)
                rollout = run_reach(controller, PAYLOAD, q_start, target, 3.0, world)
                rows.append({"world": world, "budget_s": budget, "spread_weight": weight, "method": name,
                             "region": region, "task": task, "final_error_cm": round(rollout.final_error() * 100, 4)})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as pool:
        list(pool.map(train, [(w, b) for w in WORLDS for b in BUDGETS]))
        jobs = [(w, b, l) for w in WORLDS for b in BUDGETS for l in WEIGHTS]
        rows = [row for group in pool.map(evaluate, jobs) for row in group]
    with open(OUT / "spread_dev.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scores = []
    for weight in WEIGHTS:
        cells = {}
        for r in rows:
            if r["spread_weight"] == weight:
                cells.setdefault((r["world"], r["budget_s"], r["method"]), []).append(r["final_error_cm"])
        score = float(np.mean([np.log10(np.median(v)) for v in cells.values()]))
        scores.append(score)
        by_region = {region: np.median([r["final_error_cm"] for r in rows
                                        if r["spread_weight"] == weight and r["region"] == region])
                     for region in TASKS}
        print(f"weight {weight:6g}: mean log10 of cell medians {score:+.3f} | median inside {by_region['inside']:.3f} cm,"
              f" outside {by_region['outside']:.3f} cm")
    best = WEIGHTS[int(np.argmin(scores))]
    (OUT / "spread_choice.json").write_text(json.dumps({"spread_weight": best, "scores": dict(zip(map(str, WEIGHTS), scores))},
                                                       indent=2), encoding="utf-8")
    print("chosen spread weight:", best)


if __name__ == "__main__":
    main()
