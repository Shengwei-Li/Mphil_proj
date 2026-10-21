"""M6b: does an ensemble that knows where it is unsure make the learned models safer?

Run:  python experiments/m6b_ensemble.py              (uses all CPU cores, several minutes)
      python experiments/m6b_ensemble.py --plot-only  (redraw the figure from the saved tables)

Needs results/m6/spread_choice.json from m6b_dev_spread.py (the penalty weight,
chosen on development material and frozen) and results/m6/runs.csv from
m6a_outside.py (the single-network results on the same tasks).

For every world, data budget and data seed of M5/M6a, four more networks are
trained per method (same data, different random start), giving 5-member
ensembles. Two ways of using them are compared with the single networks:
  C2-E avg, C3-E avg   average of the five, no penalty
  C2-E pen, C3-E pen   average of the five, and the MPC is penalised for
                       steering into states where the five disagree
on exactly the inside and outside tasks of M6a, with the same MPC seeds.

Planning time per round is measured separately, one run at a time.

Writes results/m6/ensemble_runs.csv, ensemble_summary.csv, ensemble_timing.csv
and ensemble_vs_single.png.
"""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lac import methods
from lac.mpc import MPCController
from lac.simulate import run_reach

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m6a_outside import TASKS, expectation_metrics, settle_time  # noqa: E402  same tasks and measures as M6a

WORLDS = ("W1", "W2", "W3")
PAYLOAD = 1.0
BUDGETS = (9.0, 30.0, 90.0)
DATA_SEEDS = (1, 2, 3)
DURATION = 3.0
OUT = Path(__file__).resolve().parents[1] / "results" / "m6"
SPREAD_WEIGHT = json.loads((OUT / "spread_choice.json").read_text(encoding="utf-8"))["spread_weight"]
VARIANTS = {"avg": 0.0, "pen": SPREAD_WEIGHT}


def train(job) -> None:
    world, budget, seed = job
    methods.build_ensembles(world, PAYLOAD, budget, seed, train_missing=True)


def run_group(job) -> list[dict]:
    world, budget, seed = job
    ensembles = methods.build_ensembles(world, PAYLOAD, budget, seed)
    rows = []
    for name, predictor in ensembles.items():
        for variant, weight in VARIANTS.items():
            for region, tasks in TASKS.items():
                for task, (q_start, target) in enumerate(tasks):
                    controller = MPCController(predictor=predictor, seed=seed, show_plan=False, spread_weight=weight)
                    rollout = run_reach(controller, PAYLOAD, q_start, target, DURATION, world)
                    prediction_error, optimism = expectation_metrics(controller, rollout, target)
                    rows.append({
                        "world": world, "region": region, "method": f"{name} {variant}", "budget_s": budget,
                        "data_seed": seed, "mpc_seed": seed, "task": task,
                        "final_error_cm": round(rollout.final_error() * 100, 4),
                        "settle_time_s": round(settle_time(rollout), 3),
                        "prediction_error_cm": round(prediction_error, 5),
                        "optimism_cm": round(optimism, 5),
                    })
    return rows


def main() -> None:
    jobs = [(w, b, s) for w in WORLDS for b in BUDGETS for s in DATA_SEEDS]
    with ProcessPoolExecutor() as pool:
        list(pool.map(train, jobs))
        rows = [row for group in pool.map(run_group, jobs) for row in group]
    write("ensemble_runs.csv", rows)
    all_rows = rows + single_network_rows()
    summary = summarise(all_rows)
    write("ensemble_summary.csv", summary)
    for row in summary:
        print(row)
    timing = measure_timing()
    write("ensemble_timing.csv", timing)
    for row in timing:
        print(row)
    plot(all_rows)


def single_network_rows() -> list[dict]:
    """C1, C2 and C3 from M6a on the same tasks, for comparison."""
    with open(OUT / "runs.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] in ("C1", "C2", "C3")]
    for r in rows:
        for key in ("budget_s", "final_error_cm", "settle_time_s", "prediction_error_cm", "optimism_cm"):
            r[key] = float(r[key])
    return rows


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


NAMES = ("C1", "C2", "C2-E avg", "C2-E pen", "C3", "C3-E avg", "C3-E pen")


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for region in TASKS:
        for world in WORLDS:
            for method in NAMES:
                for budget in BUDGETS:
                    g = [r for r in rows if (r["region"], r["world"], r["method"], float(r["budget_s"]))
                         == (region, world, method, budget)]
                    errors = [float(r["final_error_cm"]) for r in g]
                    out.append({
                        "region": region, "world": world, "method": method, "budget_s": budget, "runs": len(g),
                        "median_final_error_cm": round(float(np.median(errors)), 3),
                        "p90_final_error_cm": round(float(np.percentile(errors, 90)), 3),
                        "worst_final_error_cm": round(max(errors), 3),
                        "runs_not_settled": sum(float(r["settle_time_s"]) >= DURATION for r in g),
                    })
    return out


def measure_timing() -> list[dict]:
    """Planning time per round with nothing else running (W2, 30 s, data seed 1, two inside tasks)."""
    predictors = methods.build_predictors("W2", PAYLOAD, 30.0, 1) | methods.build_ensembles("W2", PAYLOAD, 30.0, 1)
    rows = []
    for name, weight in (("C1", 0.0), ("C3", 0.0), ("C3-E", 0.0), ("C3-E", SPREAD_WEIGHT)):
        solve = []
        for q_start, target in TASKS["inside"][:2]:
            controller = MPCController(predictor=predictors[name], seed=0, show_plan=False, spread_weight=weight)
            run_reach(controller, PAYLOAD, q_start, target, 1.0, "W2")
            solve += controller.solve_seconds
        rows.append({"method": name + (" pen" if weight else ""), "rounds": len(solve),
                     "median_solve_ms": round(1000 * float(np.median(solve)), 2),
                     "p95_solve_ms": round(1000 * float(np.percentile(solve, 95)), 2), "period_ms": 20})
    return rows


def plot(rows: list[dict]) -> None:
    styles = {"C1": ("tab:orange", "o", "-"), "C2": ("tab:green", "s", ":"), "C2-E pen": ("tab:green", "s", "-"),
              "C3": ("tab:red", "^", ":"), "C3-E pen": ("tab:red", "^", "-")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, region in enumerate(TASKS):
        for j, world in enumerate(WORLDS):
            ax = axes[i, j]
            for method, (color, marker, style) in styles.items():
                values = [np.median([float(r["final_error_cm"]) for r in rows
                                     if (r["region"], r["world"], r["method"], float(r["budget_s"]))
                                     == (region, world, method, b)]) for b in BUDGETS]
                ax.plot(BUDGETS, values, color=color, marker=marker, linestyle=style, label=method)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(BUDGETS, [f"{b:g}" for b in BUDGETS])
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{world}, tasks {region} the training region", fontsize=9)
            if i == 1:
                ax.set_xlabel("data (seconds of motion)")
            if j == 0:
                ax.set_ylabel("median final tip error (cm)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Dotted: single network (M6a). Solid: 5-network ensemble with disagreement penalty "
                 f"(weight {SPREAD_WEIGHT:g}, chosen on development tasks)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ensemble_vs_single.png", dpi=130)


def read(name: str) -> list[dict]:
    with open(OUT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(read("ensemble_runs.csv") + single_network_rows())
    else:
        main()
