"""M5b: hand each method's predictor to the same MPC and compare how well the arm reaches.

Run:  python experiments/m5b_control.py              (uses all CPU cores, several minutes)
      python experiments/m5b_control.py --plot-only  (redraw figures from the saved table)

Frozen before this script was run: the MPC settings (M2b-M4), the network
settings (results/m5/dev_choice.json) and the trained networks (cache/nets/,
from m5a_prediction.py). Nothing here is tuned.

Test tasks are new: 8 reaches generated with seed 5000, start and target both
inside the training region. Development tasks A-F are not used.

Planning time is measured separately, one run at a time (W2, 30 s, data seed 1,
first two tasks), because times from the parallel runs are inflated by sharing
the CPU.

Writes results/m5/control_runs.csv, control_summary.csv, timing.csv and control_vs_data.png.
"""

from __future__ import annotations

import csv
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

WORLDS = ("W1", "W2", "W3")
PAYLOAD = 1.0
BUDGETS = (9.0, 30.0, 90.0)
DATA_SEEDS = (1, 2, 3)
TASKS = methods.make_tasks(8, seed=5000, region="train")
NO_DATA_SEEDS = (0, 1, 2)  # C0 and the reference use no data, so they get several MPC seeds instead
DURATION = 3.0
SETTLED = 0.01  # m
OUT = Path(__file__).resolve().parents[1] / "results" / "m5"
LEARNING = ("C1", "C2", "C3")


def settle_time(rollout) -> float:
    above = np.nonzero(rollout.error > SETTLED)[0]
    if len(above) == 0:
        return 0.0
    return DURATION if above[-1] == len(rollout.error) - 1 else float(rollout.time[above[-1] + 1])


def run_group(job) -> list[dict]:
    """All tasks for one world and one source of predictors (a data budget and seed, or none)."""
    world, budget, seed = job
    predictors = methods.build_predictors(world, PAYLOAD, budget or BUDGETS[0], seed or DATA_SEEDS[0])
    if budget is None:
        jobs = [(name, mpc_seed) for name in ("C0", "reference") for mpc_seed in NO_DATA_SEEDS]
    else:
        jobs = [(name, seed) for name in LEARNING]
    rows = []
    for name, mpc_seed in jobs:
        for task, (q_start, target) in enumerate(TASKS):
            controller = MPCController(predictor=predictors[name], seed=mpc_seed)
            rollout = run_reach(controller, PAYLOAD, q_start, target, DURATION, world)
            rows.append({
                "world": world, "method": name, "budget_s": budget if budget is not None else "",
                "data_seed": seed if budget is not None else "", "mpc_seed": mpc_seed, "task": task,
                "final_error_cm": round(rollout.final_error() * 100, 4),
                "settle_time_s": round(settle_time(rollout), 3),
            })
    return rows


def main() -> None:
    jobs = [(w, None, None) for w in WORLDS] + [(w, b, s) for w in WORLDS for b in BUDGETS for s in DATA_SEEDS]
    with ProcessPoolExecutor() as pool:
        rows = [row for group in pool.map(run_group, jobs) for row in group]
    write("control_runs.csv", rows)
    summary = summarise(rows)
    write("control_summary.csv", summary)
    for row in summary:
        print(row)
    plot(rows)
    timing = measure_timing()
    write("timing.csv", timing)
    for row in timing:
        print(row)


def measure_timing() -> list[dict]:
    """Planning time per round with nothing else running: 2 tasks per method, one at a time."""
    predictors = methods.build_predictors("W2", PAYLOAD, 30.0, 1)
    rows = []
    for name in methods.METHODS:
        solve = []
        for q_start, target in TASKS[:2]:
            controller = MPCController(predictor=predictors[name], seed=0)
            run_reach(controller, PAYLOAD, q_start, target, 1.0, "W2")
            solve += controller.solve_seconds
        rows.append({"method": name, "rounds": len(solve), "median_solve_ms": round(1000 * float(np.median(solve)), 2),
                     "p95_solve_ms": round(1000 * float(np.percentile(solve, 95)), 2), "period_ms": 20})
    return rows


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    groups = [(w, m, "") for w in WORLDS for m in ("C0", "reference")]
    groups += [(w, m, b) for w in WORLDS for m in LEARNING for b in BUDGETS]
    for world, method, budget in groups:
        g = [r for r in rows if (r["world"], r["method"], r["budget_s"]) == (world, method, budget)]
        errors = [r["final_error_cm"] for r in g]
        out.append({
            "world": world, "method": method, "budget_s": budget, "runs": len(g),
            "median_final_error_cm": round(float(np.median(errors)), 3),
            "p90_final_error_cm": round(float(np.percentile(errors, 90)), 3),
            "worst_final_error_cm": round(max(errors), 3),
            "runs_not_settled": sum(r["settle_time_s"] >= DURATION for r in g),
            "median_settle_time_s": round(float(np.median([r["settle_time_s"] for r in g])), 3),
        })
    return out


def plot(rows: list[dict]) -> None:
    styles = {"C1": ("tab:orange", "o"), "C2": ("tab:green", "s"), "C3": ("tab:red", "^")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, world in zip(axes, WORLDS):
        for method, (color, marker) in styles.items():
            per_budget = [[r["final_error_cm"] for r in rows
                           if (r["world"], r["method"], r["budget_s"]) == (world, method, b)] for b in BUDGETS]
            median = [np.median(v) for v in per_budget]
            p90 = [np.percentile(v, 90) for v in per_budget]
            ax.plot(BUDGETS, median, color=color, marker=marker, label=f"{method} median")
            ax.plot(BUDGETS, p90, color=color, marker=marker, linestyle=":", alpha=0.7, label=f"{method} 90th percentile")
        for method, color in (("C0", "tab:blue"), ("reference", "k")):
            v = [r["final_error_cm"] for r in rows if (r["world"], r["method"]) == (world, method)]
            ax.axhline(np.median(v), color=color, linestyle="--", linewidth=1, label=f"{method} median")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(BUDGETS, [f"{b:g}" for b in BUDGETS])
        ax.set_xlabel("data (seconds of motion)")
        ax.set_title(world, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("final tip error (cm)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("1 kg payload, 8 new test tasks inside the training region, 3 data seeds per point", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "control_vs_data.png", dpi=130)


def read(name: str) -> list[dict]:
    with open(OUT / name, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key not in ("world", "method"):
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(read("control_runs.csv"))
    else:
        main()
