"""M6a: does a learned model hurt control outside the region it was trained on?

Run:  python experiments/m6a_outside.py              (uses all CPU cores, several minutes)
      python experiments/m6a_outside.py --plot-only  (redraw figures from the saved tables)

Nothing is trained or tuned here. The same MPC, the same frozen networks
(cache/nets/) and the same five methods as M5b are run on two sets of tasks:
  inside   the 8 M5b test tasks (seed 5000); these runs must reproduce M5b exactly
  outside  8 new tasks (seed 6000) whose start and target both lie in the region
           the training data never aimed at (x <= -0.1 m)

Besides the final error, every run records what the MPC's own model expected:
each round, where the tip would be one period (0.02 s) later, compared with
where it actually was. Two numbers per run:
  prediction error  how far off that expectation was (median over the run)
  optimism          how much closer to the target the model expected to be than
                    the arm really was (mean over the run). Random mistakes average
                    out near zero; a model the MPC is exploiting is optimistic,
                    because the MPC picks exactly the actions the model is
                    wrongly hopeful about.

Writes results/m6/runs.csv, summary.csv, outside_vs_inside.png and worst_outside.png.
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

from lac import arm, methods
from lac.mpc import MPCController
from lac.simulate import run_reach

WORLDS = ("W1", "W2", "W3")
PAYLOAD = 1.0
BUDGETS = (9.0, 30.0, 90.0)
DATA_SEEDS = (1, 2, 3)
NO_DATA_SEEDS = (0, 1, 2)
TASKS = {"inside": methods.make_tasks(8, seed=5000, region="train"),
         "outside": methods.make_tasks(8, seed=6000, region="outside")}
DURATION = 3.0
SETTLED = 0.01  # m
LEARNING = ("C1", "C2", "C3")
OUT = Path(__file__).resolve().parents[1] / "results" / "m6"


def settle_time(rollout) -> float:
    above = np.nonzero(rollout.error > SETTLED)[0]
    if len(above) == 0:
        return 0.0
    return DURATION if above[-1] == len(rollout.error) - 1 else float(rollout.time[above[-1] + 1])


def expectation_metrics(controller: MPCController, rollout, target: np.ndarray) -> tuple[float, float]:
    """Median prediction error and mean optimism (both cm) of the model along the executed path."""
    predicted = np.array(controller.next_tip_predictions)
    rounds = np.arange(1, len(predicted) + 1) * controller.hold
    keep = rounds < len(rollout.tip)
    predicted, actual = predicted[keep], rollout.tip[rounds[keep]]
    error = np.linalg.norm(predicted - actual, axis=1)
    optimism = np.linalg.norm(actual - target, axis=1) - np.linalg.norm(predicted - target, axis=1)
    return float(np.median(error)) * 100, float(np.mean(optimism)) * 100


def run(world, region, name, predictor, budget, data_seed, mpc_seed, task):
    q_start, target = TASKS[region][task]
    controller = MPCController(predictor=predictor, seed=mpc_seed, show_plan=False)
    rollout = run_reach(controller, PAYLOAD, q_start, target, DURATION, world)
    prediction_error, optimism = expectation_metrics(controller, rollout, target)
    return rollout, {
        "world": world, "region": region, "method": name, "budget_s": budget, "data_seed": data_seed,
        "mpc_seed": mpc_seed, "task": task,
        "final_error_cm": round(rollout.final_error() * 100, 4),
        "settle_time_s": round(settle_time(rollout), 3),
        "prediction_error_cm": round(prediction_error, 5),
        "optimism_cm": round(optimism, 5),
    }


def run_group(job) -> list[dict]:
    """Same grouping and seeds as M5b, so the inside runs repeat M5b exactly."""
    world, budget, seed = job
    predictors = methods.build_predictors(world, PAYLOAD, budget or BUDGETS[0], seed or DATA_SEEDS[0])
    if budget is None:
        jobs = [(name, mpc_seed) for name in ("C0", "reference") for mpc_seed in NO_DATA_SEEDS]
    else:
        jobs = [(name, seed) for name in LEARNING]
    rows = []
    for region in TASKS:
        for name, mpc_seed in jobs:
            for task in range(len(TASKS[region])):
                _, row = run(world, region, name, predictors[name], budget if budget is not None else "",
                             seed if budget is not None else "", mpc_seed, task)
                rows.append(row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(w, None, None) for w in WORLDS] + [(w, b, s) for w in WORLDS for b in BUDGETS for s in DATA_SEEDS]
    with ProcessPoolExecutor() as pool:
        rows = [row for group in pool.map(run_group, jobs) for row in group]
    write("runs.csv", rows)
    summary = summarise(rows)
    write("summary.csv", summary)
    for row in summary:
        print(row)
    check_against_m5b(rows)
    plot(rows)


def check_against_m5b(rows: list[dict]) -> None:
    """The inside runs use M5b's tasks, seeds and networks, so their final errors must match M5b's."""
    path = OUT.parent / "m5" / "control_runs.csv"
    with open(path, encoding="utf-8") as f:
        m5b = {(r["world"], r["method"], r["budget_s"], r["data_seed"], r["mpc_seed"], r["task"]): r["final_error_cm"]
               for r in csv.DictReader(f)}
    mismatches = [r for r in rows if r["region"] == "inside" and
                  m5b[(r["world"], r["method"], str(r["budget_s"]), str(r["data_seed"]), str(r["mpc_seed"]),
                       str(r["task"]))] != str(r["final_error_cm"])]
    print(f"inside runs matching M5b exactly: {sum(r['region'] == 'inside' for r in rows) - len(mismatches)}"
          f" of {sum(r['region'] == 'inside' for r in rows)}")


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    groups = [(w, m, "") for w in WORLDS for m in ("C0", "reference")]
    groups += [(w, m, b) for w in WORLDS for m in LEARNING for b in BUDGETS]
    for region in TASKS:
        for world, method, budget in groups:
            g = [r for r in rows if (r["region"], r["world"], r["method"], r["budget_s"]) == (region, world, method, budget)]
            errors = [r["final_error_cm"] for r in g]
            out.append({
                "region": region, "world": world, "method": method, "budget_s": budget, "runs": len(g),
                "median_final_error_cm": round(float(np.median(errors)), 3),
                "p90_final_error_cm": round(float(np.percentile(errors, 90)), 3),
                "worst_final_error_cm": round(max(errors), 3),
                "runs_not_settled": sum(r["settle_time_s"] >= DURATION for r in g),
                "median_prediction_error_cm": round(float(np.median([r["prediction_error_cm"] for r in g])), 4),
                "median_optimism_cm": round(float(np.median([r["optimism_cm"] for r in g])), 4),
            })
    return out


def plot(rows: list[dict]) -> None:
    styles = {"C1": ("tab:orange", "o"), "C2": ("tab:green", "s"), "C3": ("tab:red", "^")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey="row")
    for j, world in enumerate(WORLDS):
        for i, (metric, label) in enumerate((("final_error_cm", "median final tip error (cm)"),
                                             ("optimism_cm", "median optimism (cm)"))):
            ax = axes[i, j]
            for method, (color, marker) in styles.items():
                for region, style in (("inside", "-"), ("outside", "--")):
                    values = [np.median([r[metric] for r in rows if (r["world"], r["method"], r["budget_s"], r["region"])
                                         == (world, method, b, region)]) for b in BUDGETS]
                    ax.plot(BUDGETS, values, color=color, marker=marker, linestyle=style,
                            label=f"{method} {region}", alpha=1.0 if region == "outside" else 0.6)
            ax.set_xscale("log")
            ax.set_xticks(BUDGETS, [f"{b:g}" for b in BUDGETS])
            ax.grid(alpha=0.3, which="both")
            if i == 0:
                ax.set_yscale("log")
                ax.set_title(world, fontsize=10)
            else:
                ax.axhline(0, color="k", linewidth=0.8)
                ax.set_xlabel("data (seconds of motion)")
            if j == 0:
                ax.set_ylabel(label)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Solid: test tasks inside the training region (as M5b). Dashed: tasks outside it. "
                 "Optimism > 0: the model expected to be closer to the target than the arm really was", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "outside_vs_inside.png", dpi=130)
    plot_worst(rows)


def plot_worst(rows: list[dict]) -> None:
    """Re-run (deterministically) the three worst learned-model runs outside, next to C1 on the same task."""
    learned_outside = [r for r in rows if r["region"] == "outside" and r["method"] in ("C2", "C3")]
    worst = sorted(learned_outside, key=lambda r: -float(r["final_error_cm"]))[:3]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, r in zip(axes, worst):
        world, budget, seed, task = r["world"], float(r["budget_s"]), int(float(r["data_seed"])), int(float(r["task"]))
        predictors = methods.build_predictors(world, PAYLOAD, budget, seed)
        q_start, target = TASKS["outside"][task]
        for name, color in ((r["method"], "tab:green" if r["method"] == "C2" else "tab:red"), ("C1", "tab:orange")):
            rollout, row = run(world, "outside", name, predictors[name], budget, seed, seed, task)
            ax.plot(rollout.tip[:, 0], rollout.tip[:, 1], color=color, linewidth=1.2,
                    label=f"{name}: final error {row['final_error_cm']:.1f} cm")
        start = arm.forward_kinematics(q_start)
        ax.plot(*start, "ko", markersize=5, label="start")
        ax.plot(*target, "*", color="tab:blue", markersize=14, label="target")
        ax.axvspan(-0.65, -0.1, color="0.92", zorder=0)
        ax.set_xlim(-0.65, 0.65)
        ax.set_ylim(-0.65, 0.65)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{world}, {budget:g} s of data, seed {seed}, task {task}", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("z (m)")
    for ax in axes:
        ax.set_xlabel("x (m)")
    fig.suptitle("Tip paths of the three worst learned-model runs outside the training region (grey), "
                 "with C1 on the same task", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "worst_outside.png", dpi=130)


def read(name: str) -> list[dict]:
    with open(OUT / name, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key not in ("world", "method", "region"):
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(read("runs.csv"))
    else:
        main()
