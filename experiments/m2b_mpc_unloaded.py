"""M2b: does the sampling MPC reach targets on the unloaded arm, and what do its settings do?

Run:  python experiments/m2b_mpc_unloaded.py

No payload anywhere in this experiment: the model the MPC uses is exact. What
is being tested is the controller itself.

Task A is the one the default settings were tuned on. Tasks B-F were not
looked at during tuning.

Writes results/m2b/runs.csv, summary.csv, error_vs_time.png and sweeps.png.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lac.control import PDController
from lac.mpc import MPCController
from lac.simulate import run_reach

# name: (starting joint angles in degrees, target (x, z) in metres)
TASKS = {
    "A": ((-60, 60), (0.35, 0.25)),
    "B": ((-90, 0), (0.30, 0.30)),
    "C": ((0, 90), (-0.20, 0.40)),
    "D": ((30, 45), (0.45, -0.10)),
    "E": ((-120, -30), (-0.35, 0.10)),
    "F": ((60, -60), (0.10, 0.50)),
}
DURATION = 3.0
SEEDS = range(5)
SWEEP_SEEDS = range(3)
HORIZONS = (0.1, 0.2, 0.3, 0.5)
SAMPLES = (32, 64, 256, 1024)
SETTLED = 0.01  # metres: "arrived" means the tip error stays below this
OUT = Path(__file__).resolve().parents[1] / "results" / "m2b"


def settle_time(rollout) -> float:
    """Time after which the tip error never exceeds SETTLED again (DURATION if it never settles)."""
    above = np.nonzero(rollout.error > SETTLED)[0]
    if len(above) == 0:
        return 0.0
    return DURATION if above[-1] == len(rollout.error) - 1 else float(rollout.time[above[-1] + 1])


def run(task: str, controller, label: str, setting: str, seed) -> tuple[dict, object]:
    q_start, target = TASKS[task]
    rollout = run_reach(controller, 0.0, np.radians(q_start), np.array(target), DURATION)
    solve = getattr(controller, "solve_seconds", None)
    row = {
        "controller": label,
        "setting": setting,
        "task": task,
        "seed": seed,
        "final_error_cm": round(rollout.final_error() * 100, 4),
        "worst_error_last_second_cm": round(float(rollout.error[rollout.time >= DURATION - 1].max()) * 100, 4),
        "settle_time_s": round(settle_time(rollout), 3),
        "median_solve_ms": round(1000 * float(np.median(solve)), 3) if solve else "",
    }
    return row, rollout


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}

    for task in TASKS:
        row, rollout = run(task, PDController(60.0, 6.0, 0.0), "pd_gc", "default", "")
        rows.append(row)
        curves[("pd_gc", task)] = [rollout]
        curves[("mpc", task)] = []
        for seed in SEEDS:
            row, rollout = run(task, MPCController(seed=seed), "mpc", "default", seed)
            rows.append(row)
            curves[("mpc", task)].append(rollout)

    for name, values in (("horizon", HORIZONS), ("samples", SAMPLES)):
        for value in values:
            for task in TASKS:
                for seed in SWEEP_SEEDS:
                    row, _ = run(task, MPCController(seed=seed, **{name: value}), "mpc", f"{name}={value}", seed)
                    rows.append(row)

    with open(OUT / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows)
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)

    plot_error_vs_time(curves)
    plot_sweeps(summary)


def summarise(rows: list[dict]) -> list[dict]:
    """One line per controller setting, pooled over tasks and seeds; tuned task A reported apart."""
    out = []
    keys = dict.fromkeys((r["controller"], r["setting"]) for r in rows)
    for controller, setting in keys:
        for scope, tasks in (("A (tuned on)", ("A",)), ("B-F (not tuned on)", tuple("BCDEF"))):
            group = [r for r in rows if (r["controller"], r["setting"]) == (controller, setting) and r["task"] in tasks]
            solve = [r["median_solve_ms"] for r in group if r["median_solve_ms"] != ""]
            out.append(
                {
                    "controller": controller,
                    "setting": setting,
                    "tasks": scope,
                    "runs": len(group),
                    "median_final_error_cm": round(float(np.median([r["final_error_cm"] for r in group])), 3),
                    "worst_final_error_cm": round(max(r["final_error_cm"] for r in group), 3),
                    "median_settle_time_s": round(float(np.median([r["settle_time_s"] for r in group])), 3),
                    "runs_not_settled": sum(r["settle_time_s"] >= DURATION for r in group),
                    "median_solve_ms": round(float(np.median(solve)), 2) if solve else "",
                }
            )
    return out


def plot_error_vs_time(curves) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for ax, task in zip(axes.flat, TASKS):
        for rollout in curves[("mpc", task)]:
            ax.plot(rollout.time, rollout.error * 100, color="tab:blue", linewidth=0.8, alpha=0.7)
        pd = curves[("pd_gc", task)][0]
        ax.plot(pd.time, pd.error * 100, color="k", linestyle="--", linewidth=1.2)
        ax.axhline(SETTLED * 100, color="gray", linewidth=0.6, linestyle=":")
        ax.set_title(f"task {task}" + (" (tuned on)" if task == "A" else ""), fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 150)
        ax.grid(alpha=0.3, which="both")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("tip error (cm)")
    axes[0, 0].plot([], [], color="tab:blue", label="MPC, 5 random seeds")
    axes[0, 0].plot([], [], color="k", linestyle="--", label="PD + gravity comp.")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Unloaded arm, exact model. Dotted line: 1 cm.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_time.png", dpi=130)


def plot_sweeps(summary: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, name, values, unit in ((axes[0], "horizon", HORIZONS, "s"), (axes[1], "samples", SAMPLES, "")):
        rows = [r for v in values for r in summary if r["setting"] == f"{name}={v}" and r["tasks"].startswith("B-F")]
        ax.plot(values, [r["median_final_error_cm"] for r in rows], marker="o", label="median final error (cm)")
        ax.plot(values, [r["worst_final_error_cm"] for r in rows], marker="^", linestyle=":", label="worst final error (cm)")
        ax.set_yscale("log")
        ax.set_xlabel(f"{name}{' (' + unit + ')' if unit else ''}")
        ax.set_ylabel("tip error (cm)")
        ax.grid(alpha=0.3, which="both")
        twin = ax.twinx()
        twin.plot(values, [r["median_solve_ms"] for r in rows], marker="s", color="tab:red", label="compute per round (ms)")
        twin.set_ylabel("compute per round (ms)", color="tab:red")
        if name == "samples":
            ax.set_xscale("log", base=2)
        lines = ax.get_lines() + twin.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], fontsize=7)
    fig.suptitle("Tasks B-F, 3 seeds each; other settings at their defaults", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "sweeps.png", dpi=130)


if __name__ == "__main__":
    main()
