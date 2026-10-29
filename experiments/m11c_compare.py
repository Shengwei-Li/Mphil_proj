"""M11c: do the two-link arm's C0 and C1 results hold on the Panda?

Run:  python experiments/m11c_compare.py              (all CPU cores, about 15 minutes)
      python experiments/m11c_compare.py --plot-only  (redraw the figure from the saved tables)

Methods (lac.panda_control.make_controller), settings frozen in M11b:

    C0         MPC believing there is no payload
    C1         MPC starting from 0 kg and re-estimating the payload every 10 periods
    reference  MPC told the true payload and world (an upper bound)

Test tasks (lac.panda.make_tasks): 16 inside the training region (seed 5000),
16 outside (seed 6000); worlds W1-W3; payloads 0, 1 and 2.5 kg; 3 s per reach;
MPC seed = task index. One run per process with one thread, all cores busy.
The expectations H9-H12 were written in docs/PROGRESS.md before this script.

Writes results/m11/runs.csv, summary.csv and compare.png.
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

from lac import panda
from lac.panda_control import make_controller

WORLDS = ("W1", "W2", "W3")
METHODS = ("C0", "C1", "reference")
PAYLOADS = (0.0, 1.0, 2.5)
TASKS = {"inside": panda.make_tasks(16, seed=5000, region="train"),
         "outside": panda.make_tasks(16, seed=6000, region="outside")}
DURATION = 3.0
SETTLED = 0.01  # m
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m11"


def settle_time(r: panda.Rollout) -> float:
    above = np.nonzero(r.error > SETTLED)[0]
    if len(above) == 0:
        return 0.0
    return DURATION if above[-1] == len(r.error) - 1 else float(r.time[above[-1] + 1])


def run_one(job) -> dict:
    split, world, method, payload, task = job
    q_start, target = TASKS[split][task]
    c = make_controller(method, world, payload, seed=task)
    r = panda.run_reach(c, payload, q_start, target, DURATION, world)
    masses = getattr(c, "mass_estimates", [])
    return {"split": split, "world": world, "method": method, "payload_kg": payload, "task": task,
            "final_error_cm": round(r.final_error() * 100, 4), "settle_time_s": round(settle_time(r), 3),
            "final_mass_estimate_kg": round(masses[-1][1], 4) if masses else "",
            "mean_solve_ms": round(float(np.mean(c.solve_seconds)) * 1000, 2)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--plot-only" not in sys.argv:
        jobs = [(s, w, m, p, t) for s in TASKS for w in WORLDS for m in METHODS for p in PAYLOADS
                for t in range(len(TASKS[s]))]
        with ProcessPoolExecutor() as pool:
            rows = list(pool.map(run_one, jobs, chunksize=4))
        write("runs.csv", rows)
        summary = summarise(rows)
        write("summary.csv", summary)
        for r in summary:
            print(r)
    plot(read("summary.csv"))


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for split in TASKS:
        for world in WORLDS:
            for payload in PAYLOADS:
                for method in METHODS:
                    g = [r for r in rows if (r["split"], r["world"], r["payload_kg"], r["method"])
                         == (split, world, payload, method)]
                    errors = [r["final_error_cm"] for r in g]
                    masses = [r["final_mass_estimate_kg"] for r in g if r["final_mass_estimate_kg"] != ""]
                    out.append({
                        "split": split, "world": world, "payload_kg": payload, "method": method, "runs": len(g),
                        "median_final_error_cm": round(float(np.median(errors)), 3),
                        "p90_final_error_cm": round(float(np.percentile(errors, 90)), 3),
                        "worst_final_error_cm": round(max(errors), 3),
                        "runs_not_settled": sum(r["settle_time_s"] >= DURATION for r in g),
                        "median_settle_time_s": round(float(np.median([r["settle_time_s"] for r in g])), 2),
                        "median_mass_estimate_kg": round(float(np.median(masses)), 4) if masses else "",
                        "worst_mass_error_kg": round(max(abs(m - payload) for m in masses), 4) if masses else "",
                        "mean_solve_ms": round(float(np.mean([r["mean_solve_ms"] for r in g])), 1),
                    })
    return out


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read(name: str) -> list[dict]:
    with open(OUT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot(summary: list[dict]) -> None:
    """Median final error per payload, method and world; inside and outside."""
    colors = {"C0": "#3182bd", "C1": "#de2d26", "reference": "#000000"}
    fig, axes = plt.subplots(2, len(WORLDS), figsize=(13, 7), sharey=True)
    width = 0.8 / len(METHODS)
    for i, split in enumerate(TASKS):
        for j, world in enumerate(WORLDS):
            ax = axes[i, j]
            for k, method in enumerate(METHODS):
                values = [float(next(r for r in summary if (r["split"], r["world"], r["method"]) == (split, world, method)
                                     and float(r["payload_kg"]) == p)["median_final_error_cm"]) for p in PAYLOADS]
                ax.bar(np.arange(len(PAYLOADS)) + (k - 1) * width, values, width, color=colors[method], label=method)
            ax.set_yscale("log")
            ax.set_xticks(range(len(PAYLOADS)), [f"{p:g} kg" for p in PAYLOADS])
            ax.set_title(f"{world}, {split}", fontsize=10)
            ax.grid(alpha=0.3, axis="y")
            if j == 0:
                ax.set_ylabel("median final tip error (cm)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Franka Panda: MPC believing no payload (C0), estimating it online (C1), told the truth (reference)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "compare.png", dpi=130)


if __name__ == "__main__":
    main()
