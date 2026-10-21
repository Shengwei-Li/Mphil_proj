"""M4: three worlds, motion data, and C1 (estimate the payload mass).

Run:  python experiments/m4_worlds_and_c1.py              (takes several minutes)
      python experiments/m4_worlds_and_c1.py --plot-only  (redraw figures from the saved tables)

Part 1, gap check. In each world, with a 1 kg payload, the uncorrected MPC (C0)
is compared with the reference MPC that is told everything about the real arm.
Development tasks A and B only.

Part 2, C1. Data is recorded in each world and C1 estimates the payload mass
from it. Smaller data budgets are the first episodes of the largest one, so the
budgets are nested. Each predictor is then judged on separate validation data:
  inside   new episodes in the same region of the workspace as the training data
  outside  episodes with targets in a region the training data never aimed at

Only prediction is measured here; control with C1 is compared in M5.

Writes results/m4/gap_check.csv, estimates.csv, prediction.csv and two figures.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lac import arm, data, estimate, model
from lac.mpc import MPCController
from lac.predictors import PhysicsPredictor
from lac.simulate import run_reach

WORLDS = ("W1", "W2", "W3")
MAIN_PAYLOAD = 1.0
OTHER_PAYLOADS = (0.5, 1.5)
BUDGETS = (3.0, 9.0, 30.0, 90.0)  # seconds of motion; multiples of the 3 s episode
OTHER_BUDGET = 30.0
DATA_SEEDS = (1, 2, 3)
DEV_TASKS = {"A": ((-60, 60), (0.35, 0.25)), "B": ((-90, 0), (0.30, 0.30))}
CONTROL_SEEDS = range(3)
OUT = Path(__file__).resolve().parents[1] / "results" / "m4"


def reference(world: str, payload: float) -> PhysicsPredictor:
    """Told the real payload and world: an upper bound, not a method."""
    return PhysicsPredictor(model.Belief(payload_mass=payload, **arm.WORLDS[world]))


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gap_check() -> list[dict]:
    rows = []
    for world in WORLDS:
        for name in ("c0", "reference"):
            for task, (q_start, target) in DEV_TASKS.items():
                for seed in CONTROL_SEEDS:
                    predictor = PhysicsPredictor(0.0) if name == "c0" else reference(world, MAIN_PAYLOAD)
                    rollout = run_reach(MPCController(predictor=predictor, seed=seed), MAIN_PAYLOAD,
                                        np.radians(q_start), np.array(target), 3.0, world)
                    rows.append({"world": world, "controller": name, "task": task, "seed": seed,
                                 "final_error_cm": round(rollout.final_error() * 100, 4)})
        print(f"gap check {world} done", flush=True)
    return rows


def prediction_rows(world, payload, budget, seed, mass, validation) -> list[dict]:
    predictors = {"c0": PhysicsPredictor(0.0), "c1": PhysicsPredictor(mass), "reference": reference(world, payload)}
    rows = []
    for name, predictor in predictors.items():
        for split, d in validation.items():
            rows.append({
                "world": world, "payload_kg": payload, "budget_s": budget, "data_seed": seed, "predictor": name,
                "validation": split,
                "velocity_rmse_rad_s": round(data.one_period_velocity_error(predictor, d), 6),
                "median_tip_error_at_0.2s_cm": round(float(np.median(data.horizon_tip_errors(predictor, d))) * 100, 5),
            })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gap = gap_check()
    write("gap_check.csv", gap)

    estimates, prediction = [], []
    for world in WORLDS:
        for payload in (MAIN_PAYLOAD, *OTHER_PAYLOADS):
            validation = {
                "inside": data.collect(world, payload, 30.0, seed=1000),
                "outside": data.collect(world, payload, 15.0, seed=2000, region="outside"),
            }
            budgets = BUDGETS if payload == MAIN_PAYLOAD else (OTHER_BUDGET,)
            for seed in DATA_SEEDS:
                recorded = data.collect(world, payload, max(budgets), seed=seed)
                for budget in budgets:
                    train = recorded.episodes(range(round(budget / data.EPISODE_SECONDS)))
                    mass = estimate.payload_mass(train)
                    estimates.append({"world": world, "payload_kg": payload, "budget_s": budget, "data_seed": seed,
                                      "estimated_mass_kg": round(mass, 4)})
                    prediction += prediction_rows(world, payload, budget, seed, mass, validation)
            print(f"C1 {world} {payload} kg done", flush=True)
    write("estimates.csv", estimates)
    write("prediction.csv", prediction)
    plot(estimates, prediction)
    summarise(gap, estimates, prediction)


def summarise(gap, estimates, prediction) -> None:
    for world in WORLDS:
        for name in ("c0", "reference"):
            e = [r["final_error_cm"] for r in gap if r["world"] == world and r["controller"] == name]
            print(f"gap {world} {name:9s} median final error {np.median(e):.3f} cm (max {max(e):.3f})")
    for world in WORLDS:
        for payload in (MAIN_PAYLOAD, *OTHER_PAYLOADS):
            for budget in sorted({r["budget_s"] for r in estimates if r["payload_kg"] == payload}):
                m = [r["estimated_mass_kg"] for r in estimates
                     if (r["world"], r["payload_kg"], r["budget_s"]) == (world, payload, budget)]
                line = f"{world} {payload} kg {budget:>4g} s: mass {np.mean(m):.3f} ({min(m):.3f}-{max(m):.3f})"
                for name in ("c0", "c1", "reference"):
                    for split in ("inside", "outside"):
                        v = [r["median_tip_error_at_0.2s_cm"] for r in prediction
                             if (r["world"], r["payload_kg"], r["budget_s"], r["predictor"], r["validation"])
                             == (world, payload, budget, name, split)]
                        line += f" | {name} {split} {np.mean(v):.3f}"
                print(line)


def plot(estimates, prediction) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for world in WORLDS:
        rows = [r for r in estimates if r["world"] == world and r["payload_kg"] == MAIN_PAYLOAD]
        means = [np.mean([r["estimated_mass_kg"] for r in rows if r["budget_s"] == b]) for b in BUDGETS]
        low = [min(r["estimated_mass_kg"] for r in rows if r["budget_s"] == b) for b in BUDGETS]
        high = [max(r["estimated_mass_kg"] for r in rows if r["budget_s"] == b) for b in BUDGETS]
        ax.errorbar(BUDGETS, means, yerr=[np.subtract(means, low), np.subtract(high, means)], marker="o",
                    capsize=3, label=world)
    ax.axhline(MAIN_PAYLOAD, color="k", linestyle="--", linewidth=1, label="real payload mass")
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS, [f"{b:g}" for b in BUDGETS])
    ax.set_xlabel("data (seconds of motion)")
    ax.set_ylabel("C1 estimated payload mass (kg)")
    ax.set_title("Mean of 3 data seeds; bars: lowest and highest", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "estimated_mass.png", dpi=130)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    names = {"c0": "C0 (assumes no payload)", "c1": "C1 (estimated mass)"}
    for ax, split in zip(axes, ("inside", "outside")):
        for i, name in enumerate(names):
            values = []
            for world in WORLDS:
                v = [r["median_tip_error_at_0.2s_cm"] for r in prediction
                     if (r["world"], r["payload_kg"], r["budget_s"], r["predictor"], r["validation"])
                     == (world, MAIN_PAYLOAD, 30.0, name, split)]
                values.append(np.mean(v))
            x = np.arange(len(WORLDS)) + (i - 0.5) * 0.35
            bars = ax.bar(x, values, width=0.35, label=names[name])
            ax.bar_label(bars, fmt="%.1f", fontsize=8)
        ax.set_xticks(range(len(WORLDS)), WORLDS)
        ax.set_title(f"validation {split} the training region", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("median tip error after 0.2 s (cm)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{MAIN_PAYLOAD:g} kg payload, C1 fitted on 30 s of data (mean of 3 data seeds). "
                 "Reference model (told everything): 0.0 cm everywhere", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "prediction_error.png", dpi=130)


def read(name: str) -> list[dict]:
    with open(OUT / name, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(read("estimates.csv"), read("prediction.csv"))
    else:
        main()
