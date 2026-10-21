"""M5a: how well do the learned models predict, compared with C0, C1 and the reference?

Run:  python experiments/m5a_prediction.py     (uses all CPU cores, a few minutes)

Needs results/m5/dev_choice.json from m5_dev_choices.py: the network settings
are frozen there and not changed here.

Same data as M4: 1 kg payload, data seeds 1-3, budgets 9, 30 and 90 s (nested),
validation seed 1000 inside the training region and 2000 outside it. Every
trained network is saved to cache/nets/ (see lac.methods) so M5b controls
with exactly these.

Writes results/m5/prediction.csv and prediction_vs_data.png.
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lac import data, methods

WORLDS = ("W1", "W2", "W3")
PAYLOAD = 1.0
BUDGETS = (9.0, 30.0, 90.0)
DATA_SEEDS = (1, 2, 3)
OUT = Path(__file__).resolve().parents[1] / "results" / "m5"
SETTING = methods.frozen_setting()


def evaluate(job) -> list[dict]:
    world, budget, seed = job
    predictors = methods.build_predictors(world, PAYLOAD, budget, seed, train_missing=True)
    validation = {
        "inside": data.load_or_collect(world, PAYLOAD, 30.0, 1000),
        "outside": data.load_or_collect(world, PAYLOAD, 15.0, 2000, region="outside"),
    }
    rows = []
    for name, predictor in predictors.items():
        for split, d in validation.items():
            rows.append({
                "world": world, "budget_s": budget, "data_seed": seed, "method": name, "validation": split,
                "velocity_rmse_rad_s": round(data.one_period_velocity_error(predictor, d), 6),
                "median_tip_error_at_0.2s_cm": round(float(np.median(data.horizon_tip_errors(predictor, d))) * 100, 5),
                "epochs": len(predictor.net.history["train"]) if hasattr(predictor, "net") else "",
            })
    return rows


def main() -> None:
    for world in WORLDS:  # record the data once, before the workers start
        for seed in DATA_SEEDS:
            data.load_or_collect(world, PAYLOAD, methods.RECORDED_SECONDS, seed)
        data.load_or_collect(world, PAYLOAD, 30.0, 1000)
        data.load_or_collect(world, PAYLOAD, 15.0, 2000, region="outside")
    jobs = [(w, b, s) for w in WORLDS for b in BUDGETS for s in DATA_SEEDS]
    with ProcessPoolExecutor() as pool:
        rows = [row for result in pool.map(evaluate, jobs) for row in result]
    with open(OUT / "prediction.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("settings:", SETTING)
    for split in ("inside", "outside"):
        print(f"--- median tip error after 0.2 s (cm), validation {split}, mean over data seeds [min-max]")
        for world in WORLDS:
            for budget in BUDGETS:
                line = f"{world} {budget:>4g} s"
                for method in ("C0", "C1", "C2", "C3", "reference"):
                    v = [r["median_tip_error_at_0.2s_cm"] for r in rows
                         if (r["world"], r["budget_s"], r["method"], r["validation"]) == (world, budget, method, split)]
                    line += f" | {method} {np.mean(v):7.3f} [{min(v):.3f}-{max(v):.3f}]"
                print(line)
    plot(rows)


def plot(rows) -> None:
    styles = {"C1": ("tab:orange", "o"), "C2": ("tab:green", "s"), "C3": ("tab:red", "^")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, split in enumerate(("inside", "outside")):
        for j, world in enumerate(WORLDS):
            ax = axes[i, j]
            for method, (color, marker) in styles.items():
                values = [[r["median_tip_error_at_0.2s_cm"] for r in rows
                           if (r["world"], r["budget_s"], r["method"], r["validation"]) == (world, b, method, split)]
                          for b in BUDGETS]
                mean = np.array([np.mean(v) for v in values])
                ax.plot(BUDGETS, np.maximum(mean, 1e-3), color=color, marker=marker, label=method)
                for b, v in zip(BUDGETS, values):
                    ax.plot([b] * len(v), np.maximum(v, 1e-3), color=color, marker=marker, alpha=0.3, linestyle="none",
                            markersize=4)
            c0 = np.mean([r["median_tip_error_at_0.2s_cm"] for r in rows
                          if (r["world"], r["method"], r["validation"]) == (world, "C0", split)])
            ax.axhline(c0, color="tab:blue", linestyle="--", linewidth=1, label="C0")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(BUDGETS, [f"{b:g}" for b in BUDGETS])
            ax.set_ylim(1e-3, 100)
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"{world}, validation {split} training region", fontsize=9)
            if i == 1:
                ax.set_xlabel("data (seconds of motion)")
            if j == 0:
                ax.set_ylabel("median tip error after 0.2 s (cm)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("1 kg payload. Lines: mean of 3 data seeds; faint dots: each seed. "
                 "Values below 0.001 cm drawn at 0.001. Reference: 0 everywhere", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "prediction_vs_data.png", dpi=130)


if __name__ == "__main__":
    main()
