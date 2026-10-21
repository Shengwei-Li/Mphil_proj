"""M3: what does an unknown payload cost the MPC, and how much room is there to fix it?

Run:  python experiments/m3_c0_baseline.py     (takes a few minutes)

Controllers, all with the settings fixed in M2b:
  c0            MPC whose model assumes no payload        (the baseline to improve on)
  oracle        MPC whose model is told the real payload  (reference only: an upper bound)
  pd_gc_empty   PD + gravity compensation for an unloaded arm (for comparison with M1)

Writes results/m3/runs.csv, summary.csv, final_error_vs_payload.png and error_vs_time.png.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lac import model
from lac.control import PDController, inverse_kinematics
from lac.mpc import MPCController
from lac.simulate import run_reach

# name: (starting joint angles in degrees, target (x, z) in metres). Same tasks as M2b.
TASKS = {
    "A": ((-60, 60), (0.35, 0.25)),
    "B": ((-90, 0), (0.30, 0.30)),
    "C": ((0, 90), (-0.20, 0.40)),
    "D": ((30, 45), (0.45, -0.10)),
    "E": ((-120, -30), (-0.35, 0.10)),
    "F": ((60, -60), (0.10, 0.50)),
}
PAYLOADS = (0.0, 0.5, 1.0, 1.5)
SEEDS = range(5)
DURATION = 3.0
SETTLED = 0.01  # metres
OUT = Path(__file__).resolve().parents[1] / "results" / "m3"
LABELS = {
    "c0": "C0: MPC, model assumes no payload",
    "oracle": "reference: MPC told the payload",
    "pd_gc_empty": "PD + gravity comp. (assumes no payload)",
}


def settle_time(rollout) -> float:
    above = np.nonzero(rollout.error > SETTLED)[0]
    if len(above) == 0:
        return 0.0
    return DURATION if above[-1] == len(rollout.error) - 1 else float(rollout.time[above[-1] + 1])


def one_period_prediction_error(controller: MPCController, rollout) -> float:
    """Median distance between where the model said the tip would be one period later and where it was."""
    predicted = np.array(controller.next_tip_predictions[:-1])
    actual = rollout.tip[controller.hold :: controller.hold][: len(predicted)]
    return float(np.median(np.linalg.norm(predicted - actual, axis=1)))


def check_targets_can_be_held() -> float:
    """Largest share of a torque limit needed to hold any target pose with the heaviest payload."""
    worst = 0.0
    for _, target in TASKS.values():
        for q in inverse_kinematics(np.array(target)):
            worst = max(worst, float(np.max(np.abs(model.gravity(q, max(PAYLOADS))) / model.TORQUE_LIMITS)))
    return worst


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    share = check_targets_can_be_held()
    print(f"holding any target with {max(PAYLOADS)} kg needs at most {share:.0%} of a torque limit")

    rows, curves = [], {}
    for task, (q_start, target) in TASKS.items():
        for payload in PAYLOADS:
            jobs = [("pd_gc_empty", "", PDController(60.0, 6.0, 0.0))]
            jobs += [("c0", seed, MPCController(assumed_payload=0.0, seed=seed)) for seed in SEEDS]
            jobs += [("oracle", seed, MPCController(assumed_payload=payload, seed=seed)) for seed in SEEDS]
            for name, seed, controller in jobs:
                rollout = run_reach(controller, payload, np.radians(q_start), np.array(target), DURATION)
                last = rollout.time >= DURATION - 0.5
                saturated = np.any(np.abs(rollout.torque) >= model.TORQUE_LIMITS - 1e-9, axis=1)
                is_mpc = isinstance(controller, MPCController)
                rows.append(
                    {
                        "controller": name,
                        "task": task,
                        "payload_kg": payload,
                        "seed": seed,
                        "final_error_cm": round(rollout.final_error() * 100, 4),
                        "final_height_offset_cm": round(float((rollout.tip[last, 1] - target[1]).mean()) * 100, 4),
                        "settle_time_s": round(settle_time(rollout), 3),
                        "time_at_torque_limit_s": round(float(saturated.sum()) * model.TIMESTEP, 3),
                        "one_period_prediction_error_cm": round(one_period_prediction_error(controller, rollout) * 100, 5)
                        if is_mpc
                        else "",
                    }
                )
                curves.setdefault((name, task, payload), []).append(rollout)
        print(f"task {task} done")

    with open(OUT / "runs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for name in LABELS:
        for payload in PAYLOADS:
            group = [r for r in rows if r["controller"] == name and r["payload_kg"] == payload]
            errors = [r["final_error_cm"] for r in group]
            prediction = [r["one_period_prediction_error_cm"] for r in group if r["one_period_prediction_error_cm"] != ""]
            summary.append(
                {
                    "controller": name,
                    "payload_kg": payload,
                    "runs": len(group),
                    "median_final_error_cm": round(float(np.median(errors)), 3),
                    "min_final_error_cm": round(min(errors), 3),
                    "max_final_error_cm": round(max(errors), 3),
                    "median_height_offset_cm": round(float(np.median([r["final_height_offset_cm"] for r in group])), 3),
                    "runs_not_settled": sum(r["settle_time_s"] >= DURATION for r in group),
                    "median_time_at_torque_limit_s": round(float(np.median([r["time_at_torque_limit_s"] for r in group])), 3),
                    "median_one_period_prediction_error_cm": round(float(np.median(prediction)), 4) if prediction else "",
                }
            )
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for name, label in LABELS.items():
        group = [r for r in summary if r["controller"] == name]
        median = np.array([r["median_final_error_cm"] for r in group])
        low = np.array([r["min_final_error_cm"] for r in group])
        high = np.array([r["max_final_error_cm"] for r in group])
        ax.errorbar(PAYLOADS, median, yerr=[median - low, high - median], marker="o", capsize=3, label=label)
    ax.set_xlabel("real payload (kg)")
    ax.set_ylabel("final tip error (cm)")
    ax.set_title("Median over 6 tasks (x 5 seeds for MPC); bars: best and worst run", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "final_error_vs_payload.png", dpi=130)

    fig, axes = plt.subplots(1, len(PAYLOADS), figsize=(14, 3.6), sharey=True)
    for ax, payload in zip(axes, PAYLOADS):
        for name, color in (("c0", "tab:blue"), ("oracle", "tab:orange")):
            for rollout in curves[(name, "B", payload)]:
                ax.plot(rollout.time, rollout.error * 100, color=color, linewidth=0.8, alpha=0.7)
        pd = curves[("pd_gc_empty", "B", payload)][0]
        ax.plot(pd.time, pd.error * 100, color="k", linestyle="--", linewidth=1.2)
        ax.set_title(f"real payload {payload:g} kg", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 150)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("tip error (cm)")
    for name, color in (("c0", "tab:blue"), ("oracle", "tab:orange")):
        axes[0].plot([], [], color=color, label=LABELS[name])
    axes[0].plot([], [], color="k", linestyle="--", label=LABELS["pd_gc_empty"])
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Task B, 5 seeds per MPC controller", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_time.png", dpi=130)


if __name__ == "__main__":
    main()
