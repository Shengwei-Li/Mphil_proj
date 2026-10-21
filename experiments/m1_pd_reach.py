"""M1: how well does PD reach a target as the payload grows?

Run:  python experiments/m1_pd_reach.py

Three controllers share the same gains and differ only in gravity compensation:
  pd            none
  pd_gc_empty   computed for an unloaded arm (what the controller believes)
  pd_gc_true    computed for the real payload (reference only: it is told the answer)

Writes results/m1/summary.csv, error_vs_time.png and final_error_vs_payload.png.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lac.control import TORQUE_LIMITS, PDController
from lac.simulate import run_reach

KP, KD = 60.0, 6.0
Q_START = np.radians([-60.0, 60.0])
TARGET = np.array([0.35, 0.25])
DURATION = 3.0
PAYLOADS = (0.0, 0.5, 1.0, 1.5)
CONTROLLERS = {
    "pd": ("PD only", lambda payload: None),
    "pd_gc_empty": ("PD + gravity comp. (assumes no payload)", lambda payload: 0.0),
    "pd_gc_true": ("PD + gravity comp. (told the payload)", lambda payload: payload),
}
OUT = Path(__file__).resolve().parents[1] / "results" / "m1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (key, (label, assumed)) in zip(axes, CONTROLLERS.items()):
        for payload in PAYLOADS:
            rollout = run_reach(PDController(KP, KD, assumed(payload)), payload, Q_START, TARGET, DURATION)
            saturated = np.any(np.abs(rollout.torque) >= TORQUE_LIMITS - 1e-9, axis=1)
            rows.append(
                {
                    "controller": key,
                    "payload_kg": payload,
                    "final_error_m": round(rollout.final_error(), 5),
                    "peak_shoulder_torque_Nm": round(float(np.abs(rollout.torque[:, 0]).max()), 2),
                    "peak_elbow_torque_Nm": round(float(np.abs(rollout.torque[:, 1]).max()), 2),
                    "time_at_torque_limit_s": round(float(saturated.sum() * (rollout.time[1] - rollout.time[0])), 3),
                }
            )
            ax.plot(rollout.time, rollout.error * 100, label=f"{payload:g} kg")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_ylim(0, 12)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("tip error (cm)")
    axes[0].legend(title="real payload")
    fig.suptitle(f"Reach from rest, Kp = {KP:g}, Kd = {KD:g} (y-axis cut at 12 cm)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_time.png", dpi=130)

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for key, (label, _) in CONTROLLERS.items():
        ax.plot(PAYLOADS, [r["final_error_m"] * 100 for r in rows if r["controller"] == key], marker="o", label=label)
    ax.set_xlabel("real payload (kg)")
    ax.set_ylabel("final tip error (cm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "final_error_vs_payload.png", dpi=130)

    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
