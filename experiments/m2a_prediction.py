"""M2a: how far ahead can the hand-written model predict, and what does a wrong payload cost?

Run:  python experiments/m2a_prediction.py

For each real payload, the arm performs one reach under PD with gravity
compensation told the payload (only to produce a sensible motion). The logged
torques are then replayed through the model, which never sees the real arm
after the starting state. Two models are compared:
  matched     assumes the real payload
  unloaded    assumes no payload

Writes results/m2a/summary.csv, open_loop_1kg.png and error_vs_horizon.png.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lac import arm, model
from lac.control import PDController
from lac.simulate import run_reach

Q_START = np.radians([-60.0, 60.0])
TARGET = np.array([0.35, 0.25])
PAYLOADS = (0.0, 0.5, 1.0, 1.5)
HORIZONS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)  # seconds of prediction
START_EVERY = 0.05  # seconds between prediction start points along the reach
MOVING_PHASE = 1.0  # seconds from the start of the reach in which start points are taken
OUT = Path(__file__).resolve().parents[1] / "results" / "m2a"


def tip(q: np.ndarray) -> np.ndarray:
    return np.stack([arm.forward_kinematics(qi) for qi in q.reshape(-1, 2)]).reshape(*q.shape[:-1], 2)


def horizon_errors(real, assumed_payload: float) -> dict[float, np.ndarray]:
    """Tip prediction error after each horizon, from many start points along one reach."""
    dt = model.TIMESTEP
    longest = round(max(HORIZONS) / dt)
    # Start points cover only the moving part of the reach; once the arm has settled
    # there is nothing left to predict.
    starts = np.arange(0, round(MOVING_PHASE / dt), round(START_EVERY / dt))
    # Torque sequences for all start points at once: shape (longest, n_starts, 2).
    torques = np.stack([real.torque[s : s + longest] for s in starts], axis=1)
    q_pred, _ = model.rollout(real.q[starts], real.dq[starts], torques, assumed_payload)
    errors = {}
    for h in HORIZONS:
        k = round(h / dt)
        errors[h] = np.linalg.norm(tip(q_pred[k]) - real.tip[starts + k], axis=-1)
    return errors


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    curves: dict[tuple[float, str], list[float]] = {}
    for payload in PAYLOADS:
        real = run_reach(PDController(60.0, 6.0, payload), payload, Q_START, TARGET, duration=2.0)
        for name, assumed in (("matched", payload), ("unloaded", 0.0)):
            errors = horizon_errors(real, assumed)
            curves[(payload, name)] = [float(np.median(errors[h])) for h in HORIZONS]
            for h in HORIZONS:
                rows.append(
                    {
                        "real_payload_kg": payload,
                        "model": name,
                        "horizon_s": h,
                        "median_tip_error_m": float(f"{np.median(errors[h]):.4g}"),
                        "max_tip_error_m": float(f"{errors[h].max():.4g}"),
                        "start_points": len(errors[h]),
                    }
                )
        if payload == 1.0:
            plot_open_loop(real, payload)

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    for payload in PAYLOADS:
        if payload > 0:
            ax.plot(HORIZONS, np.array(curves[(payload, "unloaded")]) * 100, marker="o",
                    label=f"model assumes 0 kg, real {payload:g} kg")
    # With the right payload the model reproduces the simulator almost to rounding
    # error, which cannot be drawn on a log axis; state the number instead.
    worst_matched = max(r["max_tip_error_m"] for r in rows if r["model"] == "matched")
    ax.set_title(f"With the real payload assumed, the largest error is {worst_matched * 100:.1e} cm", fontsize=9)
    ax.set_xlabel("prediction horizon (s)")
    ax.set_ylabel("median tip prediction error (cm)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    print(f"largest matched-model error over all payloads, horizons and start points: {worst_matched:.2e} m")
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_horizon.png", dpi=130)

    for key, values in curves.items():
        print(key, " ".join(f"{h:g}s:{v * 100:.4f}cm" for h, v in zip(HORIZONS, values)))


def plot_open_loop(real, payload: float) -> None:
    """One long prediction from the very first state: real motion vs both models."""
    steps = round(0.6 / model.TIMESTEP)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    t = real.time[: steps + 1]
    for name, assumed, style in (("model assumes 1 kg", payload, "--"), ("model assumes 0 kg", 0.0, ":")):
        q_pred, _ = model.rollout(real.q[0], real.dq[0], real.torque[:steps], assumed)
        for j, ax in enumerate(axes):
            ax.plot(t, np.degrees(q_pred[:, j]), style, linewidth=2, label=name)
    for j, (ax, joint) in enumerate(zip(axes, ("shoulder q1", "elbow q2"))):
        ax.plot(t, np.degrees(real.q[: steps + 1, j]), color="k", linewidth=1, label="real arm (1 kg payload)")
        ax.set_title(joint, fontsize=10)
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("joint angle (deg)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Same torques replayed through the model, starting from the same state", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "open_loop_1kg.png", dpi=130)


if __name__ == "__main__":
    main()
