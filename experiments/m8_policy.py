"""M8: how much does an unknown payload hurt an imitation policy taught on the empty arm?

Run:  python experiments/m8_policy.py              (uses all CPU cores, several minutes)
      python experiments/m8_policy.py --plot-only  (redraw the figure from the saved tables)

The policies (lac.policy) imitate the reference MPC from demonstrations recorded
on the unloaded arm of each world (lac.demos): 200 reaching and 200 settling
episodes in the training region. Their settings were chosen on the dev tasks
(seed 5500, W1, no payload; see PROGRESS M8); nothing is tuned here.

  NN    copy the nearest demonstration's chunk (deterministic, one seed)
  CFDP  closed-form diffusion policy (stochastic, three seeds)

Both are run on M6a's test tasks, inside (seed 5000) and outside (seed 6000) the
training region, with no payload and with the 1 kg payload. The low-level
controller's gravity compensation assumes no payload, as the demonstrations did.

Besides the final error, every run records:
  tracking error   how far the arm was from the commanded joint position at the
                   end of each period (median, degrees). This is the gap a
                   physics-model monitor can see (M9).
  predict time     time to generate one chunk (median, ms); a chunk is due every
                   n_exec periods (80 ms).

M6a's MPC rows (C0, C3 with 90 s of data, reference; 1 kg) are copied into the
summary for comparison, not re-run.

Writes results/m8/runs.csv, summary.csv and policy_vs_mpc.png.
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

from lac import demos, methods
from lac.policy import CFDP, NearestNeighborPolicy, PolicyController
from lac.simulate import run_reach

WORLDS = ("W1", "W2", "W3")
PAYLOADS = (0.0, 1.0)
POLICIES = {"NN": (NearestNeighborPolicy, (0,)), "CFDP": (CFDP, (0, 1, 2))}
TASKS = {"inside": methods.make_tasks(8, seed=5000, region="train"),
         "outside": methods.make_tasks(8, seed=6000, region="outside")}
DURATION = 3.0
SETTLED = 0.01  # m
MPC_ROWS = (("C0", ""), ("C3", "90.0"), ("reference", ""))
OUT = Path(__file__).resolve().parents[1] / "results" / "m8"


def settle_time(rollout) -> float:
    above = np.nonzero(rollout.error > SETTLED)[0]
    if len(above) == 0:
        return 0.0
    return DURATION if above[-1] == len(rollout.error) - 1 else float(rollout.time[above[-1] + 1])


def tracking_error(controller: PolicyController, rollout) -> float:
    """Median distance (degrees) between the commanded and the reached joint position at each period's end."""
    ends = np.minimum((np.arange(len(controller.setpoints)) + 1) * controller.hold - 1, len(rollout.q) - 1)
    gap = np.linalg.norm(np.array(controller.setpoints) - rollout.q[ends], axis=1)
    return float(np.degrees(np.median(gap)))


def run_group(job) -> list[dict]:
    world, name = job
    cls, seeds = POLICIES[name]
    policy = cls().fit(demos.load_or_record(world))
    rows = []
    for region, tasks in TASKS.items():
        for payload in PAYLOADS:
            for seed in seeds:
                for task, (q_start, target) in enumerate(tasks):
                    controller = PolicyController(policy, seed=seed * 100 + task)
                    rollout = run_reach(controller, payload, q_start, target, DURATION, world)
                    rows.append({
                        "world": world, "region": region, "policy": name, "payload_kg": payload, "seed": seed,
                        "task": task,
                        "final_error_cm": round(rollout.final_error() * 100, 4),
                        "settle_time_s": round(settle_time(rollout), 3),
                        "tracking_error_deg": round(tracking_error(controller, rollout), 4),
                        "predict_ms": round(float(np.median(controller.solve_seconds)) * 1000, 2),
                    })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--plot-only" not in sys.argv:
        for world in WORLDS:  # record once, before the workers need them
            demos.load_or_record(world)
        with ProcessPoolExecutor() as pool:
            rows = [r for g in pool.map(run_group, [(w, n) for w in WORLDS for n in POLICIES]) for r in g]
        write("runs.csv", rows)
        summary = summarise(rows)
        write("summary.csv", summary)
        for row in summary:
            print(row)
    plot(read("summary.csv"))


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for region in TASKS:
        for world in WORLDS:
            for name in POLICIES:
                for payload in PAYLOADS:
                    g = [r for r in rows if (r["region"], r["world"], r["policy"], r["payload_kg"])
                         == (region, world, name, payload)]
                    errors = [r["final_error_cm"] for r in g]
                    out.append({
                        "region": region, "world": world, "method": name, "payload_kg": payload, "runs": len(g),
                        "median_final_error_cm": round(float(np.median(errors)), 3),
                        "p90_final_error_cm": round(float(np.percentile(errors, 90)), 3),
                        "worst_final_error_cm": round(max(errors), 3),
                        "runs_not_settled": sum(r["settle_time_s"] >= DURATION for r in g),
                        "median_tracking_error_deg": round(float(np.median([r["tracking_error_deg"] for r in g])), 4),
                        "median_predict_ms": round(float(np.median([r["predict_ms"] for r in g])), 2),
                    })
            for row in read("summary.csv", OUT.parent / "m6"):
                if (row["region"], row["world"]) == (region, world) and (row["method"], row["budget_s"]) in MPC_ROWS:
                    out.append({
                        "region": region, "world": world, "method": f"{row['method']} (M6a)", "payload_kg": 1.0,
                        "runs": row["runs"], "median_final_error_cm": row["median_final_error_cm"],
                        "p90_final_error_cm": row["p90_final_error_cm"],
                        "worst_final_error_cm": row["worst_final_error_cm"],
                        "runs_not_settled": row["runs_not_settled"], "median_tracking_error_deg": "",
                        "median_predict_ms": "",
                    })
    return out


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read(name: str, folder: Path = OUT) -> list[dict]:
    with open(folder / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot(summary: list[dict]) -> None:
    bars = [("NN", "0.0", "NN, no payload"), ("NN", "1.0", "NN, 1 kg"),
            ("CFDP", "0.0", "CFDP, no payload"), ("CFDP", "1.0", "CFDP, 1 kg"),
            ("C0 (M6a)", "1.0", "MPC C0, 1 kg"), ("C3 (M6a)", "1.0", "MPC C3 90 s, 1 kg"),
            ("reference (M6a)", "1.0", "MPC reference, 1 kg")]
    colors = ["#9ecae1", "#3182bd", "#a1d99b", "#31a354", "#bdbdbd", "#de2d26", "#000000"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    width = 0.8 / len(bars)
    for ax, region in zip(axes, TASKS):
        for k, ((method, payload, label), color) in enumerate(zip(bars, colors)):
            values = []
            for world in WORLDS:
                row = next(r for r in summary if (r["region"], r["world"], r["method"]) == (region, world, method)
                           and float(r["payload_kg"]) == float(payload))
                values.append(float(row["median_final_error_cm"]))
            ax.bar(np.arange(len(WORLDS)) + (k - len(bars) / 2 + 0.5) * width, values, width, color=color, label=label)
        ax.set_xticks(range(len(WORLDS)), WORLDS)
        ax.set_yscale("log")
        ax.set_title(f"tasks {region} the training region", fontsize=10)
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0].set_ylabel("median final tip error (cm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(bars), fontsize=8, frameon=False)
    fig.suptitle("Imitation policies taught on the empty arm, against M6a's MPC methods", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "policy_vs_mpc.png", dpi=130)


if __name__ == "__main__":
    main()
