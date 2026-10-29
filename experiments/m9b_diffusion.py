"""M9b: do M8's and M9's findings hold for a standard neural Diffusion Policy?

Run:  python experiments/m9b_train.py              (first: train the policies, about an hour)
      python experiments/m9b_diffusion.py --dev    (development check, unloaded, before any test run)
      python experiments/m9b_diffusion.py          (test runs and monitors, all cores)
      python experiments/m9b_diffusion.py --analyse-only

The Diffusion Policy (lac.diffusion, the paper's settings, trained by
m9b_train.py) replaces CFDP in M9's experiment, unchanged in every other
respect: the same task splits (dev 20, calibration 40, inside 32 with 0-1 kg,
outside 32 with 0 and 1 kg), the same monitors, standardisation, conformal
thresholds and measurement noise, and the same analysis code (imported from
m9_monitors.py). Every run's final error also answers M8's question.
CFDP's and NN's numbers are read from results/m9/summary.csv, not rerun.

The expectations H13-H16 were written in docs/PROGRESS.md before the test runs.

Writes results/m9b/dev.csv (with --dev), runs.csv, summary.csv and
dp_vs_cfdp.png; the simulated runs go to cache/m9b_runs.pkl.
"""

from __future__ import annotations

import csv
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m9_monitors as m9  # noqa: E402

from lac import demos, methods, monitor  # noqa: E402
from lac.diffusion import DiffusionPolicy  # noqa: E402
from lac.policy import CFDP, PolicyController  # noqa: E402
from lac.simulate import run_reach  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m9b"
RUNS = ROOT / "cache" / "m9b_runs.pkl"
WORLDS = m9.WORLDS


def run_group(job) -> list[dict]:
    """Every run of one world, split and payload, recorded as in m9_monitors.run_group."""
    world, split, payload = job
    torch.set_num_threads(1)  # one simulation per core
    policy = DiffusionPolicy().fit(demos.load_or_record(world))
    runs = []
    for task, (q_start, target) in enumerate(m9.TASKS[split]):
        c = PolicyController(policy, seed=task)
        rollout = run_reach(c, payload, q_start, target, m9.DURATION, world)
        runs.append({"world": world, "policy": "DP", "split": split, "payload_kg": payload, "task": task,
                     "final_error_cm": rollout.final_error() * 100, "periods": monitor.periods(rollout, c.hold),
                     "chunks": c.chunks, "chunk_periods": list(c.chunk_periods),
                     "solve_ms": float(np.mean(c.solve_seconds)) * 1000})
    return runs


def dev_check() -> None:
    """Unloaded, on M9's 20 dev tasks: does the trained policy reach at all? CFDP alongside."""
    rows = []
    for world in WORLDS:
        d = demos.load_or_record(world)
        for name, policy in (("DP", DiffusionPolicy().fit(d)), ("CFDP", CFDP().fit(d))):
            errors = []
            for task, (q_start, target) in enumerate(m9.TASKS["dev"]):
                c = PolicyController(policy, seed=task)
                errors.append(run_reach(c, 0.0, q_start, target, m9.DURATION, world).final_error() * 100)
            rows.append({"world": world, "policy": name, "median_final_error_cm": round(float(np.median(errors)), 3),
                         "worst_final_error_cm": round(max(errors), 3), "runs_above_1cm": sum(e > 1 for e in errors)})
            print(rows[-1])
    write("dev.csv", rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--dev" in sys.argv:
        dev_check()
        return
    if "--analyse-only" in sys.argv:
        runs = pickle.loads(RUNS.read_bytes())
    else:
        for world in WORLDS:
            methods.build_predictors(world, 0.0, m9.RESIDUAL_BUDGET, m9.RESIDUAL_SEED, train_missing=True)
        jobs = [(w, s, p) for w in WORLDS for s in m9.TASKS for p in m9.PAYLOADS[s]]
        with ProcessPoolExecutor() as pool:
            runs = [r for g in pool.map(run_group, jobs) for r in g]
        RUNS.write_bytes(pickle.dumps(runs))
    m9.POLICIES = {"DP": DiffusionPolicy}  # m9's analysis loops over its POLICIES
    m9.add_signals(runs, 1.0)
    rows = m9.analyse(runs)
    write("runs.csv", rows)
    summary = m9.summarise(rows)
    for r in summary:
        r["mean_solve_ms"] = round(float(np.mean([x["solve_ms"] for x in runs if (x["split"], x["world"], x["payload_kg"])
                                                  == (r["split"], r["world"], r["payload_kg"])])), 1)
    write("summary.csv", summary)
    plot(summary)
    for r in summary:
        if r["alpha"] == 0.05:
            print(r["split"], r["world"], r["payload_kg"], r["median_final_error_cm"],
                  {m: r[f"alarm_rate_{m}"] for m in m9.COMBINED})


def write(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict]) -> None:
    """Alarm rate against payload inside (DP solid, CFDP dashed), and outside, alpha = 0.05."""
    with open(ROOT / "results" / "m9" / "summary.csv", encoding="utf-8") as f:
        cfdp = [r for r in csv.DictReader(f) if r["policy"] == "CFDP" and float(r["alpha"]) == 0.05]
    dp = [r for r in summary if r["alpha"] == 0.05]
    styles = {"physics": "tab:blue", "learned": "tab:red", "consistency": "tab:green"}
    fig, axes = plt.subplots(2, len(WORLDS), figsize=(13, 7))
    for j, world in enumerate(WORLDS):
        ax = axes[0, j]
        for m, color in styles.items():
            for rows, style, label in ((dp, "-", "DP"), (cfdp, "--", "CFDP")):
                g = sorted((r for r in rows if r["split"] == "inside" and r["world"] == world),
                           key=lambda r: float(r["payload_kg"]))
                ax.plot([float(r["payload_kg"]) for r in g], [float(r[f"alarm_rate_{m}"]) for r in g], style,
                        color=color, marker="o", markersize=3, label=f"{m}, {label}")
        ax.axhline(0.05, color="grey", linewidth=0.8, linestyle=":")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xticks([0, 0.05, 0.1, 0.25, 0.5, 1.0], ["0", "0.05", "0.1", "0.25", "0.5", "1"])
        ax.set_title(f"{world}, inside: alarm rate against payload", fontsize=10)
        ax.set_xlabel("payload (kg)")
        ax.grid(alpha=0.3)
        ax = axes[1, j]
        width = 0.35
        for k, (rows, label, shade) in enumerate(((dp, "DP", 1.0), (cfdp, "CFDP", 0.45))):
            r = next(r for r in rows if r["split"] == "outside" and r["world"] == world and float(r["payload_kg"]) == 0)
            ax.bar(np.arange(len(styles)) + (k - 0.5) * width, [float(r[f"alarm_rate_{m}"]) for m in styles], width,
                   color=list(styles.values()), alpha=shade, label=label, edgecolor="k")
        ax.set_xticks(range(len(styles)), list(styles))
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{world}, outside, unloaded (policies fail): alarm rate", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=8, title="dark: DP, light: CFDP", title_fontsize=8)
    fig.suptitle("Monitors watching the Diffusion Policy (DP) and CFDP, thresholds for a 5% false-alarm rate", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "dp_vs_cfdp.png", dpi=130)


if __name__ == "__main__":
    main()
