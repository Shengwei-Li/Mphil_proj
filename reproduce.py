"""Re-run every experiment, then rebuild the figures, tables, report and video.

Usage (from this folder, with the project environment):
    python reproduce.py                  everything, in order (roughly 1.5 to 2 hours on a laptop)
    python reproduce.py --outputs-only   only figures, tables, report and video, from the saved results
    python reproduce.py --from m5a       start at a given step
    python reproduce.py --only m6a       run one step
    python reproduce.py --list           show the steps

Recorded data and trained networks are kept in cache/ and reused. Delete that
folder first for a fully fresh run (data collection is deterministic, so the
same data is recorded again).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# (name, what it does, rough time on a 16-thread laptop, command)
STEPS = [
    ("tests", "check the models and controllers", "1 min", [PY, "-m", "pytest", "-q"]),
    ("m1", "PD control under payload", "<1 min", [PY, "experiments/m1_pd_reach.py"]),
    ("m2a", "hand-written model against the simulator", "<1 min", [PY, "experiments/m2a_prediction.py"]),
    ("m2b", "sampling-based MPC without payload", "8 min", [PY, "experiments/m2b_mpc_unloaded.py"]),
    ("m3", "MPC with a wrong payload, and the reference", "10 min", [PY, "experiments/m3_c0_baseline.py"]),
    ("m4", "three worlds, data, payload-mass estimation", "6 min", [PY, "experiments/m4_worlds_and_c1.py"]),
    ("m5dev", "choose network settings on development data", "12 min", [PY, "experiments/m5_dev_choices.py"]),
    ("m5a", "train networks, compare prediction", "4 min", [PY, "experiments/m5a_prediction.py"]),
    ("m5b", "compare control inside the training region", "6 min", [PY, "experiments/m5b_control.py"]),
    ("m6a", "compare control inside and outside", "6 min", [PY, "experiments/m6a_outside.py"]),
    ("m6dev", "choose the ensemble penalty on development data", "10 min", [PY, "experiments/m6b_dev_spread.py"]),
    ("m6b", "train ensembles, compare", "25 min", [PY, "experiments/m6b_ensemble.py"]),
    ("m8", "imitation policies taught on the empty arm, under payload", "5 min", [PY, "experiments/m8_policy.py"]),
    ("m9", "monitors for the policy: residuals against consistency", "12 min", [PY, "experiments/m9_monitors.py"]),
    ("m9btrain", "train the Diffusion Policy (CPU)", "55 min", [PY, "experiments/m9b_train.py"]),
    ("m9b", "monitors watching the Diffusion Policy", "6 min", [PY, "experiments/m9b_diffusion.py"]),
    ("m10", "hand the policy over to the adaptive MPC on an alarm", "15 min", [PY, "experiments/m10_handover.py"]),
    ("m11a", "Panda platform: sag, exactness, speed", "<1 min", [PY, "experiments/m11a_platform.py"]),
    ("m11c", "Panda: C0, C1 and the reference", "15 min", [PY, "experiments/m11c_compare.py"]),
    ("m12a", "Panda at soft joint stiffness", "15 min", [PY, "experiments/m11c_compare.py", "--gains", "soft"]),
    ("m12demos", "record the Panda demonstrations", "60 min", [PY, "experiments/m12_demos.py"]),
    ("m12c", "CFDP and monitors on the Panda", "15 min", [PY, "experiments/m12c_monitors.py"]),
    ("m12ho", "handover on the Panda", "60 min", [PY, "experiments/m12c_handover.py"]),
    ("figures", "report figures", "1 min", [PY, "report/figures.py"]),
    ("readme", "README banner and results chart", "<1 min", [PY, "report/readme_figures.py"]),
    ("tables", "report tables", "<1 min", [PY, "report/tables.py"]),
    ("report", "compile the report (needs a LaTeX installation with latexmk)", "<1 min",
     ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-cd", "report/main.tex"]),
    ("video", "demo video", "1 min", [PY, "report/video.py"]),
]
OUTPUTS = ("figures", "readme", "tables", "report", "video")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--outputs-only", action="store_true")
    parser.add_argument("--from", dest="start")
    parser.add_argument("--only")
    args = parser.parse_args()
    names = [name for name, *_ in STEPS]

    if args.list:
        for name, what, duration, _ in STEPS:
            print(f"{name:8s} {duration:>7s}  {what}")
        return
    if args.only:
        selected = [args.only]
    elif args.outputs_only:
        selected = list(OUTPUTS)
    elif args.start:
        selected = names[names.index(args.start):]
    else:
        selected = names
    unknown = set(selected) - set(names)
    if unknown:
        sys.exit(f"unknown step(s): {', '.join(sorted(unknown))}; see --list")

    for name, what, duration, command in STEPS:
        if name not in selected:
            continue
        if command[0] == "latexmk" and shutil.which("latexmk") is None:
            print(f"[{name}] skipped: latexmk not found")
            continue
        print(f"[{name}] {what} (about {duration})", flush=True)
        began = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"[{name}] failed with exit code {result.returncode}")
        print(f"[{name}] done in {time.perf_counter() - began:.0f} s", flush=True)


if __name__ == "__main__":
    main()
