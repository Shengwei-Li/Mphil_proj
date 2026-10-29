"""M9b, step 1: train the Diffusion Policy for each world on M8's demonstrations (CPU).

Run:  python experiments/m9b_train.py     (three worlds side by side, about an hour on 16 cores)

Settings are lac.diffusion.DiffusionPolicy's defaults (the paper's, 100 epochs,
training seed 0), fixed before any run. The trained networks go to cache/dp/;
the training loss per epoch to results/m9b/train_loss.csv.
"""

from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from lac import demos
from lac.diffusion import DiffusionPolicy

WORLDS = ("W1", "W2", "W3")
OUT = Path(__file__).resolve().parents[1] / "results" / "m9b"


def train(world: str) -> tuple[str, list[float], float]:
    began = time.perf_counter()
    policy = DiffusionPolicy().fit(demos.load_or_record(world))
    return world, policy.losses, time.perf_counter() - began


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for world in WORLDS:
        demos.load_or_record(world)
    with ProcessPoolExecutor(len(WORLDS)) as pool:
        results = list(pool.map(train, WORLDS))
    with open(OUT / "train_loss.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["world", "epoch", "loss"])
        for world, losses, _ in results:
            writer.writerows([world, e + 1, round(loss, 5)] for e, loss in enumerate(losses))
    for world, losses, seconds in results:
        print(f"{world}: final loss {losses[-1]:.4f}, {seconds / 60:.1f} min (0 if loaded from the cache)")


if __name__ == "__main__":
    main()
