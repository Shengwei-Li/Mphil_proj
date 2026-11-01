"""M12c, step 1: record the Panda demonstrations (the expert MPC on the unloaded robot).

Run:  python experiments/m12_demos.py     (all CPU cores, about 30 minutes)

One set per robot that differs when unloaded: W1 (also used for W2) and W3, each
at stiff and soft joint stiffness; 200 reaching and 200 settling episodes each
(lac.panda_policy). Cached in cache/panda_demos_*.npz.
"""

from __future__ import annotations

import time

from lac import panda_policy

if __name__ == "__main__":
    for gains in ("stiff", "soft"):
        for world in ("W1", "W3"):
            began = time.perf_counter()
            d = panda_policy.load_or_record(world, gains)
            print(f"{world} {gains}: {len(d.q)} periods, {d.episode.max() + 1} episodes, "
                  f"{(time.perf_counter() - began) / 60:.1f} min (0 if cached)")
