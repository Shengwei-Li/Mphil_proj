"""LaTeX tables for the technical report, generated from the saved experiment tables.

Run:  python report/tables.py

Writes report/generated/key_results.tex. Numbers are never typed by hand.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "generated"
METHODS = (("C1", "C1"), ("C2", "C2"), ("C3", "C3"), ("C3-E avg", "C3-E"))


def read(*path: str) -> list[dict]:
    with open(ROOT.joinpath("results", *path), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


HANDOVER = (("policy", "Policy"), ("adaptive MPC", "Adapt."), ("handover-physics", "Phys."),
            ("handover-learned", "Learn."), ("handover-consistency", "Cons."))


def two_link_handover() -> None:
    """M10: policy alone, adaptive MPC alone, and the three single-monitor handovers, 32 runs a cell."""
    rows = read("m10", "summary.csv")
    scenarios = (("inside", "W1", "0.0"), ("inside", "W1", "1.0"), ("inside", "W2", "1.0"), ("inside", "W3", "0.1"),
                 ("inside", "W3", "1.0"), ("outside", "W1", "0.0"), ("outside", "W3", "0.0"))
    lines = [r"\begin{tabular}{@{}lll" + "c" * len(HANDOVER) + "@{}}", r"\toprule",
             "Tasks & W & kg & " + " & ".join(label for _, label in HANDOVER) + r"\\", r"\midrule"]
    for split, world, payload in scenarios:
        cells = []
        for name, _ in HANDOVER:
            r = next(r for r in rows if (r["split"], r["world"], r["payload_kg"], r["controller"])
                     == (split, world, payload, name))
            cells.append(f"{float(r['median_final_error_cm']):.2f} ({int(float(r['runs_not_settled']))})")
        lines.append(f"{split[:3]}. & {world} & {float(payload):g} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("two_link_handover.tex", lines)


def panda_stiffness() -> None:
    """M11c (stiff) and M12a (soft): C0, C1 and the reference inside the region, median of 16 runs."""
    stiff, soft = read("m11", "summary.csv"), read("m12a", "summary.csv")
    lines = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{stiff} & \multicolumn{3}{c}{soft}\\",
             r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
             r"W & kg & C0 & C1 & Ref. & C0 & C1 & Ref.\\", r"\midrule"]
    for world in ("W1", "W2", "W3"):
        for payload in ("1.0", "2.5"):
            cells = []
            for rows in (stiff, soft):
                for method in ("C0", "C1", "reference"):
                    r = next(r for r in rows if (r["split"], r["world"], r["payload_kg"], r["method"])
                             == ("inside", world, payload, method))
                    cells.append(f"{float(r['median_final_error_cm']):.2f}")
            lines.append(f"{world} & {float(payload):g} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("panda_stiffness.tex", lines)


def panda_handover() -> None:
    """M12c: final error pooled over worlds, per stiffness (48 runs a cell), and W3 alone at 1 kg (16 runs)."""
    rows = read("m12c", "handover_runs.csv")
    scenarios = (("inside", "2.5", None), ("inside", "1.0", None), ("inside", "1.0", "W3"), ("outside", "0.0", None))
    lines = [r"\begin{tabular}{@{}llll" + "c" * len(HANDOVER) + "@{}}", r"\toprule",
             "Tasks & kg & W & Stiff. & " + " & ".join(label for _, label in HANDOVER) + r"\\", r"\midrule"]
    for split, payload, world in scenarios:
        for gains in ("stiff", "soft"):
            cells = []
            for name, _ in HANDOVER:
                g = [r for r in rows if (r["split"], r["payload_kg"], r["gains"], r["controller"])
                     == (split, payload, gains, name) and (world is None or r["world"] == world)]
                median = sorted(float(r["final_error_cm"]) for r in g)
                m = (median[len(g) // 2 - 1] + median[len(g) // 2]) / 2
                cells.append(f"{m:.2f} ({sum(1 - int(r['settled']) for r in g)})")
            lines.append(f"{split[:3]}. & {float(payload):g} & {world or 'all'} & {gains} & " + " & ".join(cells)
                         + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("panda_handover.tex", lines)


def main() -> None:
    two_link_handover()
    panda_stiffness()
    panda_handover()
    with open(ROOT / "results" / "m6" / "ensemble_summary.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(ROOT / "results" / "m6" / "summary.csv", encoding="utf-8") as f:
        reference = {(r["region"], r["world"]): r for r in csv.DictReader(f) if r["method"] == "reference"}

    def entry(region, world, method, budget):
        r = next(r for r in rows if (r["region"], r["world"], r["method"], r["budget_s"]) == (region, world, method, budget))
        return f"{float(r['median_final_error_cm']):.2f} ({int(float(r['runs_not_settled']))})"

    lines = [
        r"\begin{tabular}{llr" + "c" * len(METHODS) + "c}",
        r"\toprule",
        "Tasks & World & Data & " + " & ".join(label for _, label in METHODS) + r" & Ref.\\",
        r"\midrule",
    ]
    for region in ("inside", "outside"):
        for world in ("W2", "W3"):
            for budget in ("30.0", "90.0"):
                ref = reference[(region, world)]
                cells = [entry(region, world, method, budget) for method, _ in METHODS]
                lines.append(f"{region} & {world} & {float(budget):g}\\,s & " + " & ".join(cells)
                             + f" & {float(ref['median_final_error_cm']):.2f} ({int(float(ref['runs_not_settled']))})\\\\")
        if region == "inside":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "key_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
