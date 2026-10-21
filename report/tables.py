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


def main() -> None:
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
