"""Draw the end-to-end pipeline figure for the final report (six labelled stages + storage layer).

    python scripts/06_pipeline_diagram.py            # -> outputs/final/fig_pipeline.png

The six stages are the ones the final-report rubric names: Data, Preprocessing, EDA, Feature
Engineering, Model, Deployment. Box colour says who performs the step (legend), so the figure also
shows where the human checkpoints sit. Text is kept short; the report text explains each stage.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

from dispute_rag.viz import TEXT, TEXT_2, apply_style  # noqa: E402

FILL = {"data": "#eaf2fb", "auto": "#f2f2f0", "llm": "#e6f6ef", "human": "#fdf0e8"}
EDGE = {"data": "#2a78d6", "auto": "#52514e", "llm": "#1baf7a", "human": "#eb6834"}

STAGES = [
    ("1  Data", "data", [
        "CFPB narratives archive", "1,610,649 complaints", "Sep 2024 - Aug 2026",
        "Reg E + Supp. I (eCFR)"]),
    ("2  Preprocessing", "auto", [
        "Reg E products, 3 dispute", "types, 40-600 words,", "de-duplication, XXXX kept",
        "Dev / test split (seeded)"]),
    ("3  EDA", "auto", [
        "Funnel, length, redaction,", "volume by quarter,", "label mix vs company",
        "response, associations"]),
    ("4  Feature Engineering", "auto", [
        "Query from narrative +", "issue (company response", "withheld); chunks 512/256/",
        "section; dense + BM25"]),
    ("5  Model", "llm", [
        "C0 none | A dense 512 |", "B dense 256 | C hybrid", "+ re-rank; Claude Sonnet",
        "4.6, T = 0, JSON, 3 runs"]),
    ("6  Deployment", "human", [
        "Draft with citations ->", "review routing ->", "investigator decides;",
        "audit log of evidence"]),
]
SUB = {  # human / evaluation steps hanging under a stage
    1: ("Gold labels + 2nd rater", "human"),
    4: ("Evaluation: accuracy,\nmacro-F1, grounding,\nMcNemar (Holm), GEE", "auto"),
}


def main(out: Path = Path("outputs/final/fig_pipeline.png")) -> None:
    apply_style()
    w, h, gap, x0, y0 = 18.0, 21.0, 2.4, 1.0, 19.0
    total = 6 * w + 5 * gap
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, total + 2)
    ax.set_ylim(-6, 46)
    ax.axis("off")
    centers = []
    for i, (title, kind, lines) in enumerate(STAGES):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=FILL[kind], ec=EDGE[kind], lw=1.6))
        ax.text(x + w / 2, y0 + h - 2.6, title, ha="center", va="center", fontsize=8.4,
                fontweight="bold", color=TEXT)
        ax.text(x + w / 2, y0 + h / 2 - 2.2, "\n".join(lines), ha="center", va="center",
                fontsize=7.6, color=TEXT, linespacing=1.5)
        centers.append(x + w / 2)
        if i:
            ax.add_patch(FancyArrowPatch((x - gap + 0.2, y0 + h / 2), (x - 0.25, y0 + h / 2),
                                         arrowstyle="-|>", mutation_scale=12, lw=1.4, color=TEXT_2))
    for i, (label, kind) in SUB.items():
        bx = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((bx, 7.2), w, 7.6, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=FILL[kind], ec=EDGE[kind], lw=1.3))
        ax.text(bx + w / 2, 11.0, label, ha="center", va="center", fontsize=7.4, color=TEXT)
        ax.add_patch(FancyArrowPatch((centers[i], y0 - 0.3), (centers[i], 15.2),
                                     arrowstyle="-|>", mutation_scale=10, lw=1.2, color=TEXT_2))
    # storage layer under everything
    ax.add_patch(FancyBboxPatch((x0, 0.6), total, 4.0, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc="white", ec=TEXT_2, lw=1.0, ls="--"))
    ax.text(x0 + total / 2, 2.6,
            "Storage: flat files in Git (CSV / JSONL)  ->  SQLite analysis database data/db/qm640.sqlite "
            "(complaints, gold_labels, corpus_chunks, answers, per_case_metrics)",
            ha="center", va="center", fontsize=7.4, color=TEXT_2)
    ax.text(x0, 44.2, "End-to-end study pipeline: six stages", fontsize=11, fontweight="bold", color=TEXT)
    handles = [Patch(fc=FILL[k], ec=EDGE[k], label=l) for k, l in
               (("data", "Public data source"), ("auto", "Automated step (Python)"),
                ("llm", "Language model (API)"), ("human", "Human decision or review"))]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8,
              frameon=False, handlelength=1.4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
