"""Render a preview image of the sampled complaints (the "screenshot" of the data for the report).

    python scripts/08_sample_preview.py                                   # dev sample, 8 rows
    python scripts/08_sample_preview.py --csv data/cases/test/sample_test.csv --out outputs/final/fig_sample_preview_test.png

Shows real rows from the sample file: ID, date, product, issue, dispute type, word count and the
first words of the narrative (redactions shown as the CFPB published them, XXXX). Rows are chosen
with a fixed seed, at least one per dispute type.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dispute_rag.viz import TEXT, TEXT_2, apply_style  # noqa: E402

PRODUCT = {"Checking or savings account": "Checking/savings", "Prepaid card": "Prepaid card",
           "Money transfer, virtual currency, or money service": "Money transfer"}
TYPE = {"unauthorized_transaction": "Unauthorized", "incorrect_amount": "Incorrect amount",
        "error_resolution": "Error resolution"}


def pick(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    per = max(1, n // df["dispute_type"].nunique())
    rows = df.groupby("dispute_type").sample(n=per, random_state=seed, replace=False)
    if len(rows) < n:
        rows = pd.concat([rows, df.drop(rows.index).sample(n - len(rows), random_state=seed)])
    return rows.head(n).sort_values("date_received")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/cases/sample_160.csv")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--words", type=int, default=28, help="narrative words shown per row")
    ap.add_argument("--out", default="outputs/final/fig_sample_preview.png")
    args = ap.parse_args()
    apply_style()

    df = pd.read_csv(args.csv, dtype={"complaint_id": str})
    if "narrative_words" not in df:
        df["narrative_words"] = df["narrative"].str.split().str.len()
    rows = pick(df, args.rows, args.seed)

    cols = ["Complaint ID", "Received", "Product", "Issue", "Dispute type", "Words", "Narrative (first words)"]
    widths = [0.085, 0.075, 0.105, 0.165, 0.105, 0.045, 0.42]
    wraps = [10, 10, 16, 26, 16, 6, 68]
    body = []
    for _, r in rows.iterrows():
        excerpt = " ".join(str(r["narrative"]).split()[: args.words]) + " …"
        body.append([r["complaint_id"], str(r["date_received"])[:10], PRODUCT.get(r["product"], r["product"]),
                     r["issue"], TYPE.get(r["dispute_type"], r["dispute_type"]), str(int(r["narrative_words"])),
                     excerpt])
    cells = [["\n".join(textwrap.wrap(str(v), w)) or "" for v, w in zip(row, wraps)] for row in body]
    lines = [max(c.count("\n") + 1 for c in row) for row in cells]

    fig_h = 0.55 + 0.17 * sum(lines) + 0.12 * len(cells)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.subplots_adjust(left=0.01, right=0.99)
    ax.axis("off")
    tab = ax.table(cellText=cells, colLabels=cols, colWidths=widths, loc="upper center", cellLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.2)
    unit = 1.0 / (sum(lines) + len(cells) + 1.6)
    for (i, _j), cell in tab.get_celld().items():
        cell.PAD = 0.02
        cell.set_edgecolor("#d9d9d4")
        cell.set_linewidth(0.6)
        if i == 0:
            cell.set_height(unit * 1.6)
            cell.set_facecolor("#eef0ee")
            cell.set_text_props(fontweight="bold", color=TEXT)
        else:
            cell.set_height(unit * (lines[i - 1] + 1))
            cell.set_text_props(color=TEXT, va="center")
    src = Path(args.csv).name
    ax.set_title(f"Sample of {len(rows)} complaints from {src} (n = {len(df)}); CFPB redactions shown as XXXX",
                 loc="left", fontsize=10, fontweight="bold", color=TEXT)
    fig.text(0.01, 0.005, "Source: CFPB Consumer Complaint Database narratives archive. Excerpt truncated; "
             "full text in the sample file.", fontsize=7, color=TEXT_2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
