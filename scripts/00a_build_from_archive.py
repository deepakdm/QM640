"""Step 0a (optional): build data/raw/cfpb/complaints.csv from the CFPB narratives archive files.

The study's raw file is data/raw/cfpb/complaints.csv. If you already have it, you do not need this
script. Nothing is downloaded unless you pass --download:

    python scripts/00a_build_from_archive.py               # uses archive ZIPs already in data/raw/cfpb/archive/
    python scripts/00a_build_from_archive.py --download    # fetches missing ZIPs from files.consumerfinance.gov

It keeps complaints received 2024-09-18 to 2026-08-14 in the three Regulation E products, with a
non-empty narrative, de-duplicated by Complaint ID, and refuses to overwrite an existing
complaints.csv unless --force is given.
"""
import argparse
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/cfpb")
ARCH = RAW / "archive"
OUT = RAW / "complaints.csv"
BASE = "https://files.consumerfinance.gov/f/documents/"
FILES = [
    "CCDB_Export_7_August_2024_through_October_2024.zip",
    "CCDB_Export_8_November_2024_through_December_2024.zip",
    "CCDB_Export_9_January_2025_through_February_2025.zip",
    "CCDB_Export_10_March_2025_through_April_2025.zip",
    "CCDB_Export_11_May_2025_through_June_2025.zip",
    "CCDB_Export_12_July_2025_through_August_2025.zip",
    "CCDB_Export_13_September_2025_through_October_2025.zip",
    "CCDB_Export_14_November_2025_through_December_2025.zip",
    "CCDB_Export_15_January_2026_through_February_2026.zip",
    "CCDB_Export_16_March_2026.zip",
    "CCDB_Export_17_April_2026.zip",
    "CCDB_Export_18_May_2026.zip",
    "CCDB_Export_19_June_2026.zip",
    "CCDB_Export_20_July_2026.zip",
    "CCDB_Export_21_August_2026.zip",
]
START, END = "2024-09-18", "2026-08-14"
PRODUCTS = {
    "Checking or savings account",
    "Prepaid card",
    "Money transfer, virtual currency, or money service",
}

ap = argparse.ArgumentParser()
ap.add_argument("--download", action="store_true", help="download archive ZIPs that are missing")
ap.add_argument("--force", action="store_true", help="overwrite an existing complaints.csv")
args = ap.parse_args()

if OUT.exists() and not args.force:
    sys.exit(f"{OUT} already exists; nothing to do. Use --force to rebuild it from the archive ZIPs.")
ARCH.mkdir(parents=True, exist_ok=True)
missing = [n for n in FILES if not ((ARCH / n).exists() and (ARCH / n).stat().st_size > 0)]
if missing and not args.download:
    sys.exit(f"{len(missing)} of {len(FILES)} archive ZIPs are not in {ARCH} (first: {missing[0]}).\n"
             f"Either place data/raw/cfpb/complaints.csv yourself, copy the ZIPs into {ARCH}, or rerun "
             "with --download to fetch them from the CFPB.")

# 1. download (skips files already present)
for name in FILES:
    dest = ARCH / name
    if dest.exists() and dest.stat().st_size > 0:
        print("have", name)
        continue
    import requests
    print("downloading", name)
    with requests.get(BASE + name, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for part in r.iter_content(1 << 20):
                f.write(part)

# 2. read, filter
kept, scanned = [], 0
for name in FILES:
    with zipfile.ZipFile(ARCH / name) as z:
        for member in [m for m in z.namelist() if m.lower().endswith(".csv")]:
            with z.open(member) as fh:
                for chunk in pd.read_csv(fh, dtype=str, chunksize=100_000):
                    scanned += len(chunk)
                    ncols = [c for c in chunk.columns if "narrative" in c.lower()]
                    if not ncols:
                        raise SystemExit(f"{name}/{member}: no narrative column. "
                                         f"Columns: {list(chunk.columns)}")
                    chunk = chunk.rename(columns={ncols[0]: "Consumer complaint narrative"})
                    d = pd.to_datetime(chunk["Date received"], errors="coerce")
                    m = (d.between(START, END)
                         & chunk["Product"].isin(PRODUCTS)
                         & chunk["Consumer complaint narrative"].fillna("").str.strip().ne(""))
                    kept.append(chunk[m])

df = pd.concat(kept, ignore_index=True).drop_duplicates(subset="Complaint ID")
df.to_csv(OUT, index=False)

print(f"\nRows scanned: {scanned:,}   Rows kept: {len(df):,}  -> {OUT}")
print(df["Product"].value_counts().to_string())
print("Date range:", df["Date received"].min(), "to", df["Date received"].max())
print("Columns:", list(df.columns))
print("Download date:", date.today().isoformat(), "(record this for the report)")
