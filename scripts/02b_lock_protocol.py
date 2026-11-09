"""Step 2b (final report): lock the analysis protocol before the TEST set is run.

Records a SHA-256 hash of every file that decides the test-set numbers (configs, prompt, retrieval,
scoring and analysis code, corpus, split manifest, frozen test gold labels, pre-registration note).
scripts/03_run_experiment.py --split test checks these hashes and refuses to run if anything has
changed, which is what lets the report state that the test results were not tuned.

    python scripts/02b_lock_protocol.py           # lock (fails if a required file is missing)
    python scripts/02b_lock_protocol.py --check   # verify only; exit 1 if anything changed
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dispute_rag.splits import PROTOCOL_LOCK, current_hashes, verify_lock  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        bad = verify_lock()
        print("Protocol intact." if not bad else "Changed since lock:\n  " + "\n  ".join(bad))
        sys.exit(1 if bad else 0)
    hashes = current_hashes()
    missing = [f for f, h in hashes.items() if h is None]
    if missing:
        sys.exit("Cannot lock; missing:\n  " + "\n  ".join(missing))
    gold = Path("data/gold/test/gold_cases.jsonl")
    n_gold = sum(1 for line in gold.read_text().splitlines() if line.strip())
    if n_gold < 30:
        sys.exit(f"Cannot lock: only {n_gold} test gold labels in {gold}. Label the test sheet and run "
                 "scripts/02_import_gold.py --split test first.")
    if PROTOCOL_LOCK.exists():
        sys.exit(f"{PROTOCOL_LOCK} already exists. A protocol is locked once.")
    PROTOCOL_LOCK.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_LOCK.write_text(json.dumps(
        {"locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sha256": hashes}, indent=2))
    print(f"Locked {len(hashes)} files -> {PROTOCOL_LOCK}. Commit and tag the repository now "
          "(e.g. git tag test-protocol-lock), then run the test set.")


if __name__ == "__main__":
    main()
