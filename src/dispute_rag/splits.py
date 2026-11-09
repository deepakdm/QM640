"""Development / test split for the final report.

Nothing in this study is trained, so the split is not a train/test split in the machine-learning
sense. It separates the evidence used to *make decisions about the method* from the evidence used
to *report it*:

  dev   The 125-complaint pilot labelled for the interim report. Everything learned from it
        (prompt wording, clause-matching rule, benchmarks, the best-model rule) was decided while
        looking at these results, so dev results are reported as development results only.
  test  A new, quarter-stratified sample drawn by scripts/01c_draw_test_sample.py, excluding every
        pilot complaint and every narrative identical to a pilot narrative. It is labelled and
        frozen, and the protocol is locked (scripts/02b_lock_protocol.py), before any model output
        on it is produced. Confirmatory claims in the final report come from test only.

The dev paths are the original pilot paths, so the interim outputs remain valid without moving.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("dev", "test")


@dataclass(frozen=True)
class SplitPaths:
    name: str
    cases: Path        # cases.jsonl
    gold: Path         # gold_cases.jsonl
    labeling: Path     # labeling sheet (input to 02_import_gold.py)
    runs: Path         # outputs/runs[...]/<config>/
    reports: Path      # outputs/reports[...]/
    final: Path        # outputs/final/<split>/


def paths(split: str = "dev", root: Path | str = ".") -> SplitPaths:
    root = Path(root)
    if split == "dev":
        return SplitPaths("dev", root / "data/cases/cases.jsonl", root / "data/gold/gold_cases.jsonl",
                          root / "data/gold/labeling_sheet.csv", root / "outputs/runs",
                          root / "outputs/reports", root / "outputs/final/dev")
    if split == "test":
        return SplitPaths("test", root / "data/cases/test/cases.jsonl", root / "data/gold/test/gold_cases.jsonl",
                          root / "data/gold/test/labeling_sheet.csv", root / "outputs/runs_test",
                          root / "outputs/reports_test", root / "outputs/final/test")
    raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")


MANIFEST = Path("data/splits/split_manifest.json")
PROTOCOL_LOCK = Path("data/splits/protocol_lock.json")

# Files whose content decides what the test-set numbers will be. Their hashes are recorded by
# scripts/02b_lock_protocol.py before the test runs, and 03_run_experiment.py --split test refuses
# to run if any of them has changed since.
LOCKED_FILES = [
    "configs/control_no_retrieval.yaml", "configs/baseline.yaml", "configs/small_chunks.yaml",
    "configs/hybrid_rerank.yaml",
    "src/dispute_rag/pipeline.py", "src/dispute_rag/retriever.py", "src/dispute_rag/corpus.py",
    "src/dispute_rag/refs.py", "src/dispute_rag/evaluate.py", "src/dispute_rag/failure_analysis.py",
    "src/dispute_rag/classification.py", "src/dispute_rag/analysis.py",
    "src/dispute_rag/final_analysis.py",
    "data/corpus/reg_e.md", "data/corpus/reg_e_interp.md",
    "data/splits/split_manifest.json", "data/gold/test/gold_cases.jsonl",
    "docs/preregistration_final.md",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_hashes() -> dict[str, str | None]:
    return {f: (sha256(Path(f)) if Path(f).exists() else None) for f in LOCKED_FILES}


def verify_lock(lock_path: Path = PROTOCOL_LOCK) -> list[str]:
    """Files whose hash differs from the protocol lock (empty list = protocol intact)."""
    if not Path(lock_path).exists():
        return ["<no protocol lock: run scripts/02b_lock_protocol.py>"]
    locked = json.loads(Path(lock_path).read_text())["sha256"]
    now = current_hashes()
    return [f for f in LOCKED_FILES if locked.get(f) != now.get(f)]
