"""Step 0e: (re)build the SQLite analysis database from the flat files.

    python scripts/00e_build_db.py                     # data/db/qm640.sqlite
    python scripts/00e_build_db.py --db other.sqlite

Safe to run at any stage: it loads whatever exists (dev and/or test cases, labels, runs) and
skips the rest. The database is derived; rebuilding it never changes a source file. See
src/dispute_rag/db.py for the schema and docs/data_access.md for example queries.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from dispute_rag.analysis import CONDITION_LABELS  # noqa: E402
from dispute_rag.cfpb_cases import load_cases  # noqa: E402
from dispute_rag.config import ChunkingCfg  # noqa: E402
from dispute_rag.corpus import build_chunks  # noqa: E402
from dispute_rag.db import DB_PATH, SCHEMA, VIEWS, connect  # noqa: E402
from dispute_rag.gold import load_gold  # noqa: E402
from dispute_rag.splits import paths  # noqa: E402

CHUNKING = {"fixed_512": ChunkingCfg(strategy="fixed", chunk_size=512, overlap=64),
            "fixed_256": ChunkingCfg(strategy="fixed", chunk_size=256, overlap=32),
            "section_512": ChunkingCfg(strategy="section", chunk_size=512, overlap=64)}


def quarter(d: str | None) -> str | None:
    ts = pd.to_datetime(d, errors="coerce")
    return None if pd.isna(ts) else f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--corpus", default="data/corpus")
    args = ap.parse_args()
    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    con = connect(db)
    con.executescript(SCHEMA)
    counts, sources = {}, []

    comp, gold_rows, ans_rows, pcm, fail = [], [], [], [], []
    for split in ("dev", "test"):
        sp = paths(split)
        if not sp.cases.exists():
            continue
        sources.append(str(sp.cases))
        for c in load_cases(sp.cases):
            k = c.complaint
            comp.append({"case_id": c.case_id, "split": split, "date_received": k.date_received,
                         "quarter": quarter(k.date_received), "product": k.product,
                         "sub_product": k.sub_product, "issue": k.issue, "sub_issue": k.sub_issue,
                         "company": k.company, "state": k.state, "dispute_type": c.dispute_type,
                         "narrative": c.narrative, "narrative_words": len(c.narrative.split()),
                         "redaction_tokens": c.narrative.upper().count("XXXX"),
                         "company_response": k.company_response, "timely_response": k.timely_response})
        if sp.gold.exists():
            sources.append(str(sp.gold))
            for g in load_gold(sp.gold).values():
                gold_rows.append({"case_id": g.case_id, "split": split, "determination": g.determination,
                                  "governing_clauses": "|".join(g.governing_clauses),
                                  "ambiguous": int(bool(g.ambiguous)), "rationale": g.rationale,
                                  "labeled_by": g.labeled_by,
                                  "second_rater_determination": g.second_rater_determination,
                                  "second_rater_clauses": "|".join(g.second_rater_clauses or [])})
        for run_dir in sorted(p for p in sp.runs.glob("*") if p.is_dir()) if sp.runs.exists() else []:
            cond = CONDITION_LABELS.get(run_dir.name, run_dir.name)
            a_path = run_dir / "answers.jsonl"
            if a_path.exists():
                sources.append(str(a_path))
                for line in a_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    a = json.loads(line)
                    ans_rows.append({"split": split, "condition": cond, "config": run_dir.name,
                                     "case_id": a["case_id"], "run_idx": a["run_idx"],
                                     "determination": a.get("determination"),
                                     "cited_section_refs": "|".join(a.get("cited_section_refs") or []),
                                     "retrieved_section_refs": "|".join(a.get("retrieved_section_refs") or []),
                                     "retrieved_chunk_ids": "|".join(a.get("retrieved_chunk_ids") or []),
                                     "reasoning": a.get("reasoning"), "latency_s": a.get("latency_s"),
                                     "model": a.get("model"), "parse_ok": int(bool(a.get("parse_ok", True))),
                                     "raw_response": a.get("raw_response")})
            for fname, sink in (("per_case_metrics.csv", pcm), ("failure_records.csv", fail)):
                f = run_dir / fname
                if f.exists():
                    sink.append(pd.read_csv(f, dtype={"case_id": str}).assign(split=split, condition=cond))

    ids = pd.Series([r["case_id"] for r in comp])
    if ids.duplicated().any():
        both = sorted(set(ids[ids.duplicated()]))
        sys.exit(f"Integrity check failed: {len(both)} complaint(s) appear in both dev and test "
                 f"(e.g. {both[:3]}). The split must be disjoint; redraw the test sample.")
    known = set(ids)
    # Only rows for sampled complaints; anything else would break the foreign keys.
    pd.DataFrame(comp).to_sql("complaints", con, if_exists="append", index=False)
    pd.DataFrame([g for g in gold_rows if g["case_id"] in known]).to_sql("gold_labels", con, if_exists="append", index=False)
    pd.DataFrame([a for a in ans_rows if a["case_id"] in known]).to_sql("answers", con, if_exists="append", index=False)
    if pcm:
        pd.concat(pcm, ignore_index=True).to_sql("per_case_metrics", con, index=False)
    if fail:
        pd.concat(fail, ignore_index=True).to_sql("failure_records", con, index=False)

    chunks = []
    for name, cfg in CHUNKING.items():
        try:
            for ch in build_chunks(Path(args.corpus), cfg):
                chunks.append({"strategy": name, "chunk_id": ch.chunk_id, "doc_id": ch.doc_id,
                               "section_ref": ch.section_ref,
                               "section_refs": "|".join(ch.section_refs or []),
                               "n_words": len(ch.text.split()), "text": ch.text})
        except Exception as e:  # corpus missing or unreadable: the rest of the database is still useful
            print(f"corpus_chunks ({name}) skipped: {e!r}")
    if chunks:
        pd.DataFrame(chunks).to_sql("corpus_chunks", con, if_exists="append", index=False)

    if pcm:
        con.executescript(VIEWS)
    for t in ("complaints", "gold_labels", "corpus_chunks", "answers", "per_case_metrics", "failure_records"):
        try:
            counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            counts[t] = 0
    info = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": json.dumps(sorted(set(sources))), "row_counts": json.dumps(counts)}
    con.executemany("INSERT INTO build_info VALUES (?, ?)", list(info.items()))
    con.commit()
    con.close()
    print(f"Built {db}")
    for t, n in counts.items():
        print(f"  {t:<17} {n:>7,}")


if __name__ == "__main__":
    main()
