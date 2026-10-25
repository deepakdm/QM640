"""Step 3: run one condition over the gold set (runs_per_case times), save answers + metrics.

Usage:
    python scripts/03_run_experiment.py configs/baseline.yaml
    python scripts/03_run_experiment.py configs/baseline.yaml --limit 10     # quick validation run
    python scripts/03_run_experiment.py configs/baseline.yaml --score-only   # re-score saved answers

Answers are appended to outputs/runs/<config>/answers.jsonl as each call returns, so an
interrupted run resumes where it stopped. A call that fails after the SDK's retries is logged to
errors.jsonl and left for the next run; it is never scored as a model error.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from dispute_rag.cfpb_cases import load_cases  # noqa: E402
from dispute_rag.classification import classification_report_for_run  # noqa: E402
from dispute_rag.config import ExperimentConfig  # noqa: E402
from dispute_rag.corpus import corpus_section_index, load_documents  # noqa: E402
from dispute_rag.evaluate import aggregate, summarize_run  # noqa: E402
from dispute_rag.failure_analysis import failure_records, failure_table  # noqa: E402
from dispute_rag.gold import load_gold  # noqa: E402
from dispute_rag.schemas import RAGAnswer  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("--limit", type=int, default=None, help="only the first N gold cases")
p.add_argument("--case-ids", default=None, help="file with one case_id per line")
p.add_argument("--score-only", action="store_true", help="do not call the model")
p.add_argument("--no-judge", action="store_true")
p.add_argument("--out-root", default="outputs/runs")
args = p.parse_args()

cfg = ExperimentConfig.load(args.config)
gold = load_gold(cfg.gold_path)
cases = [c for c in load_cases(Path("data/cases/cases.jsonl")) if c.case_id in gold]
if args.case_ids:
    wanted = {l.strip() for l in open(args.case_ids) if l.strip()}
    cases = [c for c in cases if c.case_id in wanted]
if args.limit:
    cases = cases[: args.limit]

out_dir = Path(args.out_root) / cfg.name
out_dir.mkdir(parents=True, exist_ok=True)
ans_path, err_path = out_dir / "answers.jsonl", out_dir / "errors.jsonl"

done: dict[tuple[str, int], RAGAnswer] = {}
if ans_path.exists():
    for line in open(ans_path):
        if line.strip():
            a = RAGAnswer.model_validate_json(line)
            done[(a.case_id, a.run_idx)] = a

docs = load_documents(cfg.corpus_dir)
corpus_index = corpus_section_index(docs)
(out_dir / "corpus_index.txt").write_text("\n".join(sorted(corpus_index)))

pipe = None
if not args.score_only:
    from tqdm import tqdm
    from dispute_rag.pipeline import RAGPipeline
    pipe = RAGPipeline(cfg)
    todo = [(c, r) for c in cases for r in range(cfg.evaluation.runs_per_case)
            if (c.case_id, r) not in done]
    print(f"{cfg.name}: {len(done)} answers on file, {len(todo)} to run")
    with open(ans_path, "a") as fa, open(err_path, "a") as fe:
        for case, r in tqdm(todo, desc=cfg.name):
            try:
                a = pipe.answer(case, run_idx=r)
            except Exception as e:  # API failure after retries: leave for the next run
                fe.write(json.dumps({"case_id": case.case_id, "run_idx": r, "error": repr(e)}) + "\n")
                fe.flush()
                continue
            done[(a.case_id, a.run_idx)] = a
            fa.write(a.model_dump_json() + "\n")
            fa.flush()

ids = {c.case_id for c in cases}
answers = [a for (cid, _), a in sorted(done.items()) if cid in ids]
if not answers:
    sys.exit("No answers to score.")

# Context text for the groundedness judge (retrieval conditions only; the control saw none)
context_lookup = {}
if cfg.retrieval.mode != "none":
    from dispute_rag.corpus import build_chunks
    from dispute_rag.pipeline import format_context
    chunks = {c.chunk_id: c for c in (pipe.chunks if pipe else build_chunks(cfg.corpus_dir, cfg.chunking))}
    for a in answers:
        if a.run_idx == 0:
            context_lookup[a.case_id] = format_context([chunks[i] for i in a.retrieved_chunk_ids if i in chunks])

judge = None
if not args.no_judge and not args.score_only and cfg.retrieval.mode != "none" and cfg.evaluation.judge_sample_size:
    from dispute_rag.llm import LLM
    judge = LLM(model=os.getenv("JUDGE_MODEL"), temperature=0.0, max_tokens=1200)

df = summarize_run(answers, gold, corpus_index, context_lookup, judge, cfg.evaluation.judge_sample_size)
df.to_csv(out_dir / "per_case_metrics.csv", index=False)
agg = aggregate(df)
agg.to_json(out_dir / "summary.json", indent=2)
failure_records(answers, gold, corpus_index).to_csv(out_dir / "failure_records.csv", index=False)
failure_table(answers, gold, corpus_index).to_csv(out_dir / "failure_table.csv")
print(agg.to_string())

cls = classification_report_for_run(answers, gold)
cls["confusion_matrix"].to_csv(out_dir / "confusion_matrix.csv")
cls["per_class"].to_csv(out_dir / "per_class_metrics_by_class.csv")
with open(out_dir / "kfold_stability.json", "w") as f:
    json.dump(cls["kfold_stability"], f, indent=2)
print("\nConfusion matrix (rows = gold, cols = predicted)")
print(cls["confusion_matrix"].to_string())
print("\nPer-class metrics")
print(cls["per_class"].round(3).to_string())
