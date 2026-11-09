"""Offline tests for the final-report additions (split, test sampler, lock, sensitivity, best model)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dispute_rag import final_analysis as fa  # noqa: E402
from dispute_rag import splits  # noqa: E402
from dispute_rag.refs import cites_gold_mode  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ clause matching rules
@pytest.mark.parametrize("cited,gold,expected", [
    (["1005.11(a)(1)"], ["1005.11(a)"], (True, True, True)),    # more specific: all rules
    (["1005.11"], ["1005.11(a)"], (False, True, True)),         # parent: strict misses it
    (["1005.11(b)"], ["1005.11(a)"], (False, False, True)),     # sibling paragraph: section only
    (["§ 1005.6(b)"], ["1005.11(a)"], (False, False, False)),   # different section
    ([], ["1005.11(a)"], (False, False, False)),
])
def test_clause_match_modes(cited, gold, expected):
    assert tuple(cites_gold_mode(cited, gold, m) for m in ("strict", "parent", "section")) == expected


def test_strict_mode_equals_primary_rule():
    from dispute_rag.refs import cites_gold
    for cited, gold in [(["1005.11"], ["1005.11(a)"]), (["1005.6(b)(1)"], ["1005.6(b)"])]:
        assert cites_gold_mode(cited, gold, "strict") == cites_gold(cited, gold)


# ------------------------------------------------------------------ test-set sampler
def test_allocation_respects_floor_capacity_and_total():
    m = _load_script("01c_draw_test_sample.py")
    w = {"q1": 0.14, "q2": 1.0, "q3": 1.0, "q4": 0.46}
    cap = {"q1": 3, "q2": 500, "q3": 500, "q4": 500}
    a = m.allocate(w, 60, 4, cap)
    assert sum(a.values()) == 60
    assert a["q1"] == 3                      # capped below the floor
    assert a["q2"] >= a["q4"] >= 4


def test_draw_is_stratified_and_reproducible():
    m = _load_script("01c_draw_test_sample.py")
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-09-18", "2026-08-11", freq="D")
    pool = pd.DataFrame({"complaint_id": [str(i) for i in range(4000)],
                         "date_received": rng.choice(dates, 4000).astype(str),
                         "dispute_type": rng.choice(["unauthorized_transaction", "incorrect_amount",
                                                     "error_resolution"], 4000)})
    s1, design = m.draw(pool, 205, 2026, "2024-09-18", "2026-08-11", 4)
    s2, _ = m.draw(pool, 205, 2026, "2024-09-18", "2026-08-11", 4)
    assert len(s1) == 205 and s1["complaint_id"].is_unique
    assert s1["complaint_id"].tolist() == s2["complaint_id"].tolist()
    assert set(s1["quarter"]) == set(design["quarter_days"])          # every quarter represented
    full = [q for q, d in design["quarter_days"].items() if d >= 90]
    counts = s1["quarter"].value_counts()
    assert counts[full].max() - counts[full].min() <= 1               # equal across full quarters


# ------------------------------------------------------------------ protocol lock
def test_lock_detects_change(tmp_path, monkeypatch):
    f = tmp_path / "a.txt"
    f.write_text("v1")
    monkeypatch.setattr(splits, "LOCKED_FILES", [str(f)])
    lock = tmp_path / "lock.json"
    assert splits.verify_lock(lock)                                    # no lock yet
    lock.write_text(json.dumps({"sha256": splits.current_hashes()}))
    assert splits.verify_lock(lock) == []
    f.write_text("v2")
    assert splits.verify_lock(lock) == [str(f)]


def test_split_paths():
    assert splits.paths("dev").runs == Path("outputs/runs")           # interim paths unchanged
    assert splits.paths("test").runs == Path("outputs/runs_test")
    with pytest.raises(ValueError):
        splits.paths("train")


# ------------------------------------------------------------------ analysis helpers
def _toy_long():
    gold = ["approve_dispute", "deny_dispute", "request_more_info", "request_more_info"] * 10
    rows = []
    preds = {"C0": ["escalate", "deny_dispute", "request_more_info", "escalate"] * 10,
             "A": ["approve_dispute", "deny_dispute", "escalate", "request_more_info"] * 10}
    for c, pr in preds.items():
        for i, (g, p) in enumerate(zip(gold, pr)):
            rows.append({"case_id": str(i), "condition": c, "config": c, "gold_determination": g,
                         "pred_determination": p, "determination_correct": int(g == p),
                         "fabricated_citation": 0, "correct_clause_cited": 1, "clause_in_context": 1,
                         "consistent": 1, "review_required": 1, "latency_s": 10.0 + i % 3})
    return pd.DataFrame(rows)


def test_needs_human_collapse_counts_handover_as_correct():
    long = _toy_long()
    res = fa.needs_human_collapse(long)
    # C0: deny and both request_more_info cases now correct (escalate ~ more info) -> 3 of 4
    assert res["per_condition"]["C0"]["rate"] == pytest.approx(0.75)
    assert res["per_condition"]["A"]["rate"] == pytest.approx(1.0)


def test_best_model_rule_and_summary():
    long = _toy_long()
    cls = fa.classification_by_condition(long)
    summ = fa.condition_summary(long, cls)
    best = fa.best_model(summ, long)
    assert best["best"] == "A" and best["runner_up"] == "C0"
    assert summ.loc["A", "accuracy"] == pytest.approx(0.75)
    assert "statement" in best and best["mcnemar_vs_runner_up"]["discordant"] > 0
    assert fa.majority_baseline(long)["label"] == "request_more_info"


def test_dev_test_table_differences():
    s = pd.DataFrame({"n": [100, 100], "accuracy": [.3, .5], "macro_f1": [.2, .4], "fabricated_rate": [0, 0],
                      "correct_clause_rate": [.7, .5], "consistent_rate": [1, 1], "latency_p95_s": [10, 12]},
                     index=pd.Index(["C0", "A"], name="condition"))
    t = s.copy()
    t["accuracy"] = [.35, .45]
    tab = fa.dev_test_table(s, t)
    assert tab.loc["A", "accuracy_diff"] == pytest.approx(-0.05)


def test_prompt_versions_differ_only_in_rule_5():
    from dispute_rag.pipeline import PROMPTS
    v6, v7 = PROMPTS["v6"].splitlines(), PROMPTS["v7"].splitlines()
    diff = [(a, b) for a, b in zip(v6, v7) if a != b]
    assert len(v6) == len(v7) and len(diff) == 1 and diff[0][0].startswith("5.")


def test_best_by_metric_handles_direction_and_ties():
    s = pd.DataFrame({"accuracy": [.3, .3, .2], "macro_f1": [.2, .1, .3], "fabricated_rate": [.05, 0, 0],
                      "correct_clause_rate": [.7, .4, .4], "consistent_rate": [1, .9, .9],
                      "latency_p95_s": [10, 12, 9]}, index=["C0", "A", "C"])
    b = fa.best_by_metric(s).set_index("column")
    assert b.loc["accuracy", "best"] == "C0 = A"
    assert b.loc["fabricated_rate", "best"] == "A = C"           # lower is better
    assert b.loc["latency_p95_s", "best"] == "C"


# ------------------------------------------------------------------ raw-data handling
def test_require_raw_csv_accepts_real_file_and_rejects_lfs_stub(tmp_path):
    from dispute_rag.cfpb_cases import require_raw_csv
    real = tmp_path / "complaints.csv"
    real.write_text("Complaint ID,Product\n1,Prepaid card\n")
    assert require_raw_csv(real) == real
    stub = tmp_path / "complaints.csv.gz"
    stub.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(SystemExit) as e:
        require_raw_csv(stub)
    assert "Git LFS placeholder" in str(e.value)
    with pytest.raises(SystemExit) as e:
        require_raw_csv(tmp_path / "missing.csv")
    assert "data/raw/cfpb/README.md" in str(e.value)
