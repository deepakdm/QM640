# Pre-registration for the final-report test set

Fill in every `[DECIDE]` item, commit, then run `python scripts/02b_lock_protocol.py`. This file is
one of the locked files, so it cannot be edited after the lock without the test runner refusing to
run. Everything below is decided **before** any model output on the test set exists.

Date: [DECIDE: YYYY-MM-DD]   Author: Deepak Dharani

## 1. Roles of the two sets

| Set | Source | n labelled | Role |
|---|---|---|---|
| Development | Interim pilot, seed 42 | 125 | Method decisions were made while viewing these results. Reported as development results. |
| Test | `scripts/01c_draw_test_sample.py`, seed 2026 | [DECIDE: actual n after labelling] | Confirmatory. No output viewed before the lock. |

Test sample design: quarter × dispute type, quarter weights proportional to days in the window
(floor of 4 per quarter), pilot IDs and narratives identical to a pilot narrative excluded,
`--n-draw 205` (target about 160 in scope at the pilot's 78% in-scope rate). Power for RQ1 at
n = 160, δ = .15, ψ = .30, α = .05/3: about .87 (Connor, 1987).

## 2. What is held fixed from the interim

Generator (Claude Sonnet 4.6, temperature 0, 800 tokens, 3 runs), corpus (Regulation E and
Supplement I), the four conditions and their settings, the deployment candidate (C), the
hypotheses H1–H4, α = .05 with Holm correction over three comparisons, the adoption benchmarks,
and the failure taxonomy and its order.

## 3. Changes made on the development set (and why)

| Item | Decision | Reason from the development set |
|---|---|---|
| Prompt | [DECIDE: keep `v6` / switch to `v7`] | Pilot: `escalate` predicted for 66% of complaints vs 1.6% of gold labels. If v7, set `generation.prompt_version: v7` in all four configs (identical across conditions). If v6, state that the over-escalation finding is carried to the test set unchanged. |
| Primary correct-clause rule | Strict (unchanged) | Kept as primary so the interim result is replicated, not redefined. |
| Clause-rule sensitivity | Parent and section rules reported as sensitivity | Pilot: strict rule penalises citing the parent section; gap to C0 shrinks from ~33 to 8–18 points under looser rules. |
| Secondary accuracy | `escalate` + `request_more_info` merged into `needs_human` | Both send the case to an investigator; tests whether the system decides when it should. |
| Secondary subset | Complaints not flagged ambiguous | Separates label ambiguity from model error. |
| Best model | Ranking rule in `final_analysis.BEST_MODEL_RULE` | Rubric requires a stated best model; rule fixed before test results. |

## 4. Primary and secondary outcomes (test set)

Primary: determination accuracy (RQ1); fabricated-citation and strict correct-clause rates (RQ2);
GEE odds ratio for clause in context (RQ3); dominant failure share and adoption benchmarks (RQ4).
Secondary: macro-F1, needs-human accuracy, unambiguous-subset accuracy, clause-rule sensitivity,
consistency, latency, cost model.

## 5. Reporting rule

Development and test results are reported side by side (`outputs/final/dev_vs_test.csv`).
Hypothesis decisions in the final report use the test set only. Any deviation from this document is
reported as a deviation, with its reason.
