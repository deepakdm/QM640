"""Indicative cost-per-case / cycle-time model with sensitivity analysis (overview doc §2A.8).

ALL inputs are assumptions to be sourced (industry benchmarks, published reports) and stated in the
report. This is analytical, not empirical.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    manual_minutes_per_case: float = 25.0      # avg investigator handling time, manual
    assisted_minutes_per_case: float = 12.0    # handling time when assistant output is accepted after review
    review_overhead_minutes: float = 4.0       # time to read/verify assistant output
    rework_minutes_if_wrong: float = 20.0      # extra time when assistant is wrong and investigator must redo
    loaded_hourly_cost: float = 45.0           # fully loaded investigator cost, USD/hr
    monthly_case_volume: int = 20_000
    detection_rate_of_errors: float = 0.85     # share of assistant errors the investigator catches on review
    review_share: float = 1.0                  # share of drafts routed to full review; set to the
                                               # observed review burden from RQ4 (1.0 = every draft)


def cost_per_case(acc: float, a: Assumptions) -> dict:
    err = 1 - acc
    caught, missed = err * a.detection_rate_of_errors, err * (1 - a.detection_rate_of_errors)
    assisted = (a.assisted_minutes_per_case + a.review_share * a.review_overhead_minutes
                + caught * a.rework_minutes_if_wrong)
    manual = a.manual_minutes_per_case
    return {"accuracy": acc, "manual_min": manual, "assisted_min": assisted,
            "minutes_saved_pct": (manual - assisted) / manual,
            "manual_cost": manual / 60 * a.loaded_hourly_cost,
            "assisted_cost": assisted / 60 * a.loaded_hourly_cost,
            "monthly_saving_usd": (manual - assisted) / 60 * a.loaded_hourly_cost * a.monthly_case_volume,
            "undetected_error_share": missed}


def sensitivity(a: Assumptions | None = None, accuracies=np.arange(0.5, 1.0001, 0.05)) -> pd.DataFrame:
    a = a or Assumptions()
    return pd.DataFrame([cost_per_case(float(x), a) for x in accuracies])


def breakeven_accuracy(a: Assumptions | None = None) -> float:
    """Accuracy at which assisted and manual handling take equal time (closed form).

    assisted(acc) = base + (1 - acc) * d * rework, so assisted = manual when
    acc = 1 - (manual - base) / (d * rework), clipped to [0, 1].
    """
    a = a or Assumptions()
    base = a.assisted_minutes_per_case + a.review_share * a.review_overhead_minutes
    slope = a.detection_rate_of_errors * a.rework_minutes_if_wrong
    if slope <= 0:
        return 0.0 if base < a.manual_minutes_per_case else 1.0
    return float(min(1.0, max(0.0, 1 - (a.manual_minutes_per_case - base) / slope)))


def tornado(a: Assumptions | None = None, accuracy: float = 0.70, swing: float = 0.25) -> pd.DataFrame:
    """One-way sensitivity of break-even accuracy and minutes saved to each input (+/- swing)."""
    a = a or Assumptions()
    rows = []
    for field in ("manual_minutes_per_case", "assisted_minutes_per_case", "review_overhead_minutes",
                  "rework_minutes_if_wrong", "detection_rate_of_errors", "review_share"):
        base_val = getattr(a, field)
        for label, mult in (("low", 1 - swing), ("high", 1 + swing)):
            v = base_val * mult
            if field in ("detection_rate_of_errors", "review_share"):
                v = min(1.0, v)
            b = Assumptions(**{**a.__dict__, field: v})
            rows.append({"input": field, "level": label, "value": v,
                         "breakeven_accuracy": breakeven_accuracy(b),
                         "minutes_saved_pct_at_acc": cost_per_case(accuracy, b)["minutes_saved_pct"]})
    return pd.DataFrame(rows)
