"""Canonical financial metric schema shared by all hardened evaluators."""
from __future__ import annotations

from typing import Dict, Mapping

METRIC_SCHEMA_VERSION = "loto7-metrics-2026.07.31-v2"


def financial_metrics(total_cost: int, total_payout: int) -> Dict[str, object]:
    cost = int(total_cost)
    payout = int(total_payout)
    profit = payout - cost
    payout_ratio = payout / cost if cost else 0.0
    profit_ratio = profit / cost if cost else 0.0
    return {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "total_cost": cost,
        "total_payout": payout,
        "profit": profit,
        "payout_ratio": round(payout_ratio, 6),
        "payout_roi_percent": round(payout_ratio * 100.0, 3),
        "profit_ratio": round(profit_ratio, 6),
        "profit_roi_percent": round(profit_ratio * 100.0, 3),
        # Compatibility aliases. Under schema v2 ROI always means profit ROI.
        "roi": round(profit_ratio, 6),
        "roi_percent": round(profit_ratio * 100.0, 3),
        "payout_roi": round(payout_ratio, 6),
        "deprecated_metric_aliases": ["roi", "roi_percent", "payout_roi"],
    }


def normalize_financial_metrics(payload: Mapping[str, object]) -> Dict[str, object]:
    result = dict(payload)
    cost = int(float(result.get("total_cost", 0) or 0))
    payout = int(float(result.get("total_payout", 0) or 0))
    result.update(financial_metrics(cost, payout))
    return result


def get_payout_roi_percent(payload: Mapping[str, object]) -> float:
    value = payload.get("payout_roi_percent")
    if value is not None:
        return float(value)
    ratio = payload.get("payout_ratio", payload.get("payout_roi"))
    if ratio is not None:
        return float(ratio) * 100.0
    cost = float(payload.get("total_cost", 0) or 0)
    payout = float(payload.get("total_payout", 0) or 0)
    return payout / cost * 100.0 if cost else 0.0


def get_profit_roi_percent(payload: Mapping[str, object]) -> float:
    value = payload.get("profit_roi_percent")
    if value is not None:
        return float(value)
    ratio = payload.get("profit_ratio")
    if ratio is not None:
        return float(ratio) * 100.0
    cost = float(payload.get("total_cost", 0) or 0)
    profit = float(payload.get("profit", 0) or 0)
    return profit / cost * 100.0 if cost else 0.0
