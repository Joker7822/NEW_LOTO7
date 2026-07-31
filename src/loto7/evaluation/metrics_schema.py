"""Canonical financial metric schema shared by all hardened evaluators."""
from __future__ import annotations

from typing import Dict, Mapping

METRIC_SCHEMA_VERSION = "loto7-metrics-2026.07.31-v2"


def _number(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


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
    cost = int(_number(result.get("total_cost")))
    payout = int(_number(result.get("total_payout")))
    result.update(financial_metrics(cost, payout))
    return result


def get_payout_roi_percent(payload: Mapping[str, object]) -> float:
    value = payload.get("payout_roi_percent")
    if value is not None:
        return _number(value)
    ratio = payload.get("payout_ratio", payload.get("payout_roi"))
    if ratio is not None:
        return _number(ratio) * 100.0
    cost = _number(payload.get("total_cost"))
    payout = _number(payload.get("total_payout"))
    return payout / cost * 100.0 if cost else 0.0


def get_profit_roi_percent(payload: Mapping[str, object]) -> float:
    value = payload.get("profit_roi_percent")
    if value is not None:
        return _number(value)
    ratio = payload.get("profit_ratio")
    if ratio is not None:
        return _number(ratio) * 100.0
    cost = _number(payload.get("total_cost"))
    profit = _number(payload.get("profit"))
    return profit / cost * 100.0 if cost else 0.0
