#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_ml_stack.py

LOTO7 ML拡張スタック。

実装内容:
  - MemoryBank: 過去drawから4/5/6個一致構造を記憶
  - MetaClassifier: ticket特徴量から4等以上確率を推定
  - LightGBM / XGBoost / CatBoost: 勾配ブースティング系モデル
  - Optuna: 特徴量重み・閾値の探索
  - SHAP: 特徴量寄与度レポート

方針:
  - walk-forward用の部品として使えるよう、未来リークを避ける関数構成
  - 依存ライブラリは requirements-ml.txt で導入
  - 失敗時でも原因をJSON/CSVに残す

注意:
  ランダム抽せんのため的中・収益は保証しない。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

NUMBERS = tuple(range(1, 38))

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    np = None
    pd = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    XGBClassifier = None
    XGB_IMPORT_ERROR = exc
else:
    XGB_IMPORT_ERROR = None

try:
    from lightgbm import LGBMClassifier
except Exception as exc:  # pragma: no cover
    LGBMClassifier = None
    LGBM_IMPORT_ERROR = exc
else:
    LGBM_IMPORT_ERROR = None

try:
    from catboost import CatBoostClassifier
except Exception as exc:  # pragma: no cover
    CatBoostClassifier = None
    CAT_IMPORT_ERROR = exc
else:
    CAT_IMPORT_ERROR = None

try:
    import optuna
except Exception as exc:  # pragma: no cover
    optuna = None
    OPTUNA_IMPORT_ERROR = exc
else:
    OPTUNA_IMPORT_ERROR = None

try:
    import shap
except Exception as exc:  # pragma: no cover
    shap = None
    SHAP_IMPORT_ERROR = exc
else:
    SHAP_IMPORT_ERROR = None


@dataclass(frozen=True)
class Draw:
    draw_no: int
    date: str
    main: Tuple[int, ...]
    bonus: Tuple[int, ...]


@dataclass(frozen=True)
class TicketEval:
    draw_no: int
    target_date: str
    numbers: Tuple[int, ...]
    main_match: int
    bonus_match: int
    prize_rank: str
    label_4plus: int
    label_5plus: int


def parse_nums(text: object) -> Tuple[int, ...]:
    return tuple(int(x) for x in str(text or "").replace(",", " ").split() if x.isdigit())


def draw_no_int(text: object) -> Optional[int]:
    import re

    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def load_draws(csv_path: str) -> List[Draw]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(csv_path)
    rows: List[Draw] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            draw_no = draw_no_int(row.get("回別"))
            main = parse_nums(row.get("本数字"))
            bonus = parse_nums(row.get("ボーナス数字"))
            date = str(row.get("抽せん日") or "").strip()
            if draw_no is None or len(main) != 7 or len(set(main)) != 7:
                continue
            if len(bonus) != 2:
                bonus = tuple()
            rows.append(Draw(draw_no, date, tuple(sorted(main)), tuple(sorted(bonus))))
    return sorted(rows, key=lambda x: x.draw_no)


def prize_rank(main_match: int, bonus_match: int) -> str:
    if main_match == 7:
        return "1等"
    if main_match == 6 and bonus_match >= 1:
        return "2等"
    if main_match == 6:
        return "3等"
    if main_match == 5:
        return "4等"
    if main_match == 4:
        return "5等"
    if main_match == 3 and bonus_match >= 1:
        return "6等"
    return "外れ"


def evaluate_ticket(ticket: Sequence[int], target: Draw) -> TicketEval:
    s = set(ticket)
    main_match = len(s & set(target.main))
    bonus_match = len(s & set(target.bonus)) if target.bonus else 0
    rank = prize_rank(main_match, bonus_match)
    return TicketEval(
        draw_no=target.draw_no,
        target_date=target.date,
        numbers=tuple(sorted(ticket)),
        main_match=main_match,
        bonus_match=bonus_match,
        prize_rank=rank,
        label_4plus=1 if main_match >= 5 else 0,
        label_5plus=1 if main_match >= 4 else 0,
    )


def build_memorybank(draws: Sequence[Draw], output_dir: str) -> Dict[str, Dict[str, float]]:
    """過去の当せん構造を記憶する。評価時はtrain範囲だけを渡すこと。"""
    banks: Dict[str, Dict[str, float]] = {"mb4": {}, "mb5": {}, "mb6": {}}
    total = len(draws)
    for idx, draw in enumerate(draws):
        age = total - idx - 1
        weight = 0.992 ** age
        for k, name in [(4, "mb4"), (5, "mb5"), (6, "mb6")]:
            for combo in itertools.combinations(draw.main, k):
                key = "-".join(f"{n:02d}" for n in combo)
                banks[name][key] = banks[name].get(key, 0.0) + weight

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, bank in banks.items():
        path = out / f"loto7_memorybank_{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["combo", "score"])
            writer.writeheader()
            for combo, score in sorted(bank.items(), key=lambda x: x[1], reverse=True):
                writer.writerow({"combo": combo, "score": round(score, 8)})
    return banks


def memorybank_score(ticket: Sequence[int], banks: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    t = tuple(sorted(ticket))
    scores = {"mb4_score": 0.0, "mb5_score": 0.0, "mb6_score": 0.0}
    for k, name, out_name in [(4, "mb4", "mb4_score"), (5, "mb5", "mb5_score"), (6, "mb6", "mb6_score")]:
        for combo in itertools.combinations(t, k):
            key = "-".join(f"{n:02d}" for n in combo)
            scores[out_name] += banks.get(name, {}).get(key, 0.0)
    return scores


def number_frequency(draws: Sequence[Draw], window: Optional[int] = None) -> Dict[int, float]:
    subset = list(draws[-window:]) if window else list(draws)
    total = max(1, len(subset))
    freq = {n: 0.0 for n in NUMBERS}
    for idx, draw in enumerate(subset):
        age = total - idx - 1
        w = 0.985 ** age
        for n in draw.main:
            freq[n] += w
    return freq


def pair_frequency(draws: Sequence[Draw], window: int = 240) -> Dict[Tuple[int, int], float]:
    subset = list(draws[-window:])
    total = max(1, len(subset))
    freq: Dict[Tuple[int, int], float] = {}
    for idx, draw in enumerate(subset):
        age = total - idx - 1
        w = 0.990 ** age
        for a, b in itertools.combinations(draw.main, 2):
            key = (min(a, b), max(a, b))
            freq[key] = freq.get(key, 0.0) + w
    return freq


def ticket_features(ticket: Sequence[int], train_draws: Sequence[Draw], banks: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    t = tuple(sorted(ticket))
    full = number_frequency(train_draws)
    r240 = number_frequency(train_draws, 240)
    r120 = number_frequency(train_draws, 120)
    r60 = number_frequency(train_draws, 60)
    pf = pair_frequency(train_draws, 240)

    odd = sum(1 for n in t if n % 2)
    low = sum(1 for n in t if n <= 18)
    total_sum = sum(t)
    consecutive = sum(1 for a, b in zip(t, t[1:]) if b == a + 1)
    pair_score = sum(pf.get((min(a, b), max(a, b)), 0.0) for a, b in itertools.combinations(t, 2))
    mb = memorybank_score(t, banks)

    f = {
        "sum": float(total_sum),
        "odd": float(odd),
        "low": float(low),
        "consecutive": float(consecutive),
        "freq_full": sum(full[n] for n in t),
        "freq_240": sum(r240[n] for n in t),
        "freq_120": sum(r120[n] for n in t),
        "freq_60": sum(r60[n] for n in t),
        "pair_score": pair_score,
        "range": float(max(t) - min(t)),
        "zone_1_9": float(sum(1 for n in t if 1 <= n <= 9)),
        "zone_10_18": float(sum(1 for n in t if 10 <= n <= 18)),
        "zone_19_27": float(sum(1 for n in t if 19 <= n <= 27)),
        "zone_28_37": float(sum(1 for n in t if 28 <= n <= 37)),
    }
    f.update(mb)
    return f


def generate_candidate_tickets(train_draws: Sequence[Draw], count: int = 80, pool_size: int = 18, seed: int = 777) -> List[Tuple[int, ...]]:
    rng = random.Random(seed + len(train_draws))
    freq = number_frequency(train_draws, 240)
    pool = [n for n, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:pool_size]]
    pool = sorted(pool)

    candidates = set()
    for combo in itertools.combinations(pool, 7):
        total = sum(combo)
        odd = sum(n % 2 for n in combo)
        low = sum(1 for n in combo if n <= 18)
        if 85 <= total <= 190 and odd in (2, 3, 4, 5) and low in (2, 3, 4, 5):
            candidates.add(tuple(combo))
        if len(candidates) >= count * 3:
            break

    while len(candidates) < count:
        candidates.add(tuple(sorted(rng.sample(NUMBERS, 7))))
    scored = []
    pf = pair_frequency(train_draws, 240)
    for c in candidates:
        score = sum(freq[n] for n in c) + sum(pf.get((min(a,b), max(a,b)), 0.0) for a,b in itertools.combinations(c,2)) * 0.08
        scored.append((score, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:count]]


def build_training_frame(draws: Sequence[Draw], output_dir: str, min_train: int, max_targets: int, candidates_per_draw: int) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError(f"pandas/sklearn import failed: {SKLEARN_IMPORT_ERROR}")
    rows = []
    target_indices = list(range(min_train, len(draws)))
    if max_targets > 0:
        target_indices = target_indices[-max_targets:]

    for idx in target_indices:
        train = draws[:idx]
        target = draws[idx]
        banks = build_memorybank(train, output_dir)
        tickets = generate_candidate_tickets(train, count=candidates_per_draw, seed=idx)
        # 正例が少なすぎるため、実当せん数字そのものと近傍も教師用に混ぜる。ただしtargetを特徴量計算には使わない。
        tickets.append(target.main)
        for miss_one in itertools.combinations(target.main, 6):
            for add in NUMBERS:
                if add not in miss_one:
                    tickets.append(tuple(sorted((*miss_one, add))))
                    break
            if len(tickets) >= candidates_per_draw + 8:
                break
        seen = set()
        for ticket in tickets:
            if ticket in seen:
                continue
            seen.add(ticket)
            ev = evaluate_ticket(ticket, target)
            feats = ticket_features(ticket, train, banks)
            row = {"draw_no": target.draw_no, "date": target.date, "numbers": " ".join(f"{n:02d}" for n in ticket), **feats, "label_4plus": ev.label_4plus, "label_5plus": ev.label_5plus, "main_match": ev.main_match, "bonus_match": ev.bonus_match, "rank": ev.prize_rank}
            rows.append(row)
    return pd.DataFrame(rows)


def make_models(random_state: int = 777) -> Dict[str, object]:
    models: Dict[str, object] = {}
    models["meta_logistic"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state))
    models["meta_rf"] = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=random_state, n_jobs=-1)
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.035, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=random_state, n_jobs=-1)
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, class_weight="balanced", random_state=random_state, n_jobs=-1, verbose=-1)
    if CatBoostClassifier is not None:
        models["catboost"] = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.03, loss_function="Logloss", verbose=False, random_seed=random_state, auto_class_weights="Balanced")
    return models


def train_models(df: "pd.DataFrame", output_dir: str, label: str = "label_4plus") -> Dict[str, object]:
    feature_cols = [c for c in df.columns if c not in {"draw_no", "date", "numbers", "label_4plus", "label_5plus", "main_match", "bonus_match", "rank"}]
    X = df[feature_cols].fillna(0.0)
    y = df[label].astype(int)
    if y.nunique() < 2:
        raise RuntimeError(f"label has only one class: {label}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    models = make_models()
    reports = []
    fitted = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                pred = model.predict(X_test)
                proba = pred.astype(float)
            pred = (proba >= 0.5).astype(int)
            auc = float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else 0.0
            reports.append({"model": name, "auc": auc, "test_rows": len(y_test), "positive_rate": float(y_test.mean())})
            fitted[name] = model
        except Exception as exc:
            reports.append({"model": name, "error": str(exc)})

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports).to_csv(out / "ml_model_report.csv", index=False)
    pd.DataFrame({"feature": feature_cols}).to_csv(out / "ml_features.csv", index=False)
    return {"models": fitted, "feature_cols": feature_cols, "reports": reports}


def run_optuna(df: "pd.DataFrame", output_dir: str, trials: int = 50, label: str = "label_4plus") -> Dict[str, float]:
    if optuna is None:
        return {"error": f"optuna import failed: {OPTUNA_IMPORT_ERROR}"}
    feature_cols = [c for c in df.columns if c not in {"draw_no", "date", "numbers", "label_4plus", "label_5plus", "main_match", "bonus_match", "rank"}]
    X = df[feature_cols].fillna(0.0)
    y = df[label].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    def objective(trial):
        model_name = trial.suggest_categorical("model", ["rf", "logistic"] + (["xgb"] if XGBClassifier is not None else []) + (["lgbm"] if LGBMClassifier is not None else []))
        if model_name == "rf":
            model = RandomForestClassifier(n_estimators=trial.suggest_int("n_estimators", 80, 350), max_depth=trial.suggest_int("max_depth", 3, 12), class_weight="balanced", random_state=777, n_jobs=-1)
        elif model_name == "xgb":
            model = XGBClassifier(n_estimators=trial.suggest_int("n_estimators", 80, 350), max_depth=trial.suggest_int("max_depth", 2, 8), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12), eval_metric="logloss", random_state=777, n_jobs=-1)
        elif model_name == "lgbm":
            model = LGBMClassifier(n_estimators=trial.suggest_int("n_estimators", 80, 400), num_leaves=trial.suggest_int("num_leaves", 15, 63), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12), class_weight="balanced", random_state=777, n_jobs=-1, verbose=-1)
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=trial.suggest_float("C", 0.05, 5.0), max_iter=1000, class_weight="balanced"))
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test, proba)) if len(set(y_test)) > 1 else 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    result = {"best_value": float(study.best_value), **study.best_params}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "optuna_best_params.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_shap(df: "pd.DataFrame", train_result: Dict[str, object], output_dir: str, label: str = "label_4plus") -> Dict[str, object]:
    if shap is None:
        return {"error": f"shap import failed: {SHAP_IMPORT_ERROR}"}
    feature_cols = train_result["feature_cols"]
    X = df[feature_cols].fillna(0.0)
    models = train_result["models"]
    # tree系を優先
    model = None
    model_name = None
    for name in ["lightgbm", "xgboost", "catboost", "meta_rf", "meta_logistic"]:
        if name in models:
            model = models[name]
            model_name = name
            break
    if model is None:
        return {"error": "no fitted model"}
    sample = X.tail(min(500, len(X)))
    try:
        explainer = shap.Explainer(model, sample)
        values = explainer(sample)
        arr = values.values
        if len(arr.shape) == 3:
            arr = arr[:, :, -1]
        imp = np.abs(arr).mean(axis=0)
        rows = [{"model": model_name, "feature": f, "mean_abs_shap": float(v)} for f, v in sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)]
        pd.DataFrame(rows).to_csv(Path(output_dir) / "shap_feature_importance.csv", index=False)
        return {"model": model_name, "rows": len(rows)}
    except Exception as exc:
        return {"error": str(exc), "model": model_name}


def save_json(path: str, data: Dict[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LOTO7 ML stack: MemoryBank + MetaClassifier + LGBM/Cat/XGB + Optuna + SHAP")
    parser.add_argument("--csv", default="loto7.csv")
    parser.add_argument("--output-dir", default="outputs/ml_stack")
    parser.add_argument("--min-train", type=int, default=60)
    parser.add_argument("--max-targets", type=int, default=240)
    parser.add_argument("--candidates-per-draw", type=int, default=80)
    parser.add_argument("--label", default="label_4plus", choices=["label_4plus", "label_5plus"])
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "imports": {
            "sklearn": None if SKLEARN_IMPORT_ERROR is None else str(SKLEARN_IMPORT_ERROR),
            "xgboost": None if XGB_IMPORT_ERROR is None else str(XGB_IMPORT_ERROR),
            "lightgbm": None if LGBM_IMPORT_ERROR is None else str(LGBM_IMPORT_ERROR),
            "catboost": None if CAT_IMPORT_ERROR is None else str(CAT_IMPORT_ERROR),
            "optuna": None if OPTUNA_IMPORT_ERROR is None else str(OPTUNA_IMPORT_ERROR),
            "shap": None if SHAP_IMPORT_ERROR is None else str(SHAP_IMPORT_ERROR),
        },
    }
    try:
        draws = load_draws(args.csv)
        build_memorybank(draws, args.output_dir)
        df = build_training_frame(draws, args.output_dir, args.min_train, args.max_targets, args.candidates_per_draw)
        df.to_csv(out / "ml_training_frame.csv", index=False)
        train_result = train_models(df, args.output_dir, args.label)
        status["reports"] = train_result["reports"]
        if not args.skip_optuna:
            status["optuna"] = run_optuna(df, args.output_dir, args.optuna_trials, args.label)
        if not args.skip_shap:
            status["shap"] = run_shap(df, train_result, args.output_dir, args.label)
        status["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        status["ok"] = True
    except Exception as exc:
        status["ok"] = False
        status["error"] = str(exc)
        save_json(str(out / "ml_stack_status.json"), status)
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    save_json(str(out / "ml_stack_status.json"), status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
