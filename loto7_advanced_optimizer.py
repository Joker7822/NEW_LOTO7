#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Advanced Loto7 optimizer for NEW_LOTO7.

Features:
- grade-focused scoring for 5+ and 6+ main-number matches
- walk-forward validation without future data
- Optuna optimization with random-search fallback
- Monte Carlo candidate search
- MemoryBank from past high-match predictions
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loto7_logic_predictor import DEFAULT_PRIZE_TABLE, DEFAULT_UNIT_COST, Draw, PrizeResult, TicketScore, classify_loto7_prize, format_ticket, score_normalized_values
from loto7_enhanced_predictor import DEFAULT_HIT_PATTERN_CSV, band_counts, build_number_scores, counter_norm, make_candidate_pool, max_consecutive_run, save_latest_txt, structure_penalty, weighted_combination_counts

NUM_MIN = 1
NUM_MAX = 37
PICK_SIZE = 7
WEIGHTS_CACHE = "loto7_advanced_weights.json"
MEMORYBANK_CSV = "loto7_memorybank.csv"

@dataclass(frozen=True)
class AdvancedWeights:
    single: float = 1.00
    pair: float = 1.15
    triple: float = 1.25
    memory: float = 1.45
    grade6: float = 1.75
    structure: float = 1.00
    diversity: float = 0.25
    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()
    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "AdvancedWeights":
        base = cls().to_dict()
        for k in base:
            try:
                base[k] = float(d.get(k, base[k]))
            except Exception:
                pass
        return cls(**base)

def _parse_ticket(v: object) -> Tuple[int, ...]:
    nums = tuple(sorted(int(x) for x in re.findall(r"\d+", str(v or ""))))
    nums = tuple(n for n in nums if NUM_MIN <= n <= NUM_MAX)
    return nums if len(nums) == PICK_SIZE and len(set(nums)) == PICK_SIZE else tuple()

def _grade_label(g: Optional[int]) -> str:
    return "ハズレ" if g is None else f"{g}等"

def _avg(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _load_weights(path: str = WEIGHTS_CACHE) -> Optional[AdvancedWeights]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return AdvancedWeights.from_dict(data.get("weights", data))
    except Exception:
        return None

def _save_weights(w: AdvancedWeights, path: str = WEIGHTS_CACHE) -> None:
    Path(path).write_text(json.dumps({"weights": w.to_dict()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

class MemoryBank:
    def __init__(self) -> None:
        self.items: List[Tuple[Tuple[int, ...], float]] = []
        self.pairs: Counter = Counter()
        self.triples: Counter = Counter()
        self.sums: Counter = Counter()
    def add(self, ticket: Sequence[int], strength: float = 1.0) -> None:
        t = tuple(sorted(ticket))
        if len(t) != PICK_SIZE or len(set(t)) != PICK_SIZE:
            return
        self.items.append((t, float(strength)))
        for p in itertools.combinations(t, 2): self.pairs[p] += strength
        for tri in itertools.combinations(t, 3): self.triples[tri] += strength
        self.sums[sum(t)//10*10] += strength
    def load_detail(self, path: str, before_date: Optional[str] = None, min_matches: int = 4) -> None:
        p = Path(path)
        if not p.exists(): return
        try:
            rows = list(csv.DictReader(p.open("r", encoding="utf-8-sig", newline="")))
        except Exception:
            return
        for row in rows:
            date = row.get("抽せん日", "")
            if before_date and date >= before_date: continue
            for i in range(1, 101):
                pk = f"予測{i}"; mk = f"予測{i}_本数字一致"
                if pk not in row: break
                try: m = int(row.get(mk, 0) or 0)
                except Exception: m = 0
                if m >= min_matches:
                    t = _parse_ticket(row.get(pk, ""))
                    if t: self.add(t, 1.0 + max(0, m-4)*0.8)
    def load_memorybank(self, path: str = MEMORYBANK_CSV, before_date: Optional[str] = None) -> None:
        p = Path(path)
        if not p.exists(): return
        try:
            for row in csv.DictReader(p.open("r", encoding="utf-8-sig", newline="")):
                if before_date and row.get("抽せん日", "") >= before_date: continue
                t = _parse_ticket(row.get("組合せ", ""))
                if t: self.add(t, float(row.get("強度", 1) or 1))
        except Exception:
            return
    def save(self, path: str = MEMORYBANK_CSV) -> None:
        with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["組合せ", "強度"]); w.writeheader()
            for t, s in self.items[-1000:]: w.writerow({"組合せ": format_ticket(t), "強度": round(s, 6)})
    def score(self, ticket: Sequence[int]) -> float:
        if not self.items: return 0.0
        s = set(ticket); vals = []
        for pat, strength in self.items[-300:]:
            o = len(s & set(pat))
            vals.append(({6:0.3, 5:1.0, 4:0.55, 3:0.15}.get(o, 0.0)) * strength)
        pair = sum(counter_norm(self.pairs, p) for p in itertools.combinations(sorted(ticket), 2)) / 21.0
        tri = sum(counter_norm(self.triples, t) for t in itertools.combinations(sorted(ticket), 3)) / 35.0
        return 0.60 * max(vals) + 0.20 * _avg(vals) + 0.12 * pair + 0.08 * tri

def build_memory(before_date: Optional[str], detail_csv: str) -> MemoryBank:
    b = MemoryBank(); b.load_detail(detail_csv, before_date); b.load_memorybank(MEMORYBANK_CSV, before_date); return b

def grade_bonus(ticket: Sequence[int], draws: Sequence[Draw]) -> float:
    t = tuple(sorted(ticket)); low, mid, high = band_counts(t)
    odd = sum(n % 2 for n in t); total = sum(t); run = max_consecutive_run(t)
    repeat = len(set(t) & set(draws[-1].main)); ldmax = max(Counter(n % 10 for n in t).values())
    score = 0.0
    if low in (2,3) and mid in (2,3) and high in (1,2,3): score += 0.30
    if odd in (3,4): score += 0.22
    if 125 <= total <= 165: score += 0.25
    elif 115 <= total <= 175: score += 0.10
    if 2 <= repeat <= 4: score += 0.20
    if run <= 2: score += 0.15
    elif run >= 4: score -= 0.55
    if ldmax <= 2: score += 0.12
    elif ldmax >= 4: score -= 0.40
    return score

def context(draws: Sequence[Draw], before_date: Optional[str], detail_csv: str) -> Dict[str, object]:
    return {
        "num": build_number_scores(draws),
        "p1": weighted_combination_counts(draws, 2, 9.0, 70),
        "p2": weighted_combination_counts(draws, 2, 30.0, 180),
        "p3": weighted_combination_counts(draws, 2, 95.0, 0),
        "t1": weighted_combination_counts(draws, 3, 32.0, 220),
        "t2": weighted_combination_counts(draws, 3, 110.0, 0),
        "mem": build_memory(before_date, detail_csv),
    }

def score_ticket(ticket: Sequence[int], draws: Sequence[Draw], ctx: Dict[str, object], w: AdvancedWeights, strategy: str) -> TicketScore:
    t = tuple(sorted(ticket)); nums: Dict[int, float] = ctx["num"]  # type: ignore
    single = sum(nums.get(n, 0.0) for n in t)
    pair = sum(0.40*counter_norm(ctx["p1"], p)+0.28*counter_norm(ctx["p2"], p)+0.10*counter_norm(ctx["p3"], p) for p in itertools.combinations(t, 2))  # type: ignore
    triple = sum(0.52*counter_norm(ctx["t1"], tri)+0.18*counter_norm(ctx["t2"], tri) for tri in itertools.combinations(t, 3))  # type: ignore
    mem: MemoryBank = ctx["mem"]  # type: ignore
    memory = mem.score(t); gb = grade_bonus(t, draws); penalty = structure_penalty(t, draws)
    low, mid, high = band_counts(t); div = 0.10 if len(set(n//10 for n in t)) >= 4 else 0.0
    bonus = 0.35 if strategy == "GRADE3" and memory > 0 else 0.08 if strategy == "MONTECARLO" else 0.0
    total = w.single*single + w.pair*pair + w.triple*triple + w.memory*memory + w.grade6*gb + w.diversity*div + bonus - w.structure*penalty
    return TicketScore(t, total, {"single":single,"pair":pair,"triple":triple,"memory":memory,"pattern":memory,"grade6":gb,"penalty":penalty,"sum":float(sum(t)),"odd":float(sum(n%2 for n in t)),"low":float(low),"mid":float(mid),"high":float(high),"repeat_last":float(len(set(t)&set(draws[-1].main)))}, strategy)

def sample_ticket(draws: Sequence[Draw], pool_size: int, rng: random.Random) -> Tuple[int, ...]:
    ns = build_number_scores(draws); pool = make_candidate_pool(draws, max(pool_size, 16)); alln = list(range(1,38)); hot = sorted(alln, key=lambda n: ns.get(n,0), reverse=True)[:max(20,pool_size)]
    s=set()
    while len(s)<7: s.add(rng.choice(pool if rng.random()<0.7 else hot if rng.random()<0.9 else alln))
    return tuple(sorted(s))

def optimize_weights(draws: Sequence[Draw], trials: int = 25) -> AdvancedWeights:
    cached = _load_weights()
    if cached: return cached
    rng = random.Random(len(draws)*99991); best = AdvancedWeights(); best_score = -1.0
    def eval_w(w: AdvancedWeights) -> float:
        start = max(20, len(draws)-30); val=[]
        for i in range(start, len(draws)):
            preds = advanced_predict(draws[:i], 5, 15, DEFAULT_HIT_PATTERN_CSV, draws[i].date, 250, w, False)
            bh = max((len(set(p.ticket)&set(draws[i].main)) for p in preds), default=0)
            val.append(45 if bh>=6 else 13 if bh==5 else 4 if bh==4 else 0.5*bh)
        return _avg(val)
    try:
        import optuna  # type: ignore
        def obj(trial):
            w = AdvancedWeights(*(trial.suggest_float(k, a, b) for k,a,b in [("single",0.65,1.35),("pair",0.7,1.65),("triple",0.75,1.85),("memory",0.7,2.2),("grade6",0.9,2.6),("structure",0.6,1.55),("diversity",0.05,0.55)]))
            return eval_w(w)
        st = optuna.create_study(direction="maximize"); st.optimize(obj, n_trials=int(os.getenv("LOTO7_OPTUNA_TRIALS", str(trials))), show_progress_bar=False)
        best = AdvancedWeights.from_dict(st.best_params)
    except Exception:
        for _ in range(int(os.getenv("LOTO7_OPTUNA_TRIALS", str(trials)))):
            w = AdvancedWeights(rng.uniform(.65,1.35),rng.uniform(.7,1.65),rng.uniform(.75,1.85),rng.uniform(.7,2.2),rng.uniform(.9,2.6),rng.uniform(.6,1.55),rng.uniform(.05,.55))
            sc = eval_w(w)
            if sc > best_score: best, best_score = w, sc
    _save_weights(best); return best

def advanced_predict(draws: Sequence[Draw], num_tickets: int = 10, pool_size: int = 24, hit_pattern_csv: str = DEFAULT_HIT_PATTERN_CSV, before_date: Optional[str] = None, monte_carlo_iterations: Optional[int] = None, weights: Optional[AdvancedWeights] = None, optimize: bool = True) -> List[TicketScore]:
    if weights is None: weights = optimize_weights(draws) if optimize and len(draws)>=120 and os.getenv("LOTO7_DISABLE_OPTIMIZE","0")!="1" else (_load_weights() or AdvancedWeights())
    mc = int(os.getenv("LOTO7_MONTE_CARLO", str(monte_carlo_iterations if monte_carlo_iterations is not None else 12000)))
    ctx = context(draws, before_date, hit_pattern_csv); ranked: Dict[Tuple[int,...], TicketScore] = {}
    pool = make_candidate_pool(draws, pool_size)
    for t in itertools.combinations(pool, 7):
        item = score_ticket(t, draws, ctx, weights, "GRADE3"); ranked[item.ticket]=item
    rng = random.Random(len(draws)*1009 + sum(ord(c) for c in draws[-1].date))
    for _ in range(max(0, mc)):
        item = score_ticket(sample_ticket(draws, pool_size, rng), draws, ctx, weights, "MONTECARLO")
        old = ranked.get(item.ticket)
        if old is None or item.score > old.score: ranked[item.ticket]=item
    selected=[]; use=Counter()
    for item in sorted(ranked.values(), key=lambda x:(-x.score,x.ticket)):
        if any(len(set(item.ticket)&set(s.ticket))>4 for s in selected): continue
        if any(use[n]>=3 for n in item.ticket): continue
        selected.append(item); use.update(item.ticket)
        if len(selected)>=num_tickets: return selected
    return sorted(ranked.values(), key=lambda x:(-x.score,x.ticket))[:num_tickets]

def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows: return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def advanced_backtest(draws: Sequence[Draw], min_train: int, num_tickets: int, pool_size: int, hit_pattern_csv: str, max_backtest_draws: int, summary_csv: str, detail_csv: str) -> Dict[str, object]:
    start = max(min_train, len(draws)-max_backtest_draws) if max_backtest_draws and max_backtest_draws>0 else min_train
    weights = optimize_weights(draws[:start]) if start>=120 else (_load_weights() or AdvancedWeights())
    rows=[]; top=[]; best=[]; gd=Counter(); bgd=Counter(); purchase=prize=0; bank=MemoryBank()
    for i in range(start, len(draws)):
        actual=draws[i]
        preds=advanced_predict(draws[:i], num_tickets, pool_size, hit_pattern_csv, actual.date, int(os.getenv("LOTO7_BACKTEST_MONTE_CARLO","2500")), weights, False)
        res=[classify_loto7_prize(p.ticket, actual.main, actual.bonus, DEFAULT_PRIZE_TABLE) for p in preds]
        hits=[r.main_matches for r in res]; top.append(hits[0] if hits else 0); best.append(max(hits) if hits else 0)
        dp=len(preds)*DEFAULT_UNIT_COST; dw=sum(r.prize for r in res); purchase+=dp; prize+=dw
        grades=[r.grade for r in res if r.grade is not None]
        for r in res: gd[_grade_label(r.grade)]+=1
        bg=_grade_label(min(grades) if grades else None); bgd[bg]+=1
        row={"抽せん日":actual.date,"回別":actual.draw_no or "","本数字":format_ticket(actual.main),"ボーナス数字":format_ticket(actual.bonus),"口数":len(preds),"購入金額":dp,"当せん金額":dw,"収支":dw-dp,"最高等級":bg,"最高本数字一致数":max(hits) if hits else 0,"当せん口数":sum(1 for r in res if r.grade is not None)}
        for idx,(p,r) in enumerate(zip(preds,res),1):
            row[f"予測{idx}"]=format_ticket(p.ticket); row[f"予測{idx}_戦略"]=p.strategy; row[f"予測{idx}_本数字一致"]=r.main_matches; row[f"予測{idx}_ボーナス一致"]=r.bonus_matches; row[f"予測{idx}_等級"]=_grade_label(r.grade); row[f"予測{idx}_当せん金額"]=r.prize
            if r.main_matches>=4: bank.add(p.ticket, 1.0+max(0,r.main_matches-4)*0.8)
        rows.append(row)
    def rate(v,t): return sum(1 for x in v if x>=t)/len(v) if v else 0.0
    summary={"検証回数":len(top),"初期学習回数":min_train,"検証開始回相当":start+1,"要求バックテスト候補プール":pool_size,"実効バックテスト候補プール":pool_size,"直近検証回数指定":max_backtest_draws,"1回あたり口数":num_tickets,"1口購入金額":DEFAULT_UNIT_COST,"1口目平均一致数":round(_avg(top),6),"全口ベスト平均一致数":round(_avg(best),6),"1口目_2個以上率":round(rate(top,2),6),"1口目_3個以上率":round(rate(top,3),6),"1口目_4個以上率":round(rate(top,4),6),"全口ベスト_2個以上率":round(rate(best,2),6),"全口ベスト_3個以上率":round(rate(best,3),6),"全口ベスト_4個以上率":round(rate(best,4),6),"総購入金額":purchase,"総当せん金額":prize,"総収支":prize-purchase,"総回収率":round(prize/purchase,6) if purchase else 0,"総当せん口数":sum(1 for row in rows for k,v in row.items() if k.endswith("_等級") and v != "ハズレ"),"全予測口等級分布":dict(sorted(gd.items())),"各回ベスト等級分布":dict(sorted(bgd.items())),"1口目一致数分布":dict(sorted(Counter(top).items())),"全口ベスト一致数分布":dict(sorted(Counter(best).items())),"最適化重み":weights.to_dict(),"MemoryBank件数":len(bank.items)}
    _write_csv(detail_csv, rows); _write_csv(summary_csv, [summary]); bank.save(MEMORYBANK_CSV); return summary

__all__=["AdvancedWeights","MemoryBank","advanced_predict","advanced_backtest","optimize_weights","save_latest_txt"]
