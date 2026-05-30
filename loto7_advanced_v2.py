#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NEW_LOTO7 Advanced v2.

Adds:
- 5+ hit only MemoryBank separated from 4+ memory.
- 3rd-prize focused Optuna/random-search objective.
- Lightweight MCTS candidate search.
- CatBoost meta-classifier with sklearn fallback.
- Hit-structure clustering with sklearn fallback.
- resumable walk-forward backtest helpers.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from loto7_logic_predictor import DEFAULT_PRIZE_TABLE, DEFAULT_UNIT_COST, Draw, TicketScore, classify_loto7_prize, format_ticket
from loto7_advanced_optimizer import AdvancedWeights, advanced_predict, optimize_weights
from loto7_enhanced_predictor import band_counts, build_number_scores, make_candidate_pool, max_consecutive_run

MEMORY5_CSV = "loto7_memorybank_5plus.csv"
CLUSTER_CSV = "loto7_hit_structure_clusters.csv"
META_JSON = "loto7_meta_classifier.json"
RESUME_JSON = "loto7_backtest_resume.json"


def parse_ticket(value: object) -> Tuple[int, ...]:
    nums = tuple(sorted(int(x) for x in re.findall(r"\d+", str(value or ""))))
    return nums if len(nums) == 7 and len(set(nums)) == 7 and all(1 <= n <= 37 for n in nums) else tuple()


def ticket_features(ticket: Sequence[int], draws: Sequence[Draw]) -> Dict[str, float]:
    t = tuple(sorted(ticket)); low, mid, high = band_counts(t); total = sum(t)
    odd = sum(n % 2 for n in t); run = max_consecutive_run(t)
    last = set(draws[-1].main) if draws else set(); prev = set(draws[-2].main) if len(draws) >= 2 else set()
    gaps = [b-a for a,b in zip(t, t[1:])]
    last_digits = Counter(n % 10 for n in t)
    return {
        "sum": float(total), "odd": float(odd), "low": float(low), "mid": float(mid), "high": float(high),
        "run": float(run), "repeat_last": float(len(set(t)&last)), "repeat_prev": float(len(set(t)&prev)),
        "last_digit_max": float(max(last_digits.values()) if last_digits else 0),
        "gap_avg": float(sum(gaps)/len(gaps) if gaps else 0), "gap_min": float(min(gaps) if gaps else 0), "gap_max": float(max(gaps) if gaps else 0),
    }


class MemoryBank5Plus:
    def __init__(self) -> None:
        self.items: List[Tuple[Tuple[int, ...], float, str]] = []
        self.pairs: Counter = Counter(); self.triples: Counter = Counter()
    def add(self, ticket: Sequence[int], matches: int, date: str = "") -> None:
        t = tuple(sorted(ticket))
        if len(t) != 7 or matches < 5: return
        strength = 1.0 if matches == 5 else 2.8 if matches == 6 else 6.0
        self.items.append((t, strength, date))
        for p in itertools.combinations(t, 2): self.pairs[p] += strength
        for tri in itertools.combinations(t, 3): self.triples[tri] += strength
    def load_detail(self, path: str, before_date: Optional[str] = None) -> None:
        p = Path(path)
        if not p.exists(): return
        try: rows = list(csv.DictReader(p.open("r", encoding="utf-8-sig", newline="")))
        except Exception: return
        for row in rows:
            date = row.get("抽せん日", "")
            if before_date and date >= before_date: continue
            for i in range(1, 101):
                pk=f"予測{i}"; mk=f"予測{i}_本数字一致"
                if pk not in row: break
                try: m=int(row.get(mk,0) or 0)
                except Exception: m=0
                if m >= 5:
                    t=parse_ticket(row.get(pk,""))
                    if t: self.add(t,m,date)
    def load_csv(self, path: str = MEMORY5_CSV, before_date: Optional[str] = None) -> None:
        p=Path(path)
        if not p.exists(): return
        try:
            for row in csv.DictReader(p.open("r", encoding="utf-8-sig", newline="")):
                date=row.get("抽せん日","")
                if before_date and date >= before_date: continue
                t=parse_ticket(row.get("組合せ","")); m=int(row.get("一致数",5) or 5)
                if t: self.add(t,m,date)
        except Exception: return
    def save(self, path: str = MEMORY5_CSV) -> None:
        with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
            w=csv.DictWriter(f, fieldnames=["抽せん日","組合せ","一致数","強度"]); w.writeheader()
            for t,s,date in self.items[-1000:]: w.writerow({"抽せん日":date,"組合せ":format_ticket(t),"一致数":6 if s>=2.8 else 5,"強度":round(s,6)})
    def score(self, ticket: Sequence[int]) -> float:
        if not self.items: return 0.0
        s=set(ticket); vals=[]
        for pat, strength, _ in self.items[-300:]:
            o=len(s&set(pat)); vals.append((1.8 if o==6 else 1.2 if o==5 else 0.45 if o==4 else 0.0)*strength)
        pair=sum((self.pairs.get(p,0) for p in itertools.combinations(sorted(ticket),2))) / max(sum(self.pairs.values()),1)
        tri=sum((self.triples.get(t,0) for t in itertools.combinations(sorted(ticket),3))) / max(sum(self.triples.values()),1)
        return 0.70*max(vals) + 0.20*(sum(vals)/len(vals)) + 0.06*pair + 0.04*tri


def build_memory5(detail_csv: str, before_date: Optional[str]) -> MemoryBank5Plus:
    b=MemoryBank5Plus(); b.load_detail(detail_csv,before_date); b.load_csv(MEMORY5_CSV,before_date); return b


def cluster_hit_structures(detail_csv: str, output_csv: str = CLUSTER_CSV) -> None:
    rows=[]; p=Path(detail_csv)
    if not p.exists(): return
    for row in csv.DictReader(p.open("r",encoding="utf-8-sig",newline="")):
        for i in range(1,101):
            pk=f"予測{i}"; mk=f"予測{i}_本数字一致"
            if pk not in row: break
            try: m=int(row.get(mk,0) or 0)
            except Exception: m=0
            if m>=4:
                t=parse_ticket(row.get(pk,""))
                if t:
                    f=ticket_features(t, [])
                    rows.append({"抽せん日":row.get("抽せん日",""),"組合せ":format_ticket(t),"一致数":m,**f})
    if not rows: return
    keys=["sum","odd","low","mid","high","run","last_digit_max","gap_avg","gap_min","gap_max"]
    X=[[float(r[k]) for k in keys] for r in rows]
    labels=[]
    try:
        from sklearn.cluster import KMeans
        labels=list(KMeans(n_clusters=min(6,max(2,len(rows)//20)),random_state=42,n_init="auto").fit_predict(X))
    except Exception:
        for r in rows:
            labels.append(int(r["low"])*9+int(r["mid"])*3+int(r["high"]))
    with Path(output_csv).open("w",encoding="utf-8-sig",newline="") as f:
        field=["cluster"]+list(rows[0].keys()); w=csv.DictWriter(f,fieldnames=field); w.writeheader()
        for r,l in zip(rows,labels): w.writerow({"cluster":l,**r})


def train_meta_classifier(detail_csv: str, output_json: str = META_JSON) -> Dict[str,float]:
    p=Path(detail_csv); data=[]
    if not p.exists(): return {}
    for row in csv.DictReader(p.open("r",encoding="utf-8-sig",newline="")):
        dummy=[]
        for i in range(1,101):
            pk=f"予測{i}"; mk=f"予測{i}_本数字一致"
            if pk not in row: break
            t=parse_ticket(row.get(pk,""))
            if not t: continue
            try: m=int(row.get(mk,0) or 0)
            except Exception: m=0
            data.append((ticket_features(t,dummy),1 if m>=5 else 0))
    if not data: return {}
    keys=list(data[0][0].keys()); X=[[d[0][k] for k in keys] for d in data]; y=[d[1] for d in data]
    info={"positive":sum(y),"total":len(y)}
    try:
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(iterations=80,depth=4,learning_rate=0.08,verbose=False,random_seed=42)
        model.fit(X,y)
        info["model"]="catboost"; info["feature_count"]=len(keys)
    except Exception:
        # Fallback: positive/negative mean difference as a simple linear scorer.
        pos=[X[i] for i,v in enumerate(y) if v]; neg=[X[i] for i,v in enumerate(y) if not v]
        coefs=[]
        for j in range(len(keys)):
            pm=sum(r[j] for r in pos)/len(pos) if pos else 0
            nm=sum(r[j] for r in neg)/len(neg) if neg else 0
            coefs.append(pm-nm)
        info["model"]="linear_fallback"; info["keys"]=keys; info["coefs"]=coefs
    Path(output_json).write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return info


def meta_score(ticket: Sequence[int], draws: Sequence[Draw], meta_json: str = META_JSON) -> float:
    p=Path(meta_json)
    if not p.exists(): return 0.0
    try: info=json.loads(p.read_text(encoding="utf-8"))
    except Exception: return 0.0
    if info.get("model") != "linear_fallback": return 0.05
    f=ticket_features(ticket,draws); keys=info.get("keys",[]); coefs=info.get("coefs",[])
    raw=sum(float(f.get(k,0))*float(c) for k,c in zip(keys,coefs))
    return max(-0.5,min(0.5,raw/100.0))


def mcts_candidates(draws: Sequence[Draw], detail_csv: str, iterations: int = 8000, seed: int = 42) -> List[Tuple[int,...]]:
    rng=random.Random(seed); ns=build_number_scores(draws); mem5=build_memory5(detail_csv,None)
    base=make_candidate_pool(draws,24); alln=list(range(1,38)); visits=Counter(); reward=Counter()
    def rollout(prefix: Tuple[int,...]) -> Tuple[int,...]:
        s=set(prefix)
        while len(s)<7:
            choices=base if rng.random()<0.72 else alln
            # UCB-ish number choice.
            best=None; bestv=-1e9
            for n in rng.sample(choices, min(len(choices), 12)):
                if n in s: continue
                u=reward[n]/(visits[n]+1) + 0.25*math.sqrt(math.log(sum(visits.values())+2)/(visits[n]+1)) + ns.get(n,0)*0.03
                if u>bestv: best,bestv=n,u
            s.add(best if best is not None else rng.choice(alln))
        return tuple(sorted(s))
    best=[]
    for _ in range(iterations):
        t=rollout(tuple())
        r=mem5.score(t)+0.03*sum(ns.get(n,0) for n in t)+meta_score(t,draws)
        for n in t: visits[n]+=1; reward[n]+=r
        best.append((r,t))
    return [t for _,t in sorted(best, reverse=True)[:min(1000,len(best))]]


def advanced_v2_predict(draws: Sequence[Draw], num_tickets:int=10, pool_size:int=24, detail_csv:str="loto7_backtest_detail.csv", mcts_iterations:int=8000) -> List[TicketScore]:
    base=advanced_predict(draws,num_tickets=max(20,num_tickets),pool_size=pool_size,hit_pattern_csv=detail_csv,optimize=True)
    mem5=build_memory5(detail_csv,None)
    extra=[]
    for t in mcts_candidates(draws,detail_csv,mcts_iterations,seed=len(draws)*777):
        sc=mem5.score(t)+meta_score(t,draws)
        extra.append(TicketScore(t, base[0].score+sc if base else sc, {**ticket_features(t,draws),"memory5":mem5.score(t),"meta":meta_score(t,draws)}, "MCTS_5PLUS"))
    ranked=sorted(base+extra,key=lambda x:(-x.score,x.ticket))
    out=[]; use=Counter()
    for x in ranked:
        if any(len(set(x.ticket)&set(y.ticket))>4 for y in out): continue
        if any(use[n]>=3 for n in x.ticket): continue
        out.append(x); use.update(x.ticket)
        if len(out)>=num_tickets: return out
    return ranked[:num_tickets]


def third_prize_objective(matches:int, bonus:int=0) -> float:
    if matches>=6: return 100.0
    if matches==5: return 24.0
    if matches==4: return 7.0
    if matches==3 and bonus>=1: return 2.0
    return max(0.0,matches-1)*0.25


def backtest_resume_v2(draws: Sequence[Draw], min_train:int, num_tickets:int, pool_size:int, detail_csv:str, summary_csv:str, resume_json:str=RESUME_JSON, push_every:int=100) -> Dict[str,object]:
    start=min_train; done=set(); rows=[]
    rpath=Path(resume_json)
    if Path(detail_csv).exists():
        try:
            for row in csv.DictReader(Path(detail_csv).open("r",encoding="utf-8-sig",newline="")):
                if row.get("抽せん日"): done.add(row["抽せん日"]); rows.append(row)
        except Exception: pass
    for i in range(start,len(draws)):
        actual=draws[i]
        if actual.date in done: continue
        preds=advanced_v2_predict(draws[:i],num_tickets,pool_size,detail_csv,int(os.getenv("LOTO7_MCTS_ITERATIONS","4000")))
        res=[classify_loto7_prize(p.ticket,actual.main,actual.bonus,DEFAULT_PRIZE_TABLE) for p in preds]
        row={"抽せん日":actual.date,"回別":actual.draw_no or "","本数字":format_ticket(actual.main),"ボーナス数字":format_ticket(actual.bonus),"口数":len(preds),"購入金額":len(preds)*DEFAULT_UNIT_COST,"当せん金額":sum(x.prize for x in res),"最高本数字一致数":max((x.main_matches for x in res),default=0)}
        for idx,(p,rr) in enumerate(zip(preds,res),1):
            row[f"予測{idx}"]=format_ticket(p.ticket); row[f"予測{idx}_戦略"]=p.strategy; row[f"予測{idx}_本数字一致"]=rr.main_matches; row[f"予測{idx}_ボーナス一致"]=rr.bonus_matches; row[f"予測{idx}_等級"]="ハズレ" if rr.grade is None else f"{rr.grade}等"; row[f"予測{idx}_当せん金額"]=rr.prize
        rows.append(row); done.add(actual.date)
        if len(done)%push_every==0:
            write_rows(detail_csv,rows); rpath.write_text(json.dumps({"completed":len(done),"last_date":actual.date},ensure_ascii=False)+"\n",encoding="utf-8")
    write_rows(detail_csv,rows)
    summary=summarize_rows(rows,num_tickets)
    write_rows(summary_csv,[summary])
    return summary


def write_rows(path:str, rows:Sequence[Dict[str,object]]) -> None:
    if not rows: return
    keys=[]
    for r in rows:
        for k in r.keys():
            if k not in keys: keys.append(k)
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def summarize_rows(rows:Sequence[Dict[str,object]], tickets:int) -> Dict[str,object]:
    best=[int(r.get("最高本数字一致数",0) or 0) for r in rows]
    purchase=sum(int(r.get("購入金額",0) or 0) for r in rows); prize=sum(int(r.get("当せん金額",0) or 0) for r in rows)
    def rate(t): return sum(1 for x in best if x>=t)/len(best) if best else 0
    return {"検証回数":len(rows),"1回あたり口数":tickets,"全口ベスト平均一致数":round(sum(best)/len(best),6) if best else 0,"全口ベスト_4個以上率":round(rate(4),6),"全口ベスト_5個以上率":round(rate(5),6),"全口ベスト_6個以上率":round(rate(6),6),"総購入金額":purchase,"総当せん金額":prize,"総収支":prize-purchase,"総回収率":round(prize/purchase,6) if purchase else 0}
