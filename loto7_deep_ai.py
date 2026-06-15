#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loto7_deep_ai.py

LOTO7 Deep AI module.

実装:
  - Transformer予測
    過去draw系列を37次元 multi-hot に変換し、次回37数字の出現確率を予測する。
  - PPO本格強化学習風Policy
    7数字選択をMulti-Bernoulli actionとして扱い、clip objective/value loss/entropyで更新する。

制約:
  - GitHub Actions CPUでも完走できる軽量設定をデフォルトにする。
  - ロト7はランダム抽せんであり、的中・収益を保証しない。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

NUMBERS = list(range(1, 38))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Draw:
    draw_no: int
    date: str
    main: Tuple[int, ...]
    bonus: Tuple[int, ...]


def parse_nums(text: object) -> Tuple[int, ...]:
    return tuple(int(x) for x in str(text or "").replace(",", " ").split() if x.isdigit())


def draw_no_int(text: object) -> Optional[int]:
    import re
    m = re.search(r"\d+", str(text or ""))
    return int(m.group(0)) if m else None


def load_draws(csv_path: str) -> List[Draw]:
    rows: List[Draw] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            no = draw_no_int(row.get("回別"))
            main = parse_nums(row.get("本数字"))
            bonus = parse_nums(row.get("ボーナス数字"))
            date = str(row.get("抽せん日") or "").strip()
            if no is None or len(main) != 7 or len(set(main)) != 7:
                continue
            if len(bonus) != 2:
                bonus = tuple()
            rows.append(Draw(no, date, tuple(sorted(main)), tuple(sorted(bonus))))
    return sorted(rows, key=lambda d: d.draw_no)


def multihot(draw: Draw) -> np.ndarray:
    x = np.zeros(37, dtype=np.float32)
    for n in draw.main:
        x[n - 1] = 1.0
    return x


def build_seq_dataset(draws: Sequence[Draw], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i in range(seq_len, len(draws)):
        xs.append(np.stack([multihot(d) for d in draws[i-seq_len:i]], axis=0))
        ys.append(multihot(draws[i]))
    if not xs:
        raise ValueError("not enough draws")
    return torch.tensor(np.stack(xs), dtype=torch.float32), torch.tensor(np.stack(ys), dtype=torch.float32)


class LotoTransformer(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 96, nhead: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(37, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*3, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 37))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x) + self.pos[:, :x.size(1)]
        z = self.encoder(z)
        pooled = z[:, -1, :]
        return self.head(pooled)


class PPOPolicy(nn.Module):
    def __init__(self, state_dim: int = 37, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 37)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


def ticket_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits / temperature)
    dist = Bernoulli(probs=probs)
    raw = dist.sample()
    # enforce exactly 7 numbers by top probability among sampled-biased score
    score = probs + raw * 0.15
    idx = torch.topk(score, k=7).indices.sort().values
    action = torch.zeros_like(probs)
    action[idx] = 1.0
    logprob = (action * torch.log(probs + 1e-8) + (1-action) * torch.log(1-probs + 1e-8)).sum()
    entropy = dist.entropy().sum()
    return [int(i.item()) + 1 for i in idx], logprob, entropy


def reward_ticket(ticket: Sequence[int], target: Draw) -> float:
    m = len(set(ticket) & set(target.main))
    b = len(set(ticket) & set(target.bonus))
    reward = 0.0
    reward += m * 0.4
    reward += max(0, m - 3) ** 2 * 0.7
    if m == 3 and b >= 1:
        reward += 3.0
    if m == 4:
        reward += 6.0
    if m == 5:
        reward += 30.0
    if m == 6:
        reward += 120.0
    if m == 7:
        reward += 500.0
    # structural regularization
    odd = sum(n % 2 for n in ticket)
    s = sum(ticket)
    reward += 0.3 if odd in (3,4) else -0.2
    reward += 0.3 if 85 <= s <= 190 else -0.4
    return reward


def train_transformer(draws: Sequence[Draw], out_dir: Path, seq_len: int, epochs: int, batch_size: int, lr: float) -> Dict[str, object]:
    x, y = build_seq_dataset(draws, seq_len)
    split = max(1, int(len(x) * 0.8))
    x_train, y_train = x[:split].to(DEVICE), y[:split].to(DEVICE)
    x_val, y_val = x[split:].to(DEVICE), y[split:].to(DEVICE)
    model = LotoTransformer(seq_len=seq_len).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train), device=DEVICE)
        losses = []
        for start in range(0, len(x_train), batch_size):
            idx = perm[start:start+batch_size]
            logits = model(x_train[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            if len(x_val) > 0:
                val_logits = model(x_val)
                val_loss = F.binary_cross_entropy_with_logits(val_logits, y_val).item()
                top7 = torch.topk(torch.sigmoid(val_logits), 7, dim=1).indices
                hits = []
                for i in range(len(top7)):
                    pred = set((top7[i] + 1).detach().cpu().numpy().tolist())
                    actual = set(np.where(y_val[i].detach().cpu().numpy() > 0.5)[0] + 1)
                    hits.append(len(pred & actual))
                avg_hit = float(np.mean(hits)) if hits else 0.0
            else:
                val_loss = 0.0
                avg_hit = 0.0
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_avg_hit": avg_hit})

    latest_seq = torch.tensor(np.stack([multihot(d) for d in draws[-seq_len:]])[None, :, :], dtype=torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(latest_seq))[0].detach().cpu().numpy()
    ranking = sorted([(i+1, float(p)) for i, p in enumerate(probs)], key=lambda x: x[1], reverse=True)
    top7 = sorted([n for n, _ in ranking[:7]])
    torch.save(model.state_dict(), out_dir / "transformer_model.pt")
    pd.DataFrame(history).to_csv(out_dir / "transformer_training_history.csv", index=False)
    pd.DataFrame([{"number": n, "probability": p} for n, p in ranking]).to_csv(out_dir / "transformer_number_probabilities.csv", index=False)
    return {"seq_len": seq_len, "epochs": epochs, "prediction": top7, "val_avg_hit_last": history[-1]["val_avg_hit"] if history else 0.0, "val_loss_last": history[-1]["val_loss"] if history else 0.0}


def state_from_history(draws: Sequence[Draw], upto: int, window: int = 60) -> np.ndarray:
    subset = draws[max(0, upto-window):upto]
    counts = np.ones(37, dtype=np.float32) * 0.01
    for age, d in enumerate(reversed(subset)):
        w = 0.985 ** age
        for n in d.main:
            counts[n-1] += w
    counts = counts / max(1e-6, counts.max())
    return counts.astype(np.float32)


def train_ppo(draws: Sequence[Draw], out_dir: Path, epochs: int, episodes_per_epoch: int, lr: float, clip_eps: float) -> Dict[str, object]:
    policy = PPOPolicy().to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = random.Random(777)
    history = []
    min_idx = 80 if len(draws) > 100 else max(10, len(draws)//3)
    for ep in range(1, epochs + 1):
        states, actions, old_logps, rewards, values, entropies = [], [], [], [], [], []
        for _ in range(episodes_per_epoch):
            idx = rng.randint(min_idx, len(draws)-1)
            state_np = state_from_history(draws, idx)
            state = torch.tensor(state_np, dtype=torch.float32, device=DEVICE)
            logits, value = policy(state)
            ticket, logp, ent = ticket_from_logits(logits, temperature=max(0.8, 1.2 - ep * 0.01))
            r = reward_ticket(ticket, draws[idx])
            action_vec = torch.zeros(37, device=DEVICE)
            for n in ticket:
                action_vec[n-1] = 1.0
            states.append(state)
            actions.append(action_vec)
            old_logps.append(logp.detach())
            rewards.append(r)
            values.append(value.detach())
            entropies.append(ent.detach())

        states_t = torch.stack(states)
        actions_t = torch.stack(actions)
        old_logps_t = torch.stack(old_logps)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        values_t = torch.stack(values)
        adv = rewards_t - values_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        returns = rewards_t

        for _ in range(4):
            logits, vals = policy(states_t)
            probs = torch.sigmoid(logits)
            new_logps = (actions_t * torch.log(probs + 1e-8) + (1-actions_t) * torch.log(1-probs + 1e-8)).sum(dim=1)
            ratio = torch.exp(new_logps - old_logps_t)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(vals, returns)
            entropy = Bernoulli(probs=probs).entropy().sum(dim=1).mean()
            loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        history.append({"epoch": ep, "reward_avg": float(np.mean(rewards)), "reward_max": float(np.max(rewards)), "loss": float(loss.detach().cpu()), "entropy": float(entropy.detach().cpu())})

    latest_state = torch.tensor(state_from_history(draws, len(draws)), dtype=torch.float32, device=DEVICE)
    policy.eval()
    with torch.no_grad():
        logits, value = policy(latest_state)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    ranking = sorted([(i+1, float(p)) for i, p in enumerate(probs)], key=lambda x: x[1], reverse=True)
    top7 = sorted([n for n, _ in ranking[:7]])
    torch.save(policy.state_dict(), out_dir / "ppo_policy.pt")
    pd.DataFrame(history).to_csv(out_dir / "ppo_training_history.csv", index=False)
    pd.DataFrame([{"number": n, "probability": p} for n, p in ranking]).to_csv(out_dir / "ppo_number_probabilities.csv", index=False)
    return {"epochs": epochs, "episodes_per_epoch": episodes_per_epoch, "prediction": top7, "reward_avg_last": history[-1]["reward_avg"] if history else 0.0, "value_estimate": float(value.detach().cpu())}


def merge_predictions(transformer: Dict[str, object], ppo: Dict[str, object], out_dir: Path) -> Dict[str, object]:
    rows = []
    trans_probs = pd.read_csv(out_dir / "transformer_number_probabilities.csv")
    ppo_probs = pd.read_csv(out_dir / "ppo_number_probabilities.csv")
    tmap = dict(zip(trans_probs["number"], trans_probs["probability"]))
    pmap = dict(zip(ppo_probs["number"], ppo_probs["probability"]))
    for n in NUMBERS:
        score = 0.55 * float(tmap.get(n, 0.0)) + 0.45 * float(pmap.get(n, 0.0))
        rows.append({"number": n, "transformer_prob": float(tmap.get(n, 0.0)), "ppo_prob": float(pmap.get(n, 0.0)), "deep_score": score})
    rows.sort(key=lambda r: r["deep_score"], reverse=True)
    prediction = sorted([r["number"] for r in rows[:7]])
    pd.DataFrame(rows).to_csv(out_dir / "deep_ai_number_ranking.csv", index=False)
    with open(out_dir / "deep_ai_prediction.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["numbers", "source"])
        writer.writeheader()
        writer.writerow({"numbers": " ".join(f"{n:02d}" for n in prediction), "source": "transformer_ppo_ensemble"})
    return {"prediction": prediction, "top_numbers": rows[:12]}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LOTO7 PPO + Transformer predictor")
    ap.add_argument("--csv", default="loto7.csv")
    ap.add_argument("--output-dir", default="outputs/deep_ai")
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--transformer-epochs", type=int, default=20)
    ap.add_argument("--ppo-epochs", type=int, default=20)
    ap.add_argument("--ppo-episodes", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    draws = load_draws(args.csv)
    if len(draws) < args.seq_len + 20:
        raise SystemExit("not enough data")

    torch.manual_seed(777)
    np.random.seed(777)
    random.seed(777)

    transformer = train_transformer(draws, out_dir, args.seq_len, args.transformer_epochs, args.batch_size, 8e-4)
    ppo = train_ppo(draws, out_dir, args.ppo_epochs, args.ppo_episodes, 3e-4, 0.2)
    ensemble = merge_predictions(transformer, ppo, out_dir)
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": str(DEVICE),
        "transformer": transformer,
        "ppo": ppo,
        "ensemble": ensemble,
        "disclaimer": "LOTO7 is random; predictions do not guarantee winning or profit.",
    }
    (out_dir / "deep_ai_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# LOTO7 Deep AI Summary",
        "",
        f"generated_at: {summary['generated_at']}",
        f"device: {summary['device']}",
        "",
        f"Transformer prediction: `{ ' '.join(f'{n:02d}' for n in transformer['prediction']) }`",
        f"PPO prediction: `{ ' '.join(f'{n:02d}' for n in ppo['prediction']) }`",
        f"Ensemble prediction: `{ ' '.join(f'{n:02d}' for n in ensemble['prediction']) }`",
        "",
        "LOTO7 is random; this does not guarantee winning or profit.",
    ]
    (out_dir / "deep_ai_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
