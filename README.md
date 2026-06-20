# NEW_LOTO7

LOTO7 のデータ更新、進化型モデル探索、walk-forward holdout検証、合議制予測、ML拡張、完全AI分析、管理付き自己進化AIをまとめて実行するリポジトリです。

> 注意: 宝くじはランダム性が高く、このリポジトリの予測・バックテスト・自己進化結果は、将来の当せんや利益を保証するものではありません。

---

## 現在の全体像

このリポジトリは、現在以下の流れで動きます。

```text
scrape
  ↓
evolve
  ↓
holdout
  ↓
ml-stack
  ↓
complete-ai
  ↓
monitor
  ↓
self-evolve
```

中心となるGitHub Actionsは次の1本です。

```text
.github/workflows/loto7_evolution.yml
```

以前は自己進化やTXTレポート集約を別workflowに分けていましたが、現在は `LOTO7 Evolution Trainer` に統合済みです。

---

## 主要機能

| 機能 | 内容 |
|---|---|
| CSV更新 | `scrapingloto7.py` で `loto7.csv` を更新 |
| 進化型モデル探索 | 8 shard 並列でモデル候補を世代交代・交叉・突然変異 |
| precision学習 | 高等級・近似一致を重視する `loto7_precision_evolution_trainer.py` |
| holdout再ランキング | 第2回から最新回まで未来リークなしで検証 |
| 合議制予測 | 8 shard の候補モデルを再スコアリングして5口を出力 |
| ML拡張 | MemoryBank / MetaClassifier / LightGBM / CatBoost / XGBoost / Optuna / SHAP |
| complete-ai | Meta Ensemble / Master Champion / ROI / Bayesian / Monte Carlo / MCTS / Dashboard |
| monitor | 進捗サマリーを `outputs/loto7_progress_summary.*` に出力 |
| self-evolve | 診断・改善案生成・安全判定・PR作成まで実施 |
| TXT集約 | `outputs/txt_reports/` に主要 `.txt/.md` を一括コピー |

---

## GitHub Actions

### LOTO7 Evolution Trainer

```text
Actions > LOTO7 Evolution Trainer > Run workflow
```

このworkflowが現在のメインです。

### 手動実行オプション

| 入力 | 内容 | 既定値 |
|---|---|---|
| `evolution_mode` | `recommended` / `full_power` / `custom` | `recommended` |
| `generations` | custom時の世代数 | `60` |
| `population` | custom時の個体数 | `120` |
| `max_targets` | custom時の検証対象数 | `160` |
| `target_stride` | 検証間隔 | `2` |
| `workers` | shard内worker数 | `2` |
| `reset_state` | stateを削除して最初から実行 | `false` |
| `run_ml_stack` | ML拡張を実行 | `true` |
| `run_complete_ai` | complete-aiを実行 | `true` |
| `run_self_evolve` | 自己進化AIを実行 | `true` |
| `create_self_evolution_pr` | 改善案がある場合にPR作成 | `true` |

### 定期実行

```text
UTC 15:00 / 23:00 / 07:00
JST 00:00 / 08:00 / 16:00
```

自己進化PRの乱発防止のため、schedule実行では UTC 23:00 の回だけ self-evolve を実行します。

---

## 自己進化AI

自己進化AIは以下のファイルで管理します。

```text
loto7_self_evolver.py
loto7_self_evolution_config.json
```

### 自己進化の段階

| Lv | 内容 | 状態 |
|---:|---|---|
| Lv1 | 進化型モデル学習結果を読む | 実装済み |
| Lv2 | 評価基準・precision scoringを調整 | 実装済み |
| Lv3 | 弱点診断と改善案生成 | 実装済み |
| Lv4 | 安全な設定変更として候補適用 | 実装済み |
| Lv5 | smoke testで検証 | 実装済み |
| Lv6 | 改善ブランチ・PR作成 | 実装済み |

### 安全設計

- mainへ直接pushしない
- 任意コード生成はしない
- 変更対象は安全なJSON設定に限定
- 実質差分がない場合は `decision: no_op`
- 時刻差分だけの重複PRは作成しない
- `ready_for_pull_request=true` の時だけPR作成対象

### 自己進化出力

```text
outputs/self_evolution/diagnosis.json
outputs/self_evolution/proposal.json
outputs/self_evolution/adoption_decision.json
outputs/self_evolution/comparison_report.txt
outputs/self_evolution/applied_config.json
outputs/self_evolution/pr_body.md
outputs/self_evolution/pr_summary.txt
```

---

## TXTレポート一括確認

主要な `.txt` / `.md` レポートは、以下の専用フォルダに集約します。

```text
outputs/txt_reports/
```

### 主なファイル

| ファイル | 内容 |
|---|---|
| `outputs/txt_reports/00_index.txt` | 一覧・おすすめ確認順 |
| `outputs/txt_reports/10_pr_summary.txt` | 自己進化PR/採用判断の要約 |
| `outputs/txt_reports/20_self_evolution_report.txt` | 自己進化AIの診断・改善案 |
| `outputs/txt_reports/30_latest_prediction_report.txt` | 最新予測5口 |
| `outputs/txt_reports/40_model_selection_report.txt` | 採用モデルと候補ランキング |
| `outputs/txt_reports/50_holdout_report.txt` | holdoutバックテスト概要 |
| `outputs/txt_reports/60_progress_summary.txt` | 学習進捗サマリー |
| `outputs/txt_reports/all_reports.txt` | 主要TXTの一括結合版 |
| `outputs/txt_reports/README.txt` | フォルダ説明 |

### 手動生成

```bash
bash scripts/collect_txt_reports.sh outputs/txt_reports
```

PRが作成される場合は、self-evolve job内で自動的に `outputs/txt_reports/` が作成され、PRブランチにも含まれます。

---

## 現在の代表的な成績

直近のholdout再ランキングでは、以下のような結果が出ています。

| 項目 | 値 |
|---|---:|
| 検証対象 | 第2回〜第682回相当 |
| 処理済み対象回数 | 681 |
| 総購入口数 | 3405 |
| 総購入額 | 1,021,500円 |
| 総払戻額 | 596,400円 |
| 総収支 | -425,100円 |
| ROI | 58.385% |
| 最大本数字一致 | 6 |
| 当選回数 | 99 |
| 当選口数 | 149 |
| 3等 | 1 |
| 4等 | 6 |
| 5等 | 55 |
| 6等 | 87 |

詳細は以下を確認してください。

```text
outputs/holdout/holdout_report.txt
outputs/txt_reports/50_holdout_report.txt
```

---

## 最新予測レポート

最新予測は以下に出力されます。

```text
outputs/holdout/latest_prediction_report.txt
outputs/evolution_best_prediction.csv
outputs/txt_reports/30_latest_prediction_report.txt
```

例:

```text
1位: 06 07 09 12 22 33 35
2位: 01 06 09 12 22 26 29
3位: 06 08 09 12 18 22 34
4位: 01 09 10 12 18 22 35
5位: 07 09 10 12 22 29 34
```

---

## 重要ファイル

| ファイル | 内容 |
|---|---|
| `loto7.csv` | LOTO7抽せんデータ |
| `scrapingloto7.py` | CSV更新 |
| `loto7_evolution_trainer.py` | 進化型モデル探索本体 |
| `loto7_precision_evolution_trainer.py` | precision scoring強化版 |
| `merge_evolution_shards.py` | shard統合・holdout再ランキング・合議制予測 |
| `holdout_evaluator.py` | full-period holdout検証 |
| `loto7_ml_stack.py` | ML拡張スタック |
| `loto7_complete_ai_system.py` | complete-ai統合分析 |
| `loto7_progress_monitor.py` | 進捗サマリー作成 |
| `loto7_self_evolver.py` | 自己進化AI制御 |
| `loto7_self_evolution_config.json` | 自己進化設定 |
| `scripts/collect_txt_reports.sh` | TXTレポート集約 |
| `.github/workflows/loto7_evolution.yml` | 統合workflow |

---

## 主要出力

| 出力 | 内容 |
|---|---|
| `loto7_best_model.json` | 採用モデル |
| `loto7_best_model_shardXX_of_08.json` | shard別ベストモデル |
| `outputs/evolution_history_shardXX_of_08.csv` | shard別進化履歴 |
| `outputs/evolution_best_summary_shardXX_of_08.csv` | shard別上位サマリー |
| `outputs/evolution_best_prediction.csv` | 最新予測CSV |
| `outputs/evolution_merged_summary.json` | 統合結果サマリー |
| `outputs/run_manifest.json` | 実行manifest |
| `outputs/holdout/holdout_result.csv` | holdout詳細CSV |
| `outputs/holdout/holdout_summary.json` | holdout集計JSON |
| `outputs/holdout/holdout_report.txt` | holdoutテキストレポート |
| `outputs/holdout/latest_prediction_report.txt` | 最新予測TXT |
| `outputs/holdout/model_selection_report.txt` | モデル選定TXT |
| `outputs/loto7_progress_summary.json` | 進捗JSON |
| `outputs/loto7_progress_summary.md` | 進捗Markdown |
| `outputs/txt_reports/` | TXT一括確認フォルダ |

---

## ローカル実行例

### precision進化学習の軽量実行

```bash
DISABLE_GIT_PUSH=1 python loto7_precision_evolution_trainer.py \
  --csv loto7.csv \
  --output-dir outputs/local_test \
  --best-model outputs/local_test/loto7_best_model.json \
  --generations 2 \
  --population 12 \
  --elite-count 3 \
  --purchase-count 5 \
  --min-train-draws 60 \
  --max-targets 20 \
  --target-stride 5 \
  --shard-id 0 \
  --num-shards 1 \
  --workers 1
```

### holdout検証

```bash
python holdout_evaluator.py \
  --csv loto7.csv \
  --best-model loto7_best_model.json \
  --holdout-start-draw 2 \
  --min-train-draws 1 \
  --purchase-count 5 \
  --output outputs/holdout/holdout_result.csv \
  --summary outputs/holdout/holdout_summary.json \
  --report outputs/holdout/holdout_report.txt \
  --state outputs/holdout/holdout_state.json \
  --resume \
  --fail-on-missing-prize
```

### TXTレポート集約

```bash
bash scripts/collect_txt_reports.sh outputs/txt_reports
```

---

## 運用メモ

### PRが作成された場合

まず以下を確認してください。

```text
outputs/txt_reports/00_index.txt
outputs/txt_reports/all_reports.txt
outputs/self_evolution/adoption_decision.json
outputs/self_evolution/proposal.json
loto7_self_evolution_config.json
```

### 不要PRの判断

以下の場合は採用しない方針です。

- `decision: no_op`
- 実質差分が時刻のみ
- `ready_for_pull_request: false`
- `outputs/self_evolution` の不要な履歴削除だけが含まれる
- `mergeable=false` かつ設定改善が重複している

### 採用PRの判断

以下の場合は採用候補です。

- `decision: propose_pr`
- `ready_for_pull_request: true`
- `loto7_self_evolution_config.json` に実質的な改善差分がある
- smoke testが通っている
- `outputs/txt_reports/10_pr_summary.txt` で改善理由が確認できる

---

## 免責

このリポジトリは、LOTO7の過去データ分析・バックテスト・予測補助・自己進化実験を目的としています。

- 当せんを保証しません
- 利益を保証しません
- バックテスト結果は将来成績を保証しません
- 購入判断は自己責任で行ってください
