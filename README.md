# NEW_LOTO7

`NEW_LOTO7`は、LOTO7の候補モデル生成、独立検証、5口ポートフォリオ選定、実運用履歴、封印証跡をGitHub Actionsで管理する実験リポジトリです。

> LOTO7は強いランダム性を持ちます。過去検証は将来の当せん・利益・回収率を保証しません。

## 現行アーキテクチャ

再利用実装の正本は`src/loto7/`です。ルートPythonと`scripts/`の移行済みファイルは、既存Workflow・import・Resume互換のための入口として残します。

```text
src/loto7/
├─ evaluation/
│  ├─ core.py          共通等級・払戻・ROI評価器
│  ├─ hit_metrics.py   本数字4+・5+・6+と5口多様性
│  └─ robust.py        払戻集中度と高一致率の統合診断
├─ evolution/
│  └─ hit_first.py     高一致率学習・時系列安定性・採用安全ゲート
├─ validation/
│  └─ hit_rate_gate.py 高一致率を優先するNested昇格ゲート
└─ paths.py            正規出力と旧Resumeパスの対応
```

ローカル導入:

```bash
python -m pip install -e .
```

## Workflow所有権

| 段階 | Workflow | 役割 |
|---|---|---|
| 全期間学習 | `LOTO7 Evolution Trainer` | 高一致率候補、Holdout、役割戦略 |
| 独立自己進化 | `LOTO7 Model Self Evolution` | 高一致率候補とResume state |
| 直近候補 | `LOTO7 Recent Era Self Evolution` | Recent / Super Recent候補 |
| sealed検証 | `LOTO7 Nested Walk Forward Validation` | Nested fold、高一致率・財務ゲート、モデル昇格 |
| G4診断 | `LOTO7 Generation 4 Evaluation` | Null League、PBO、Conformal、候補・Champion/Challenger診断 |
| 本番公開 | `LOTO7 Production Prediction Publisher` | 最新実績回+1の5口、累積履歴、結果照合、SHA-256封印 |
| 出力同期 | `LOTO7 Canonical Output Sync` | 4分類へ非破壊ミラー |
| 回帰確認 | `LOTO7 Validation Tests` | 評価器、未来リーク、所有権、Resume互換性 |
| 構造監査 | `Repository Structure Audit` | 所有権、不要ファイル、保持方針 |

**本番予測の唯一の所有者は`LOTO7 Production Prediction Publisher`です。**

Generation 4は候補・診断だけを`outputs/generation4/`へ出力します。採用可否にかかわらず、本番Publisherが現在承認済みモデルを使い、常に`最新の抽せん済み回号 + 1`を公開します。

## 高一致率学習

`loto7_model_self_evolver.py`は、候補評価、親プール順位、交叉・突然変異、世代最良、停滞探索、Resume、最終採用判定を同一の高一致率目的関数で処理します。

| 構成 | 重み | 内容 |
|---|---:|---|
| 本数字一致品質 | 70% | 平均最大一致、抽せん回単位の4+・5+・6+・7一致率 |
| 時系列安定性 | 20% | 評価期間4分割の中央値と最悪区間 |
| 5口多様性 | 10% | 使用数字数、口間重複度 |

ROI、profit、払戻額、最大払戻は学習スコアと候補順位に使用しません。候補採用時だけ安全ゲートとして検査します。

## 精度評価指標

```text
draw_main4_plus_rate
draw_main5_plus_rate
draw_main6_plus_rate
average_max_main_match
median_max_main_match
hit_objective_score
hit_first_objective_score
temporal_segment_match_score_min
temporal_segment_match_score_median
average_portfolio_unique_numbers
mean_ticket_pair_overlap
max_ticket_pair_overlap
portfolio_metrics_available
```

Holdout集計は`holdout_result.csv`の`ticket`列から抽せん回ごとの5口を復元します。ポートフォリオが欠損している場合、使用数字数や重複度を0として扱わず、`portfolio_metrics_available=false`と`null`を出力します。

## 採用ゲート

| ゲート | 現行条件 |
|---|---|
| Self Evolution | 高一致率目的、4+率、5+件数、平均最大一致、最悪時系列区間が非悪化 |
| Independent Holdout | 学習・選定・最終Holdoutを時系列分離 |
| Nested Walk-Forward | Foldごとに評価年をsealed化 |
| High-Match Gate | 2Fold以上で実質改善し、4+率・5+件数・平均最大一致が非悪化 |
| Nested合計ROI | 候補払戻率`>= 8.0%`かつ基準差`>= +0.5pt` |
| No-op拒否 | 同一モデルIDまたは同一SHA-256は昇格禁止 |
| 払戻集中度 | 最大1回払戻依存率`<= 50%` |
| Null Strategy League | `decision.passed == true`のみ採用可能 |
| Conformal | 過去データだけで4/7以上包含率を再校正 |
| Portfolio constraints | 5口、数字使用上限4、口間重複上限4、選出後の数字置換禁止 |

不合格時は本番モデルを置換しません。Publisherは既存承認モデルから次回予測を更新します。

## 出力構成

```text
outputs/
├─ production/   最新予測、累積履歴、実運用結果、公開レポート
├─ evidence/     Nested判定、採用拒否、SHA-256封印証跡
├─ state/        Full / Recent / Super RecentのResume state
└─ diagnostics/  Holdout、Role Ensemble、Generation 4診断
```

移行中は旧パスも維持します。大規模CSV、Fold内部データ、再生成可能な詳細診断はActions Artifactへ保存します。

## 実行方法

```text
Actions → LOTO7 Evolution Trainer
Actions → LOTO7 Model Self Evolution
Actions → LOTO7 Recent Era Self Evolution
Actions → LOTO7 Nested Walk Forward Validation
Actions → LOTO7 Generation 4 Evaluation
Actions → LOTO7 Production Prediction Publisher
Actions → LOTO7 Canonical Output Sync
```

## テスト

```bash
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/check_repository_architecture.py \
  --json /tmp/loto7-architecture.json \
  --markdown /tmp/loto7-architecture.md
python scripts/migrate_output_layout.py --verify-only --manifest /tmp/loto7-layout.json
```

構造ポリシー:

```text
config/repository_layout.json
config/output_layout.json
docs/architecture/REPOSITORY_LAYOUT.md
docs/architecture/WORKFLOW_OWNERSHIP.md
docs/architecture/OUTPUT_RETENTION.md
```
