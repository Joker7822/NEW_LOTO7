# NEW_LOTO7

`NEW_LOTO7`は、LOTO7の候補モデル生成、独立検証、5口ポートフォリオ選定、実運用履歴、封印証跡をGitHub Actionsで管理する実験リポジトリです。

> LOTO7は強いランダム性を持ちます。過去検証は将来の当せん・利益・回収率を保証しません。

## 現行アーキテクチャ

再利用実装の正本は`src/loto7/`です。ルートPythonと`scripts/`の移行済みファイルは、既存Workflow・import・Resume互換のための入口としてのみ残します。

```text
src/loto7/
├─ evaluation/
│  ├─ core.py          共通等級・払戻・ROI評価器
│  ├─ hit_metrics.py   本数字4+・5+・6+と5口多様性
│  └─ robust.py        払戻集中度と高一致率の統合診断
├─ evolution/
│  ├─ hit_first.py     高一致率学習・時系列安定性
│  └─ generation5.py   5-Fold・Pareto・固定Null昇格
├─ repository/
│  ├─ layout.py        構造ポリシーと出力分類
│  └─ audit.py         fail-closed構造監査
├─ validation/
│  └─ hit_rate_gate.py Nested高一致率昇格ゲート
└─ paths.py            正規出力と旧Resumeパスの対応
```

ローカル導入:

```bash
python -m pip install -e .
```

## リポジトリ構成

```text
.github/workflows/        Actionsオーケストレーション
config/                   構造・Workflow・出力ポリシー
docs/architecture/        設計文書と生成監査レポート
outputs/production/       公開予測・累積履歴
outputs/evidence/         封印・採用／拒否証跡
outputs/state/            Resume state
outputs/diagnostics/      Holdout・Role・G4・G5診断
scripts/                  薄いCLI／互換入口
src/loto7/                正規実装
tests/                    回帰・統合・互換テスト
```

構造ポリシーの正本:

```text
config/repository_layout.json
config/workflow_registry.json
config/output_layout.json
docs/architecture/REPOSITORY_LAYOUT.md
```

ルートPythonは固定allowlistです。新しい実装をルートへ追加すると、`Repository Structure Audit`が失敗します。

## Workflow所有権

全Workflowの機械可読台帳は`config/workflow_registry.json`です。

| 段階 | Workflow | 役割 |
|---|---|---|
| 全期間候補 | `LOTO7 Evolution Trainer` | 高一致率候補、Holdout、役割戦略 |
| 独立候補 | `LOTO7 Model Self Evolution` | 候補とResume state |
| 直近候補 | `LOTO7 Recent Era Self Evolution` | Recent / Super Recent候補 |
| Nested検証 | `LOTO7 Nested Walk Forward Validation` | 時系列分離検証と証跡 |
| G4診断 | `LOTO7 Generation 4 Evaluation` | Null、PBO、Conformal、Champion/Challenger |
| G5昇格 | `LOTO7 Generation 5 Precision Evolution` | 5-Fold・Pareto・固定Nullによる自動モデル昇格 |
| 本番公開 | `LOTO7 Production Prediction Publisher` | 最新実績回+1の5口、履歴、SHA-256封印 |
| 出力同期 | `LOTO7 Canonical Output Sync` | 4分類への非破壊ミラー |
| 構造監査 | `Repository Structure Audit` | Workflow・出力・互換層のfail-closed検証 |

**本番予測の唯一の所有者は`LOTO7 Production Prediction Publisher`です。**

**自動Best Model昇格の所有者は`LOTO7 Generation 5 Precision Evolution`です。**

Generation 4は候補・診断だけを出力します。Generation 5候補が全ゲートに不合格の場合、本番モデルは置換されず、Publisherが既存承認モデルから次回予測を更新します。

## Generation 5

Generation 5は次を実装しています。

- 5分割の時系列Walk-Forward
- Pareto多目的選択
- 平均一致、4+、5+/6+、安定性・多様性の4-Island探索
- 104回→260回→全期間のSuccessive Halving
- 1,000系列の決定的Null Seed Bank
- 学習700／選択150／最終150の完全分離
- Null超過率、PBO、最大払戻依存による最終拒否

ROI、profit、払戻額、最大払戻は学習スコアへ加点せず、採用時の安全ゲートだけに使用します。

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

| ゲート | 条件 |
|---|---|
| Generation 5 Fold | 5Fold中3Fold以上で実質改善 |
| 平均最大一致 | 基準から+0.03以上 |
| 回別4個以上率 | 基準から+0.50ポイント以上 |
| 5個・6個以上 | 件数悪化禁止 |
| 最悪Fold | 基準比−2%以内 |
| 5口多様性 | 平均使用数字13以上、平均重複4.2以下、最大重複4以下 |
| Null Strategy League | Null超過率10%以下 |
| PBO | 40%以下 |
| 払戻集中度 | 最大1回払戻依存率50%以下 |
| No-op拒否 | 同一モデルIDまたは同一SHA-256は昇格禁止 |
| Portfolio constraints | 5口、数字使用上限4、口間重複上限4 |

不合格時は本番モデルを置換しません。

## 出力構成

```text
outputs/
├─ production/   最新予測、累積履歴、実運用結果、公開レポート
├─ evidence/     Nested判定、採用拒否、SHA-256封印証跡
├─ state/        Full / Recent / Super RecentのResume state
└─ diagnostics/  Holdout、Role、Generation 4、Generation 5診断
```

移行中は旧パスも読み取り可能です。新規実装は正規パスを優先し、旧パスをfallbackとして扱います。大規模CSV、Fold内部データ、候補集団、全Null結果はActions Artifactへ保存します。

## 実行方法

```text
Actions → LOTO7 Evolution Trainer
Actions → LOTO7 Model Self Evolution
Actions → LOTO7 Recent Era Self Evolution
Actions → LOTO7 Nested Walk Forward Validation
Actions → LOTO7 Generation 4 Evaluation
Actions → LOTO7 Generation 5 Precision Evolution
Actions → LOTO7 Production Prediction Publisher
Actions → LOTO7 Canonical Output Sync
Actions → Repository Structure Audit
```

## テスト

```bash
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py' -v
loto7-repository-audit \
  --json /tmp/loto7-architecture.json \
  --markdown /tmp/loto7-architecture.md
python scripts/migrate_output_layout.py --verify-only --manifest /tmp/loto7-layout.json
```

構造監査は、未登録トップレベルディレクトリ、未登録ルートPython、Workflow名重複、Workflow台帳漏れ、本番writer重複、禁止Workflow、非標準concurrency、互換wrapper肥大化を検出します。
