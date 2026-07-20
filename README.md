# NEW_LOTO7

`NEW_LOTO7` は、LOTO7の候補モデル生成、独立検証、5口ポートフォリオ選定、実運用履歴、封印証跡をGitHub Actionsで管理するリポジトリです。

> LOTO7は強いランダム性を持ちます。本リポジトリの検証結果は、将来の当せん・利益・回収率を保証しません。

## 現行アーキテクチャ

再利用する実装の正本は`src/loto7/`です。ルートPythonと`scripts/`の移行済みファイルは、既存Workflow・import・Resumeを維持するための互換ラッパーとして残します。

```text
src/loto7/
├─ evaluation/
│  ├─ core.py          共通等級・払戻・ROI評価器
│  ├─ hit_metrics.py   本数字4個以上・5個以上・6個以上の高一致率指標
│  └─ robust.py        払戻集中度と高一致率の統合診断
├─ validation/
│  └─ hit_rate_gate.py 高一致率を優先するNested昇格ゲート
└─ paths.py            正規出力と旧Resumeパスの対応
```

パッケージのローカル導入:

```bash
python -m pip install -e .
```

本番予測の唯一の所有者は **LOTO7 Generation 4 Production** です。

| 段階 | Workflow | 役割 |
|---|---|---|
| 全期間学習 | `LOTO7 Evolution Trainer` | 全期間モデル、holdout、役割戦略 |
| 独立自己進化 | `LOTO7 Model Self Evolution` | 全期間候補とResume state |
| 直近候補 | `LOTO7 Recent Era Self Evolution` | Recent / Super Recent候補 |
| sealed検証 | `LOTO7 Nested Walk Forward Validation` | Nested fold、高一致率ゲート、財務ゲート、昇格 |
| 本番予測 | `LOTO7 Generation 4 Production` | 5口、履歴、Null League、e-process、SHA-256封印 |
| 出力同期 | `LOTO7 Canonical Output Sync` | 全Workflowの成果物を4分類へ非破壊同期 |
| 回帰確認 | `LOTO7 Validation Tests` | 評価器、未来リーク、Workflow、Resume互換性 |
| 構造監査 | `Repository Structure Audit` | 所有権、不要ファイル、保持方針 |

## 精度評価方針

モデル選定は、単純な収支や6等以上の当選口数だけでは評価しません。5口を1セットとして、1回の抽せんでどこまで本数字へ近づけたかを主に評価します。

主要指標:

```text
draw_main4_plus_rate
draw_main5_plus_rate
draw_main6_plus_rate
average_max_main_match
median_max_main_match
hit_objective_score
average_portfolio_unique_numbers
mean_ticket_pair_overlap
max_ticket_pair_overlap
```

`hit_objective_score`は払戻額を使用せず、平均最大一致数、4個以上・5個以上・6個以上への到達率、5口の数字カバー率を統合します。収支、最大払戻依存率、Null Strategy Leagueは安全ゲートとして残します。

## 採用ゲート

候補は次のすべてを通過した場合だけ本番へ昇格します。

| ゲート | 現行条件 |
|---|---|
| Independent Holdout | 学習・選定・最終holdoutを時系列分離 |
| Nested Walk-Forward | Foldごとに評価年をsealed化 |
| High-Match Gate | `hit_objective_score`が2Fold以上で実質改善、4個以上率・5個以上件数・平均最大一致数が非悪化 |
| Nested合計ROI | 候補払戻率`>= 8.0%`かつ基準差`>= +0.5pt` |
| No-op拒否 | 同一モデルIDまたは同一SHA-256は昇格禁止 |
| 払戻集中度 | 最大1回払戻依存率`<= 50%` |
| Null Strategy League | `decision.passed == true`の場合だけ本番採用 |
| Conformal | 過去データだけで4/7以上の包含率を再校正 |
| Portfolio constraints | 5口、数字使用上限4、口間重複上限4、選出後の数字置換禁止 |

高一致率ゲートに不合格の場合は、財務ゲートへ進まずモデルを置換しません。Null League不合格時も既存の本番予測・履歴・封印証跡を保持します。

## 出力構成

正規レイアウト:

```text
outputs/
├─ production/   最新予測、累積履歴、実運用結果、公開レポート
├─ evidence/     Nested判定、採用拒否、SHA-256封印証跡
├─ state/        Full / Recent / Super Recentの最新Resume state
└─ diagnostics/  Holdout、Role Ensemble、Generation 4のコンパクト診断
```

移行中は旧パスも維持します。

```text
outputs/evolution_best_prediction.csv
outputs/evolution_prediction_history.csv
outputs/evolution_prediction_history_result.txt
outputs/holdout/latest_prediction_report.txt
outputs/model_self_evolution/
outputs/recent_era/
outputs/super_recent/
outputs/validation/
```

`LOTO7 Canonical Output Sync`が旧パスから正規レイアウトへ非破壊コピーします。Resumeに必要な旧ファイルは削除しません。大規模CSV、Fold内部データ、再生成可能な詳細診断はGitHub Actions Artifactへ保存します。

手動同期:

```bash
python scripts/migrate_output_layout.py
```

## 実行方法

```text
Actions → LOTO7 Evolution Trainer
Actions → LOTO7 Model Self Evolution
Actions → LOTO7 Recent Era Self Evolution
Actions → LOTO7 Nested Walk Forward Validation
Actions → LOTO7 Generation 4 Production
Actions → LOTO7 Canonical Output Sync
```

## テスト

全テスト:

```bash
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py' -v
```

主要な移行・精度テスト:

```bash
python -m unittest \
  tests.test_evaluation_unification \
  tests.test_nested_promotion_gates \
  tests.test_package_layout_and_hit_metrics \
  tests.test_output_layout_compatibility \
  tests.test_workflow_resume_compatibility -v
```

構造確認:

```bash
python scripts/check_repository_architecture.py --report-only
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
