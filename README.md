# NEW_LOTO7

`NEW_LOTO7` は、LOTO7の候補モデル生成、独立検証、5口ポートフォリオ選定、実運用履歴、封印証跡を一つのGitHub Actionsパイプラインで管理するリポジトリです。

> LOTO7は強いランダム性を持ちます。本リポジトリの検証結果は、将来の当せん・利益・回収率を保証しません。

## 現行アーキテクチャ

本番予測の唯一の所有者は **LOTO7 Generation 4 Production** です。

| 段階 | Workflow | 役割 |
|---|---|---|
| 全期間学習 | `LOTO7 Evolution Trainer` | 全期間モデル、holdout、役割戦略、診断artifact |
| 独立自己進化 | `LOTO7 Model Self Evolution` | 全期間候補とresume state |
| 直近候補 | `LOTO7 Recent Era Self Evolution` | Recent / Super Recent候補と安全ガード |
| sealed検証 | `LOTO7 Nested Walk Forward Validation` | Nested fold検証とモデル昇格 |
| 本番予測 | `LOTO7 Generation 4 Production` | 5口、履歴、Null League、e-process、SHA-256封印 |
| 回帰確認 | `LOTO7 Validation Tests` | 出力整合性、未来リーク、統計ゲート |
| 構造監査 | `Repository Structure Audit` | 所有権、不要ファイル、生成物保持の監査 |

構造ポリシー:

```text
config/repository_layout.json
docs/architecture/REPOSITORY_LAYOUT.md
docs/architecture/WORKFLOW_OWNERSHIP.md
docs/architecture/OUTPUT_RETENTION.md
```

## 採用ゲート

候補が高い通常ROIを示しても、以下のいずれかに不合格なら本番採用しません。

| ゲート | 現行条件 |
|---|---|
| Independent Holdout | 学習・選定・最終holdoutを時系列分離 |
| Nested Walk-Forward | foldごとに評価年をsealed化 |
| Nested合計ROI | 候補ROI `>= 8.0%` かつ基準モデル差 `>= 0.0pt` |
| Robust ROI | 最大払戻除外、期間中央値、bootstrap下方値、集中度を確認 |
| Null Strategy League | `decision.passed == true` の場合だけ本番予測を採用 |
| Conformal | 過去データのみで、4/7以上の包含率80%を目標に14〜24数字から再校正 |
| Portfolio constraints | 5口、数字使用上限4、口間重複上限4、選出後の数字置換禁止 |

Null Leagueの不合格・判定欠損時は終了コード2で停止し、既存の本番予測・履歴・封印ファイルを上書きしません。

## 本番出力

```text
outputs/evolution_best_prediction.csv
outputs/evolution_prediction_history.csv
outputs/evolution_prediction_history_result.txt
outputs/holdout/latest_prediction_report.txt
```

封印証跡:

```text
outputs/generation4/latest_sealed_manifest.json
outputs/generation4/sealed_index.json
outputs/generation4/sealed/
```

モデルのresume stateとRecent / Super / validation evidenceはGitへ保持します。ML training frame、Complete AI候補、再生成可能な詳細診断はGitへコミットせず、GitHub Actions artifactとして保持します。

## 実行方法

### 本番予測

```text
Actions → LOTO7 Generation 4 Production → Run workflow
```

モデル進化またはNested検証が成功した場合も自動起動します。固定concurrency groupで古い実行を取り消し、最新のモデル状態を優先します。

### 全期間モデル進化

```text
Actions → LOTO7 Evolution Trainer → Run workflow
```

### Recent / Super候補生成

```text
Actions → LOTO7 Recent Era Self Evolution → Run workflow
```

### Nested検証

```text
Actions → LOTO7 Nested Walk Forward Validation → Run workflow
```

## テスト

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

主要テスト:

```bash
python -m unittest \
  tests.test_prediction_output_consistency \
  tests.test_robust_validation_and_portfolio \
  tests.test_generation4_pipeline \
  tests.test_strict_adoption_gates -v
```

構造確認:

```bash
python scripts/check_repository_architecture.py
python scripts/audit_repository_structure.py \
  --json docs/architecture/repository_structure_audit.json \
  --markdown docs/architecture/repository_structure_audit.md
```

## ディレクトリ

```text
.github/workflows/   Actions orchestration
config/              構造・運用ポリシー
scripts/             CLI、検証、レポート、保守ツール
tests/               回帰・リーク・統計ゲート
docs/architecture/   設計、所有権、保持方針、監査結果
outputs/              production / evidence / resumable state
root *.py             現行workflowで必要な互換レイヤー
```

ルート実装は、現行workflow・import・resume互換性が確認できるものだけを残します。廃止済み世代の独立バックテスト、旧predictor、旧report集約は削除済みです。
