# NEW_LOTO7

LOTO7のデータ更新、モデル進化、独立検証、5口ポートフォリオ最適化、実運用履歴、SHA-256封印予測を管理するリポジトリです。

> 宝くじはランダム性が高く、本リポジトリの予測・バックテスト・自己進化結果は、将来の当せんや利益を保証しません。

## 現在の本番構成

```text
データ更新・モデル進化
        ↓
Recent / Super候補生成
        ↓
Nested Walk-Forward・Robust昇格判定
        ↓
Generation 4 Production
        ↓
Null Strategy League / PBO
Conformal数字プール
Change-Point検出
動的モデル配分
DPP + Hypergraph 5口最適化
        ↓
予測履歴・Champion / Challenger・e-process
        ↓
SHA-256封印・成果物保存
```

本番予測の所有workflowは次の1本です。

```text
.github/workflows/loto7_generation4_run.yml
name: LOTO7 Generation 4 Production
```

Evolution系workflowはモデル・状態・検証結果を生成しますが、本番5口を直接更新しません。

## Workflow所有権

| Workflow | 役割 |
|---|---|
| `LOTO7 Evolution Trainer` | CSV更新、全期間モデル進化、holdout、role・ML診断 |
| `LOTO7 Model Self Evolution` | 全期間モデルの独立自己進化と安全ガード |
| `LOTO7 Recent Era Self Evolution` | Recent / Super Recent候補生成 |
| `LOTO7 Nested Walk Forward Validation` | sealed fold検証と候補昇格 |
| `LOTO7 Generation 4 Production` | 本番5口、履歴、e-process、封印予測 |
| `LOTO7 Quick Finish Check` | 診断専用。試験予測は本番出力と分離 |
| `LOTO7 Validation Tests` | 回帰テスト、未来リーク、構造ガード |
| `Repository Structure Audit` | リポジトリ構造と所有権の継続監査 |

詳細は以下を参照してください。

```text
docs/architecture/REPOSITORY_LAYOUT.md
docs/architecture/WORKFLOW_OWNERSHIP.md
docs/architecture/OUTPUT_RETENTION.md
config/repository_layout.json
```

## Generation 4の主要機能

| 機能 | 内容 |
|---|---|
| Nested Walk-Forward | 学習・選定・評価期間を分離したsealed検証 |
| Robust ROI | 最大払戻除外、期間中央値、下方リスクを評価 |
| Null Strategy League | ランダム・頻出・休眠・直近・バランス戦略と比較 |
| PBO | 多数試行によるバックテスト過学習を診断 |
| Conformal Pool | 過去データだけで候補数字集合を校正 |
| Change-Point | 直近分布と基準期間の変化を検出 |
| Dynamic Weighting | Full / Recent / Super / Regime配分を動的調整 |
| DPP | 似過ぎた候補を避け、5口の多様性を評価 |
| Hypergraph | ペア・トリプル被覆を5口全体で最適化 |
| Champion / Challenger | 旧方式と第4世代をシャドー比較 |
| e-process | 実抽せんを逐次確認しながら昇格証拠を蓄積 |
| SHA-256 seal | 抽せん前予測、モデル、データ、実行情報を固定 |

## 本番出力

```text
outputs/evolution_best_prediction.csv
outputs/evolution_prediction_history.csv
outputs/evolution_prediction_history_result.txt
outputs/holdout/latest_prediction_report.txt
```

封印証跡は以下へ保存します。

```text
outputs/generation4/latest_sealed_manifest.json
outputs/generation4/sealed_index.json
outputs/generation4/sealed/
```

Quick Finishの試験予測は本番出力を上書きしません。

```text
outputs/diagnostics/quick_finish/evolution_best_prediction.csv
```

## 実行方法

### 本番第4世代予測

```text
Actions
→ LOTO7 Generation 4 Production
→ Run workflow
→ null_simulations: 180
```

モデル進化・Nested検証の成功後にも自動起動します。新しいモデル状態が到着した場合は、固定concurrency groupにより古い予測実行を置き換え、最新状態を優先します。

### 全期間モデル進化

```text
Actions
→ LOTO7 Evolution Trainer
→ Run workflow
```

### Recent / Super候補生成

```text
Actions
→ LOTO7 Recent Era Self Evolution
→ Run workflow
```

### Nested検証

```text
Actions
→ LOTO7 Nested Walk Forward Validation
→ Run workflow
```

### 軽量診断

```text
Actions
→ LOTO7 Quick Finish Check
→ Run workflow
```

## テスト

```bash
python -m unittest \
  tests.test_prediction_output_consistency \
  tests.test_robust_validation_and_portfolio \
  tests.test_generation4_pipeline -v
```

構造所有権の確認:

```bash
python scripts/check_repository_architecture.py
```

構造監査レポートの生成:

```bash
python scripts/audit_repository_structure.py \
  --json docs/architecture/repository_structure_audit.json \
  --markdown docs/architecture/repository_structure_audit.md
```

## ディレクトリ構成

```text
.github/workflows/   Actions orchestration
config/              構造・運用ポリシー
scripts/             CLI、検証、レポート、保守ツール
tests/               回帰・リーク・構造テスト
docs/architecture/   設計、所有権、監査結果
outputs/              production / evidence / state / diagnostics
root *.py             既存import互換レイヤー
```

ルートPythonは既存workflowとresume stateの互換性を守るため、Phase 1では移動していません。Phase 2で互換wrapperを維持しながら`src/loto7/`へ段階移行します。

## 主要ファイル

| ファイル | 内容 |
|---|---|
| `loto7.csv` | 抽せんデータ |
| `loto7_best_model.json` | 全期間採用モデル |
| `loto7_evolution_trainer.py` | 進化・候補生成の中核 |
| `holdout_evaluator.py` | 全期間holdout検証 |
| `scripts/build_generation4_prediction.py` | 第4世代5口生成 |
| `scripts/generation4_core.py` | Conformal、DPP、Hypergraph、動的配分 |
| `scripts/null_strategy_league.py` | Null League・PBO |
| `scripts/update_generation4_shadow_history.py` | Champion / Challenger・e-process |
| `scripts/seal_generation4_prediction.py` | SHA-256封印 |
| `scripts/check_repository_architecture.py` | 構造所有権ガード |

## 出力保持方針

- 本番予測・累積履歴・封印証跡はGitへ保持
- resume可能なモデル状態は、圧縮方式導入までは保持
- 再生成可能な大型診断は、参照関係を切り替えた後にActions artifactsへ移行
- 一時実行制御ファイル、fold内部データ、ローカルキャッシュは追跡しない
