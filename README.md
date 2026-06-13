# NEW_LOTO7

LOTO7 のCSV更新、walk-forwardバックテスト、最新予測、進化型モデル探索、ML拡張スタックを実行するリポジトリです。

## 追加済みパイプライン

`loto7_pipeline.py` を追加しました。

主な機能:

- `scrapingloto7.py` を先に実行して `loto7.csv` を最新化
- 未来リークなしの walk-forward バックテスト
- `outputs/resume_state.json` による途中再開
- `outputs/loto7_backtest_result.csv` に検証結果を追記保存
- `outputs/loto7_backtest_summary.csv` にサマリー出力
- `outputs/loto7_latest_prediction.csv` に次回予測5口を出力
- GitHub Actions 実行中に100抽せんごと commit/push
- 購入口数デフォルト5口

## 通常パイプライン手動実行

```bash
python loto7_pipeline.py \
  --run-scraping \
  --csv loto7.csv \
  --output-dir outputs \
  --resume-state outputs/resume_state.json \
  --purchase-count 5 \
  --min-train-draws 60 \
  --max-targets all \
  --push-every 100 \
  --push-final
```

ローカルでGitHubへpushしたくない場合:

```bash
DISABLE_GIT_PUSH=1 python loto7_pipeline.py --run-scraping --purchase-count 5
```

Windows PowerShell:

```powershell
$env:DISABLE_GIT_PUSH="1"
python loto7_pipeline.py --run-scraping --purchase-count 5
```

## GitHub Actions

`.github/workflows/loto7_pipeline.yml` を追加済みです。

- 手動実行: Actions > LOTO7 Pipeline > Run workflow
- 定期実行: 毎週金曜 20:15 JST
- タイムアウト対策: `resume_state.json` により中断箇所から再開
- 保存対策: 100抽せんごとに途中commit/push

## 進化型モデル探索

`loto7_evolution_trainer.py` を追加しました。

固定モデルではなく、以下のパラメータを世代ごとに変異・交叉し、walk-forwardバックテストで最良モデルを選抜します。

- Full / 直近240 / 直近120 / 直近60 の重み
- 相性ペア重み
- ペア直近性重み
- ペア安定性重み
- トリプル相関重み
- 休眠数字重み
- 奇偶制約
- 合計値制約
- 低高制約
- 連番ペナルティ
- 口同士の重複上限
- 候補数字プールサイズ

実行例:

```bash
python loto7_evolution_trainer.py \
  --csv loto7.csv \
  --output-dir outputs \
  --best-model loto7_best_model.json \
  --generations 20 \
  --population 30 \
  --elite-count 6 \
  --purchase-count 5 \
  --min-train-draws 60 \
  --max-targets 240 \
  --target-stride 1 \
  --push-final
```

軽量テスト:

```bash
DISABLE_GIT_PUSH=1 python loto7_evolution_trainer.py \
  --generations 2 \
  --population 6 \
  --elite-count 2 \
  --max-targets 20 \
  --target-stride 2
```

専用Workflow:

- Actions > LOTO7 Evolution Trainer > Run workflow
- 定期実行: 毎週土曜 03:30 JST
- 世代途中: 5世代ごとにcommit/push
- 出力artifact: `loto7-evolution-outputs`

## ML拡張スタック

以下を追加しました。

- `requirements-ml.txt`
- `loto7_ml_stack.py`
- `.github/workflows/loto7_ml_stack.yml`

実装内容:

- MemoryBank
- MetaClassifier
- LightGBM
- CatBoost
- XGBoost
- Optuna
- SHAP

手動実行:

```bash
pip install -r requirements-ml.txt
python loto7_ml_stack.py \
  --csv loto7.csv \
  --output-dir outputs/ml_stack \
  --min-train 60 \
  --max-targets 240 \
  --candidates-per-draw 80 \
  --label label_4plus \
  --optuna-trials 50
```

GitHub Actions:

- Actions > LOTO7 ML Stack > Run workflow

出力:

|ファイル|内容|
|---|---|
|`outputs/ml_stack/ml_training_frame.csv`|MetaClassifier用教師データ|
|`outputs/ml_stack/ml_model_report.csv`|LightGBM/CatBoost/XGBoost等の評価|
|`outputs/ml_stack/optuna_best_params.json`|Optuna最良パラメータ|
|`outputs/ml_stack/shap_feature_importance.csv`|SHAP特徴量重要度|
|`outputs/ml_stack/loto7_memorybank_mb4.csv`|4個構造MemoryBank|
|`outputs/ml_stack/loto7_memorybank_mb5.csv`|5個構造MemoryBank|
|`outputs/ml_stack/loto7_memorybank_mb6.csv`|6個構造MemoryBank|
|`outputs/ml_stack/ml_stack_status.json`|実行結果・エラー情報|

## 出力ファイル

|ファイル|内容|
|---|---|
|`outputs/resume_state.json`|通常バックテストの最後に完了した抽せん回|
|`outputs/loto7_backtest_result.csv`|通常バックテストの各回・各5口の一致数と等級|
|`outputs/loto7_backtest_summary.csv`|通常バックテストの等級別件数・最大一致数|
|`outputs/loto7_latest_prediction.csv`|通常パイプラインの次回5口|
|`loto7_best_model.json`|進化型探索で最も良かったモデル|
|`outputs/evolution_history.csv`|進化型探索の全候補評価履歴|
|`outputs/evolution_best_summary.csv`|各世代の上位モデル|
|`outputs/evolution_best_prediction.csv`|最良モデルによる次回5口|

## 注意

宝くじの抽せんはランダム性が高く、予測は的中を保証しません。バックテストは未来データを使わない検証形式に固定しています。


## 2026-06-10 直接改善内容

`loto7_advanced_optimizer.py` に以下をコードレベルで組み込み済みです。

- 直近240回モデル
- 直近120回モデル
- 直近60回モデル
- 直近モデルEnsemble候補生成
- 相性ペア安定性スコア
- 相性ペア直近性スコア
- 奇偶・合計値・低高バランス制約スコア
- MetaClassifier用特徴量の拡張
- `loto7_memorybank_4plus.csv` 出力
- `loto7_memorybank_6hit.csv` 出力

実行例:

```bash
LOTO7_DISABLE_OPTIMIZE=1 \
LOTO7_BACKTEST_MONTE_CARLO=100 \
LOTO7_BACKTEST_MCTS=50 \
LOTO7_RECENT_POOL_SIZE=17 \
python loto7_chunked_backtest.py \
  --csv loto7.csv \
  --min-train 60 \
  --tickets 5 \
  --pool-size 17 \
  --chunk-size 10 \
  --max-chunks 1
```

軽量動作確認:

```bash
LOTO7_DISABLE_OPTIMIZE=1 \
LOTO7_BACKTEST_MONTE_CARLO=0 \
LOTO7_BACKTEST_MCTS=0 \
LOTO7_RECENT_POOL_SIZE=12 \
python loto7_chunked_backtest.py \
  --csv loto7.csv \
  --min-train 60 \
  --tickets 5 \
  --pool-size 12 \
  --chunk-size 2 \
  --max-chunks 1
```
