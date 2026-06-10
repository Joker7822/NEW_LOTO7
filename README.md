# NEW_LOTO7

LOTO7 のCSV更新、walk-forwardバックテスト、最新予測を実行するリポジトリです。

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

## 手動実行

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

## 出力ファイル

|ファイル|内容|
|---|---|
|`outputs/resume_state.json`|最後に完了した抽せん回|
|`outputs/loto7_backtest_result.csv`|各回・各5口の一致数と等級|
|`outputs/loto7_backtest_summary.csv`|等級別件数・最大一致数|
|`outputs/loto7_latest_prediction.csv`|最新CSVをもとにした次回5口|

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
