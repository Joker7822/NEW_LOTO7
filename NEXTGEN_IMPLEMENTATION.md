# NEW_LOTO7 NextGen 実装メモ

## 実装済み

- Cycle Attention Transformer
  - `loto7_nextgen_models.py`
  - `cycle_number_scores()` / `cycle_attention_score()`
  - PyTorch が無い環境でも動く Time2Vec + cycle attention fallback 実装。

- Diffusion
  - `diffusion_candidates()`
  - 37数字上の離散デノイジング生成。

- Multi-Agent PPO
  - `multi_agent_ppo_candidates()`
  - HOT / COLD / CYCLE / BALANCED の4方針を PPO 風 clipped update で候補生成。

- 6本一致専用 MetaClassifier
  - `train_meta6_classifier()`
  - `meta6_score()`
  - `loto7_meta6_classifier.json` に保存。
  - 6本一致が少ない場合は 5本一致を soft positive として使用。

- SHAP特徴量選択
  - `shap_feature_selection()`
  - `selected_feature_score()`
  - `loto7_shap_feature_selection.json` に保存。
  - SHAP が無い環境でも相関重要度 fallback で動作。

## 統合先

- `loto7_advanced_optimizer.py`
  - `AdvancedWeights` に `cycle`, `diffusion`, `ppo`, `meta6`, `shap` を追加。
  - `advanced_predict()` で Diffusion / Multi-Agent PPO 候補を追加。
  - `score_ticket()` に cycle / meta6 / shap スコアを統合。
  - `advanced_backtest()` 完了時に Meta6 / SHAP JSON を自動生成。

- `loto7_logic_predictor.py`
  - 出力に `cycle`, `meta6`, `shap` を追加表示。

## 主な環境変数

```bash
LOTO7_DISABLE_NEXTGEN=1          # NextGen候補生成を無効化
LOTO7_DIFFUSION_CANDIDATES=1200  # Diffusion候補数
LOTO7_PPO_CANDIDATES=1000        # Multi-Agent PPO候補数
LOTO7_DEBUG_NEXTGEN=1            # NextGen例外を表示
```

## 軽量テスト例

```bash
LOTO7_DISABLE_OPTIMIZE=1 \
LOTO7_DIFFUSION_CANDIDATES=20 \
LOTO7_PPO_CANDIDATES=20 \
LOTO7_MCTS_ITERATIONS=10 \
python loto7_logic_predictor.py \
  --csv loto7.csv \
  --tickets 3 \
  --pool-size 10 \
  --monte-carlo 10 \
  --disable-optimize \
  --no-save
```

## 注意

ロト7は独立抽せんのため、的中保証はできません。今回の実装は「過去データの walk-forward 検証で比較可能な候補生成器」を増やす目的です。
