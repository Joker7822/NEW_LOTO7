# LOTO7 Statistical Hardening v2

適用日: 2026-07-31

## 目的

モデルをさらに複雑化する前に、モデルの優位性を同一条件のNull戦略と比較し、偶然の差・探索回数・払戻集中を分離して評価します。

## 指標スキーマ

`metric_schema_version = loto7-metrics-2026.07.31-v2`

- `payout_roi_percent`: 払戻額 ÷ 購入額 × 100
- `profit_roi_percent`: （払戻額 − 購入額）÷ 購入額 × 100
- `roi_percent`: 互換用。v2では収支ROIを意味し、廃止予定

## 対応あり時系列推論

候補と基準モデルを同一抽せん回で比較し、Moving Block Bootstrapで次の95%信頼区間を算出します。

- 平均最大本数字一致差
- 抽せん回単位4個以上一致率差

昇格には両方の信頼区間下限が0を上回ることを要求します。

## Hit-first Null League

- 払戻金額を学習・Null主得点から除外
- 当選数字列を固定済み予測ポートフォリオに対して置換
- 探索幅ごとの最大値を採用し、候補選択による多重比較を補正
- 150、500、1,000系列で段階評価
- Null超過率のWilson信頼区間上限が10%以下の場合のみ合格

PBOと最大1回払戻依存率は、独立した財務安全ゲートとして維持します。

## 数字順位評価

- Top-7 / Top-14 / Top-18 本数字Recall
- 当選数字の平均順位
- Portfolio Inclusion AUC
- Brier Score
- Calibration Error

## アブレーション

次の3条件を同一期間で評価します。

1. 学習モデル + 現行ポートフォリオ最適化
2. ランダム数字 + 分散ポートフォリオ
3. 学習モデル + ランダム化ポートフォリオ

これにより、数字順位付け能力と5口分散能力を分離します。

## 再開・証跡

- Generation 5終了時にdataset SHA、baseline model SHA、完了世代、状態をcheckpointへ保存
- 昇格処理前に統計強化を必ず実行
- エラー時は`hardening_error.json`を保存し、fail-closedで昇格を拒否
- 終了時に`run_status.json`を保存

## リポジトリ整理

以下はGit追跡から削除し、再生成時も無視します。

- hit-first移行前の履歴バックアップ3件
- Role Ensemble全口詳細CSV
- 一回限りの移行スクリプトとWorkflow

`outputs/holdout/holdout_result.csv`は、Holdout Summary IntegrityとEvaluator Full Data PR Checkの正本入力として使用されているため保持します。

現行Resume state、現行履歴、本番予測、累積予測履歴、Holdout正本入力、封印証跡は保持します。
