# Repository Restructure Report

Generated: `2026-07-17 Asia/Tokyo`

## Applied

- `LOTO7 Generation 4 Production`を本番予測の唯一の所有workflowに統一
- 旧`LOTO7 Generation 4 Prediction`（Dual workflow）を削除
- Evolution TrainerとModel Self Evolutionから本番予測CSV・最新レポートの直接生成を撤去
- Quick Finishの試験予測を`outputs/diagnostics/quick_finish/`へ隔離
- 非標準の`concurrency.queue`設定を撤去
- 最新モデル状態を優先するGeneration 4の固定concurrency groupを設定
- workflow所有権、出力保持方針、段階的パッケージ移行方針を文書化
- 構造ガードをValidation TestsとRepository Structure Auditへ接続
- 一時実行制御ファイルをGit管理対象から除外

## Preserved intentionally

- ルート直下のPython実装は、既存workflow・resume state・import互換性を守るためPhase 1では移動していません。
- 学習状態、Recent/Super候補、Nested validation、封印証跡は削除していません。
- 大容量診断出力は、参照関係を切り替えるまでは維持します。

## Canonical ownership

| 種別 | 所有者 |
|---|---|
| モデル進化・holdout・診断 | Evolution系workflow |
| Recent/Super候補生成 | Recent Era Self Evolution |
| 候補のsealed昇格判定 | Nested Walk Forward Validation |
| 本番5口・履歴・e-process・SHA封印 | Generation 4 Production |
| Quick Finish試験予測 | `outputs/diagnostics/quick_finish/` |

## Next phase

Phase 2では、ルートPythonを`src/loto7/`へ互換wrapper付きで段階移行し、再生成可能な大型診断ファイルをActions artifactsへ移します。
