LOTO7 TXT Reports Folder
=========================

このフォルダは、各種 .txt / .md レポートを一括確認するための専用フォルダです。

生成される主なファイル:

- 00_index.txt
  集約されたレポート一覧

- all_reports.txt
  主要TXT/MDレポートを1ファイルに連結した一覧

- 10_pr_summary.txt
  自己進化PR確認用の要約

- 20_self_evolution_report.txt
  自己進化AIの診断・改善案レポート

- 30_latest_prediction_report.txt
  最新予測レポート

- 40_model_selection_report.txt
  モデル選定レポート

- 50_holdout_report.txt
  holdoutバックテストレポート

生成方法:

bash scripts/collect_txt_reports.sh outputs/txt_reports

PR作成・更新時は `.github/workflows/loto7_txt_reports.yml` が自動でこのフォルダを生成し、artifactとして保存します。
