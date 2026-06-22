#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-outputs/txt_reports}"
mkdir -p "$OUT_DIR"

copy_report() {
  local src="$1"
  local dst="$2"
  if [ -f "$src" ]; then
    cp "$src" "$OUT_DIR/$dst"
    echo "$dst <= $src" >> "$OUT_DIR/.index.tmp"
  fi
}

rm -f "$OUT_DIR/.index.tmp"

# PR/self-evolution summary
{
  echo "LOTO7 Self-Evolution PR Summary"
  echo "================================"
  echo ""
  echo "Generated at UTC: $(date -u --iso-8601=seconds)"
  echo ""
  echo "Key metrics now include:"
  echo "- profit-based ROI: profit / total cost"
  echo "- payout ROI: total payout / total cost"
  echo "- ticket hit rate: winning tickets / total tickets"
  echo "- draw hit rate: winning draws / target draws"
  echo ""
  echo "Check these files first:"
  echo "1. outputs/txt_reports/00_index.txt"
  echo "2. outputs/txt_reports/all_reports.txt"
  echo "3. outputs/txt_reports/20_self_evolution_report.txt"
  echo "4. outputs/txt_reports/30_latest_prediction_report.txt"
  echo "5. outputs/txt_reports/50_holdout_report.txt"
  echo ""
  if [ -f outputs/self_evolution/adoption_decision.json ]; then
    echo "Adoption decision JSON: outputs/self_evolution/adoption_decision.json"
  fi
  if [ -f outputs/self_evolution/proposal.json ]; then
    echo "Proposal JSON: outputs/self_evolution/proposal.json"
  fi
} > outputs/self_evolution/pr_summary.txt

copy_report outputs/self_evolution/pr_summary.txt 10_pr_summary.txt
copy_report outputs/self_evolution/comparison_report.txt 20_self_evolution_report.txt
copy_report outputs/holdout/latest_prediction_report.txt 30_latest_prediction_report.txt
copy_report outputs/holdout/model_selection_report.txt 40_model_selection_report.txt
copy_report outputs/holdout/holdout_report.txt 50_holdout_report.txt
copy_report outputs/loto7_progress_summary.md 60_progress_summary.txt

# Collect any other txt/md reports under outputs.
find outputs -type f \( -name '*.txt' -o -name '*.md' \) | sort | while read -r f; do
  case "$f" in
    outputs/txt_reports/*) continue ;;
  esac
  safe_name="90_$(echo "$f" | sed 's#[^A-Za-z0-9_.-]#_#g')"
  if [ ! -f "$OUT_DIR/$safe_name" ]; then
    cp "$f" "$OUT_DIR/$safe_name"
    echo "$safe_name <= $f" >> "$OUT_DIR/.index.tmp"
  fi
done

{
  echo "LOTO7 TXT Reports Index"
  echo "========================"
  echo ""
  echo "Generated at UTC: $(date -u --iso-8601=seconds)"
  echo ""
  echo "Metric definitions:"
  echo "- profit-based ROI = profit / total cost"
  echo "- payout ROI = total payout / total cost"
  echo "- ticket hit rate = winning tickets / total tickets"
  echo "- draw hit rate = winning draws / target draws"
  echo ""
  echo "Files:"
  if [ -f "$OUT_DIR/.index.tmp" ]; then
    sort "$OUT_DIR/.index.tmp"
  fi
} > "$OUT_DIR/00_index.txt"

{
  echo "LOTO7 ALL TXT REPORTS"
  echo "====================="
  echo ""
  echo "Generated at UTC: $(date -u --iso-8601=seconds)"
  echo ""
  for f in "$OUT_DIR"/*.txt; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    [ "$base" = "all_reports.txt" ] && continue
    echo "################################################################################"
    echo "# $base"
    echo "################################################################################"
    cat "$f"
    echo ""
  done
} > "$OUT_DIR/all_reports.txt"

rm -f "$OUT_DIR/.index.tmp"
echo "[TXT-REPORTS] generated in $OUT_DIR"
