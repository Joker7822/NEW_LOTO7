# outputs/

This directory contains four different classes of generated material:

1. **production** — the current five-ticket prediction and live history;
2. **evidence** — SHA-256 sealed manifests and immutable prediction records;
3. **state** — resumable evolution and nested-validation checkpoints;
4. **diagnostics** — reproducible backtests, reports and experimental outputs.

The canonical ownership and retention rules are defined in
`config/repository_layout.json` and `docs/architecture/OUTPUT_RETENTION.md`.
Only `LOTO7 Generation 4 Production` may build committed production outputs.
