# NEW_LOTO7 Repository Layout

Effective: 2026-07-28  
Policy: `config/repository_layout.json` schema 4

## Canonical layout

```text
.github/workflows/        GitHub Actions orchestration only
config/                   Architecture, workflow and output policy
docs/architecture/        Decisions and generated structure reports
outputs/production/       Public latest prediction and cumulative history
outputs/evidence/         Immutable sealed and validation evidence
outputs/state/            Compact resumable training state
outputs/diagnostics/      Latest compact evaluation summaries
scripts/                  Thin CLI and compatibility entry points
src/loto7/                Canonical reusable implementation
tests/                    Unit, integration and compatibility regression tests
root *.py                 Frozen compatibility allowlist; no new modules
```

## Package boundaries

```text
src/loto7/
├─ evaluation/            Shared hit, payout and robustness evaluation
├─ evolution/             Candidate search and Generation 5 promotion logic
├─ repository/            Repository policy loading and fail-closed audit
├─ validation/            Independent and Nested promotion gates
└─ paths.py               Canonical and legacy output bindings
```

Reusable logic belongs in `src/loto7`. A migrated file under `scripts/` or the
repository root may remain only as an import-compatible CLI wrapper. New root
Python files are prohibited by the architecture audit.

## Workflow ownership

The complete workflow inventory is `config/workflow_registry.json`.

| Responsibility | Owner |
|---|---|
| Production prediction publication | `LOTO7 Production Prediction Publisher` |
| Automatic Best Model promotion | `LOTO7 Generation 5 Precision Evolution` |
| Generation 4 candidate diagnostics | `LOTO7 Generation 4 Evaluation` |
| Canonical output mirroring | `LOTO7 Canonical Output Sync` |
| Repository architecture enforcement | `Repository Structure Audit` |

The production publisher is the only committed writer of the four legacy public
prediction outputs. Generation 5 owns automatic model promotion. Legacy training
workflows may create candidates and state, but direct application is fail-closed
unless an explicit supported manual override is used.

## Output classes

```text
production   User-facing latest prediction, history and compact report
evidence     Sealed manifests, adoption/rejection and Nested evidence
state        Resume data required to continue active training
diagnostics Latest compact Holdout, Role, Generation 4 and Generation 5 summaries
```

Large reproducible detail, candidate populations, Fold internals and full Null
simulation rows are uploaded as GitHub Actions artifacts rather than retained in
permanent repository history.

## Compatibility migration

1. New code reads canonical paths first.
2. Legacy paths remain fallback aliases while active workflows still resume from them.
3. State is copied, never destructively moved.
4. Sealed evidence is immutable and never pruned by layout migration.
5. A migrated CLI keeps its previous command and import surface through a thin wrapper.
6. Every migration requires a dedicated compatibility regression test.

## Enforcement

```bash
python -m pip install -e .
loto7-repository-audit \
  --json /tmp/loto7-architecture.json \
  --markdown /tmp/loto7-architecture.md
python -m unittest tests.test_repository_architecture_v4 -v
```

The audit fails on unregistered root Python modules, unexpected top-level
directories, duplicate Workflow names, missing Workflow owners, production
writer duplication, forbidden one-time Workflows and non-thin registered wrappers.
