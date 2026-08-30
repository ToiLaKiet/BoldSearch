# data/ — Machine-local runtime data

Everything in this directory is **ignored by git** except this README: the
corpus is too large for git and the derived artifacts are reproducible
per-machine. Clone the repo, then populate this directory from the team
storage or the BTC download packages.

## Layout

Directories fall into four categories; concrete names may change per round, but every directory belongs to exactly one of these:

| Category | Directories today | Content |
| --- | --- | --- |
| Served corpus | `keyframes/`, `map-keyframes/` | Keyframe images and per-video frame maps served by FastAPI at `/keyframes`, `/map-keyframes` |
| Raw downloads | `aic2026-downloads/` | Unextracted BTC download packages, as received |
| Round working dirs | `aic2026-p2/` | One per round (pattern `aic2026-<round>/`): submission ZIP, `results/`, submission CSVs, `picks.json` |
| Derived artifacts | `frames/`, `metadata/`, `vectors/`, `evaluation-artifacts/`, `csv/` | Per-video extraction/embedding outputs and evaluation runs; reproducible per machine |

## Rules

- Pipeline **scripts** must not live in this directory (git cannot see or blame
  code placed here). Retired round pipelines are recoverable from git history.
- Never copy corpus files into the frontend `public/` or `dist/`.
- Add new directories here only with a row in this table and a matching
  entry in [okf/data-layout.md](../okf/data-layout.md).
