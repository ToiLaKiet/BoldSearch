# data/ — Machine-local runtime data

Everything in this directory is **ignored by git** except this README: the
corpus is too large for git and the derived artifacts are reproducible
per-machine. Clone the repo, then populate this directory from the team
storage or the BTC download packages.

## Layout

| Directory | Content |
| --- | --- |
| `keyframes/` | Keyframe images served by FastAPI at `/keyframes` |
| `map-keyframes/` | Per-video frame-map CSVs served at `/map-keyframes` |
| `aic2026-downloads/` | Raw BTC download packages (unextracted) |
| `aic2026-p2/` | Round-2 working dir: submission ZIP, `results/`, `submission/` CSVs, `picks.json` |
| `frames/`, `metadata/`, `vectors/` | Per-video derived artifacts (extraction, embedding runs) |
| `evaluation-artifacts/` | Offline evaluation run outputs |
| `csv/` | Scratch (currently empty) |

## Rules

- Pipeline **scripts** must not live in this directory (git cannot see or blame
  code placed here). The retired P2 pipeline is recoverable from git history
  (commit `f98aa9f`).
- Never copy corpus files into the frontend `public/` or `dist/`.
- Add new top-level directories here only with a row in this table and a matching
  entry in [okf/data-layout.md](../okf/data-layout.md).
