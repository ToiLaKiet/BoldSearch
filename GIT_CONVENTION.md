# Git Convention

## Branch naming

Use lowercase slugs:

```text
<type>/<short-scope>
```

Types:

- `feature/` — user-visible capability or new pipeline component.
- `fix/` — bug fix.
- `docs/` — documentation-only change.
- `refactor/` — behavior-preserving code movement or cleanup.
- `test/` — test-only change.
- `experiment/` — benchmark, notebook, or research spike not intended as production path yet.
- `chore/` — tooling, dependency, config, or repo maintenance.

Examples:

```text
feature/retrieval-service
experiment/fgclip-qdrant-benchmark
fix/search-score-ties
docs/pipeline-init
```

## Commit messages

Use Conventional Commits:

```text
<type>(<scope>): <imperative summary>

<body with why, verification, and risks when useful>
```

Allowed commit types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `chore`, `revert`.

Recommended scopes:

- `backend`
- `frontend`
- `retrieval`
- `embedding`
- `vector-store`
- `benchmark`
- `docs`
- `git`
- `security`

Examples:

```text
feat(retrieval): add deterministic shot grouping

docs(architecture): initialize HCM AI Challenge pipeline conventions

test(vector-store): add shared provider contract fixtures
```

