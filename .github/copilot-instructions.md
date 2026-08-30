# BoldSearch Copilot Instructions

Repository-wide guidance for GitHub Copilot, including commit message generation.

## Commit messages

Use Conventional Commits:

```text
<type>(<scope>): <imperative summary>
```

- Allowed commit types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `chore`, `revert`.
- Recommended scopes: `backend`, `frontend`, `retrieval`, `embedding`, `vector-store`, `benchmark`, `docs`, `git`, `security`.
- Write the subject as an imperative summary (e.g. `add`, `fix`, `move` — not `added`, `fixes`).
- Add a body only when useful; explain why, verification performed, and risks.
- Generate messages from the actual diff. Do not claim changes or verification that did not happen.
- Do not add AI attribution, signatures, or `Co-authored-by` lines.

## Branch naming

Use lowercase slugs:

```text
<type>/<short-scope>
```

- Allowed branch prefixes: `feature`, `fix`, `docs`, `refactor`, `test`, `experiment`, `chore`.
- Branch prefix `feature/` maps to commit type `feat`.
- Never commit directly on `main` or `master`; create a branch first.

## Source of truth

The complete workflow lives in [GIT_CONVENTION.md](../GIT_CONVENTION.md). This file summarizes it for Copilot; when rules conflict, `GIT_CONVENTION.md` wins.
