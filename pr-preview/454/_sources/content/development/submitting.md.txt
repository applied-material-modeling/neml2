# Submitting a pull request

1. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
   the repository on GitHub, then branch off `main` in your fork.
2. Make your change, with tests where applicable.
3. Run `pre-commit run --all-files` and `pytest -v tests/` locally.
4. Push and open a PR against the upstream `main`. CI will run the
   same lint + test + typecheck matrix you ran locally, plus the
   wheel-build / torch-compat matrix defined in `.github/workflows/`.
5. Address review feedback by pushing new commits on top. Once a
   reviewer is in the PR, please don't force-push — it makes the
   round-over-round diff hard to follow. The maintainer will squash
   or rebase at merge time.

## Use of generative AI

Contributions written with help from large language models or other
generative-AI tools are welcome. Two ground rules:

1. **Disclose.** Make it clear which parts of the change came from an
   AI tool. The two accepted forms are:

   - A `Co-authored-by:` trailer on the relevant commit(s) naming the
     tool — e.g. `Co-authored-by: Claude <noreply@anthropic.com>`.
   - A short note in the PR description identifying which files or
     sections were AI-assisted.

   Either is fine; pick the one that fits the granularity of the help.
   Boilerplate-level autocomplete (a few-token IDE suggestion) doesn't
   need to be called out.

2. **You own the result.** Whoever opens the PR — not the AI — is
   responsible for the merged change: that it is correct, that it
   follows project conventions, that the tests cover it, and that the
   license and provenance of any embedded snippets are clean. Treat
   AI output the way you'd treat any other draft you didn't write:
   read it carefully, run the tests, and rewrite anything you'd want
   changed in code review.

This policy is intentionally permissive. If the volume or quality of
AI-assisted PRs changes the calculus, we'll revisit.
