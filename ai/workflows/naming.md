# Workflow: NAMING

Purpose: Audit naming consistency without auto-fixing.

## Context
If no file list is provided, audit files changed relative to `main` using `git diff --name-only main`.
Ignore `contrib/`.

## Checks

1. File name vs primary class name in `.h`
2. `register_NEML2_object(...)` consistency in `.cxx`
3. HIT `type =` names in `.i` files vs registered class names
4. private/protected member variable leading underscore convention

## Output Format

`FILE:LINE  [rule]  found: actual  expected: expected`

Report only concrete mismatches. Skip uncertain cases silently. Print a total violation count.
