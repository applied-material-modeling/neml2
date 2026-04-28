# Workflow: DOCS

Purpose: Add or improve documentation for a file, class, or module.

## Priority Order

1. `expected_options()` in `.cxx`
2. per-option `.doc()` strings
3. brief header Doxygen matched to local style
4. `doc/content/` narrative pages

## Procedure

1. Read the target `.cxx` and locate `expected_options()`.
2. Improve `options.doc()` so it states what the object computes, from which inputs, and its governing equation.
3. Improve per-option `.doc()` strings with units, meaning, and constraints.
4. Use `\f$ ... \f$` for LaTeX in `.cxx` docstrings.
5. Read at least 3 neighboring headers before touching a header comment.
6. If neighbors use class comments, match their local form, usually a brief `@brief`.
7. Do not add `### Variables`, `@param`, or HIT examples to headers.
8. For new models or constitutive laws, update the appropriate `doc/content/modules/*.md` page.
9. Include governing equations, variable definitions, parameter descriptions, units, and a minimal HIT example via `@list-input`.
10. After editing documentation, run [DOCS-VERIFY](./docs-verify.md).
