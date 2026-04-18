# Skill: DOC-WRITER

Purpose: Write documentation only.

## Rules

1. Primary target is `expected_options()` in `.cxx`.
2. Header comments are brief and only added if local style supports them.
3. Read at least 3 neighboring headers before modifying a header.
4. For new models, update `doc/content/modules/` or create a suitable new page.
5. Do not move detailed formulation text into headers.
6. For Python docstrings, use Google style with `Args:`, `Returns:`, and `Raises:` when applicable.

## Workflow

1. Read the `.cxx`.
2. Fill in missing or weak `.doc()` strings.
3. Infer header style from neighbors and add only minimal header docs if warranted.
4. Update narrative docs for new models.
5. Run [DOCS-VERIFY](../workflows/docs-verify.md).
