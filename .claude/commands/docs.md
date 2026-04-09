Add or improve documentation for NEML2 objects, classes, or modules.

`$ARGUMENTS` should be a file path, class name, or short description of what to document.
Read the target file(s) before writing anything.

---

## Priority

Work in this order:

1. `expected_options()` in `.cxx` — primary documentation source, feeds the website
2. Per-option `.doc()` strings — one per parameter
3. Header `/// @brief` or `/** @brief */` — one line, matched to local style
4. `doc/content/` pages — narrative documentation

---

## expected_options() pattern

```cpp
options.doc() = "What this model computes and from what inputs.";

options.set<CrossRef<Scalar>>("param");
options.get<CrossRef<Scalar>>("param").doc() =
    "Description of param: units, valid range, physical meaning.";
```

Use `\f$ ... \f$` for LaTeX in `.cxx` docstrings.

---

## Header Doxygen

Read ≥3 neighboring headers first to infer local style. Match it.
Default: one `/** @brief one line. */` if neighbors have class comments; nothing otherwise.
Do NOT add `### Variables`, `@param`, or HIT examples to headers.

---

## Python docstrings

Google-style. Every public function needs `Args:`, `Returns:`, `Raises:` (if applicable).

---

After writing, run `/docs-verify` to validate the output.
