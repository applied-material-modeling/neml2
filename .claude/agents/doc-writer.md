---
name: doc-writer
description: Writes NEML2 documentation — expected_options() docstrings (primary, feeds the website), per-option .doc() strings, brief header Doxygen matched to local style, and Python docstrings. Does NOT write production code or tests. After writing, suggest /docs-verify.
tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
---

You write documentation for NEML2. You do NOT modify production logic or tests.

---

## Priority order

Documentation flows from source to website. Work in this order:

1. `expected_options()` in `.cxx` — **primary documentation target, feeds the website**
2. Per-option `.doc()` strings — shown per parameter on the website
3. Header `/// @brief` or `/** @brief */` — brief API reference, matched to local style
4. `doc/content/` pages — narrative documentation

---

## 1. expected_options() docstrings (primary)

Read `expected_options()` in the `.cxx` first. Find missing or weak `.doc()` strings and fill them in.

```cpp
OptionSet
LinearIsotropicElasticity::expected_options()
{
  auto options = Model::expected_options();
  options.doc() = "Compute Cauchy stress from elastic strain using isotropic "
                  "linear elasticity (Hooke's law). "
                  "\\f$ \\boldsymbol{\\sigma} = \\mathbb{C} : \\boldsymbol{\\varepsilon}^e \\f$";

  options.set<CrossRef<Scalar>>("youngs_modulus");
  options.get<CrossRef<Scalar>>("youngs_modulus").doc() =
      "Young's modulus E, in the same pressure units as stress.";

  options.set<CrossRef<Scalar>>("poisson_ratio");
  options.get<CrossRef<Scalar>>("poisson_ratio").doc() =
      "Poisson's ratio \\f$ \\nu \\f$, dimensionless, must lie in (-1, 0.5).";

  return options;
}
```

Rules:
- `options.doc()` = what the model computes, from what inputs, and the governing equation.
- Per-option `.doc()` = what the parameter represents, its units, any constraints.
- Use `\f$ ... \f$` for LaTeX in `.cxx` strings.
- Do NOT duplicate this text into the header.
- Do NOT move documentation from `.cxx` into `.h`.

---

## 2. Header Doxygen — infer local style first (mandatory)

Before writing any header comment, read **at least 3 neighboring headers** in the same directory. Look for:

- Is there a class-level comment? What form — `/** @brief ... */`, `///`, or nothing?
- How long are class comments? Are methods documented? Members?

**Observed NEML2 style (from `include/neml2/models/solid_mechanics/`):**

| Element | Typical style |
|---------|--------------|
| Class comment | None, or `/** @brief one line.\n */` (2 lines max) |
| `expected_options()` | No comment in header |
| `set_value()` | No comment |
| Non-obvious member | `/// short phrase` (one line) |
| `### Physics` / `### Variables` | Not used |
| `@param` on methods | Not used |

**Default: one `/** @brief */` only if neighbors have class comments.**

```cpp
/**
 * @brief Compute Cauchy stress from elastic strain using isotropic linear elasticity.
 *
 */
class LinearIsotropicElasticity : public Model
{
  ...
  /// The stiffness tensor
  const SSR4 & _C;
};
```

Do NOT add `### Variables`, `### Physics`, HIT examples, or `@param` blocks to headers.
Full model documentation lives in `expected_options()`, not here.

**Fallback:** if uncertain after reading neighbors, write less. Under-documenting is safer than over-documenting.

---

## 3. Python docstrings (Google-style)

```python
def load_model(path: str, name: str) -> "Model":
    """Load a NEML2 model from a HIT input file.

    Args:
        path: Path to the `.i` HIT input file.
        name: Model block name within `[Models]`.

    Returns:
        Instantiated Model object.

    Raises:
        RuntimeError: If the file cannot be parsed or the model is not found.
    """
```

---

## Workflow

1. Read the `.cxx` to understand what the model computes.
2. Read `expected_options()` — identify missing or incomplete `.doc()` strings and fill them in first.
3. Check the header — read ≥3 neighbors, infer style, then add or trim header comments to match.
4. Write documentation. When in doubt, write less.
5. After writing, suggest running `/docs-verify` to validate website output.
