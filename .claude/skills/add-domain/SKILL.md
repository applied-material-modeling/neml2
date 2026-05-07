---
name: add-domain
description: Stand up a new *domain* of related Models inside an existing NEML2 submodule — e.g., `models/solid_mechanics/viscoelasticity/`, `models/solid_mechanics/damage/`, `models/chemical_reactions/<new-family>/`. Use this skill whenever the user wants to add a *family* of related models at once (more than one class on a shared theme), introduce a new physical theory area to NEML2, or scaffold an end-to-end set of models + unit tests + regression tests for a constitutive framework. Trigger on phrases like "add fundamental viscoelasticity models", "create a damage mechanics subfolder", "implement a family of yield surfaces", "scaffold creep models under solid_mechanics", or any request that names a *theory* rather than a single model. For a single new Model class, use `add-model` instead.
---

# add-domain

Add a new *domain* (subfolder) of related Models inside an existing submodule of NEML2 — header tree, source tree, unit tests, and regression tests. The single-class scaffolding is delegated to `add-model`; this skill handles the planning, composition, and verification work that single-class scaffolding doesn't cover.

## When to use

Trigger this skill when the user wants to introduce a *theme* of models, not just one. A few clues:

- They name a physical theory ("viscoelasticity", "damage", "creep", "phase transformation kinetics") rather than a single class name.
- They list multiple models in one breath ("Maxwell, Kelvin-Voigt, Zener, …").
- They mention end-to-end testing alongside the models.

For a **single new Model**, use `add-model` directly — the planning overhead in this skill is wasted on one class.
For a **brand-new top-level submodule** (a sibling of `models/`, `solvers/`, `drivers/`, etc.) — that needs `neml2_add_submodule(...)`, a `link_anchor.cxx`, and a force-link symbol declared in `include/neml2/neml2.h`. Surface that to the user instead of attempting it; this skill targets *domain subfolders inside an existing submodule*, not new submodules.

**Always confirm the destination before scaffolding** (see Step 0). A request like "add a bunch of crystal plasticity models" can land in the existing `models/solid_mechanics/crystal_plasticity/` folder rather than a new domain — picking wrong creates a parallel directory tree that fragments the same theory across two locations and is painful to undo.

## The workflow

0. **Confirm the destination** — verify with the user *whether* a new domain folder is needed and *where* it goes. Skip nothing here even if the request seems unambiguous.
1. **Plan the math and the composition** — interactively, with the user, before writing code.
2. **Scaffold each model** via `add-model`, then fill in the constitutive equations and analytic derivatives.
3. **Verify each model in isolation** with `ModelUnitTest` (`.i` files) — both forward values and derivatives.
4. **Verify the composition** with `neml2-inspect` *before* trying to time-step — confirm the wiring before the simulation.
5. **Verify against ground truth** with `add-verification` — establish that the model produces a *correct* answer (not just a stable one). When no defensible reference exists, surface the uncertainty to the user and skip only with their consent.
6. **Pin the verified behavior** with `add-regression` — freeze the now-known-good output so future changes don't silently break it.
7. **Document the new domain** under `doc/content/modules/` so that the physics and a worked input file land in the user-facing docs.

The first step is where this skill earns its keep. The rest is mechanical, but the gotchas in steps 4–7 are easy to miss.

---

## Step 0 — Confirm the destination (do this BEFORE Step 1)

The skill is named `add-domain`, but adding a domain is *not always the right answer*. A request that names a theory ("add some crystal plasticity slip rules", "implement a few damage models") can mean any of:

1. **Add to an existing domain folder** — the theory already lives somewhere in NEML2. New models go *into* `models/solid_mechanics/crystal_plasticity/`, not into a new sibling folder. This is the common case and the skill should default to assuming it.
2. **Add a new domain folder under an existing submodule** — the theory is genuinely new to NEML2 (or is a distinct sub-theory that doesn't fit the existing folder). This is what the rest of this skill covers.
3. **Add a brand-new top-level submodule** — out of scope for this skill (see "When to use" above); surface to the user.

Before doing any planning, do this small audit:

```bash
# What submodules exist?
ls include/neml2/models/

# What domain folders exist inside the most plausible submodule?
ls include/neml2/models/<submodule>/
```

Then **explicitly confirm with the user** which of (1)/(2)/(3) applies. Frame it concretely — name the existing folder you found and ask whether the new models belong there or in a new folder, and *why*. For example:

> I see `models/solid_mechanics/crystal_plasticity/` already exists with `{...list a few files...}`. Should the new slip rules go in there, or do you want a new sibling folder (e.g. `crystal_plasticity_<something>/`)? If the latter, what distinguishes it from the existing folder?

If the answer is (1), drop out of this skill and use `add-model` per file instead — the multi-model planning workflow below is still useful, but the directory creation and doc registration in steps 1, 7 don't apply.

Skipping this step is the most common way `add-domain` goes wrong: a new folder gets stood up next to a perfectly good existing one, and the theory ends up split across two locations. Renaming or merging folders later is much more painful than asking one question now.

---

## Step 1 — Plan the math and composition (do this BEFORE writing code)

The hardest part of adding a domain is not C++ — it's deciding **which equations belong in which Model**, **which variables are inputs vs outputs vs internal state**, and **what already exists in NEML2 that you can reuse**. Get this wrong and you'll re-design midway through, throwing away code.

NEML2 models are small composable pieces, not monoliths. A constitutive theory typically becomes 3–5 small Models that compose with existing utilities to form a complete simulation. Pick the wrong split and you either (a) build a god-object that's hard to reuse, or (b) build pieces that don't compose with the rest of NEML2.

**For each model the user wants, draft a short specification and ask the user to confirm before writing code:**

```
Model: <Name>
Physics:        <one-sentence description, ideally with the governing equation>
Inputs:         <variable name : tensor type, e.g. "stress : SR2", "viscous_strain : SR2">
Outputs:        <variable name : tensor type>
Parameters:     <name : tensor type — e.g. "viscosity : Scalar">
Internal state: <which outputs are *rates* of variables that the framework will time-integrate>
Equation:       <the actual equation, in LaTeX-ish form, that set_value will implement>
```

Then list the **composition plan** — what existing NEML2 pieces this model needs to plug into to form a complete simulation. Common reusable pieces:

- `LinearIsotropicElasticity` (and friends) — stress ↔ elastic strain
- `SR2VariableRate` / `ScalarVariableRate` — turn a state variable into its rate via finite differencing in time
- `SR2LinearCombination` / `ScalarLinearCombination` — sum / subtract variables (e.g. `elastic_strain = strain - viscous_strain`)
- `SR2BackwardEulerTimeIntegration` / `ScalarBackwardEulerTimeIntegration` — turn a rate into an implicit residual on the underlying state variable
- `ComposedModel` — glue several Models into one
- `NonlinearSystem` + `Newton` + `DenseLU` + `ImplicitUpdate` — solve for unknowns at each time step
- `ConstantExtrapolationPredictor` — initial guess for the implicit solve
- `TransientDriver` + `TransientRegression` — drive a load history and compare against a gold reference

Survey the target subdirectory (`ls include/neml2/models/<submodule>/`) and any obviously-related subdirectories before drafting. Often a domain interface already exists (e.g., `IsotropicHardening`, `FlowRule`, `Elasticity`) that the new models should subclass instead of `Model` directly. **Check before deciding** — copying the wrong base class is the most common architectural mistake.

For a structured catalog of registered objects, use `neml2-syntax`. Three flags make planning fast:

- `--section Models` filters by input-file section (also valid: `Solvers`, `Drivers`, `Tensors`, `Schedulers`, `Data`, `EquationSystems`, `Settings`).
- `--summary` drops the per-option detail and emits just `type`, `section`, and `doc` — exactly the level of detail you need to pick candidates.
- `--type <Name>` narrows to one specific object — use this *after* the summary catalog has helped you pick a name.

The intended two-step workflow:

```bash
# 1. Browse one-line descriptions of every registered Model.
./build/dev/src/tools/neml2-syntax --section Models --summary | less

# 2. Once you've picked a candidate, drill in for its full option list.
./build/dev/src/tools/neml2-syntax --type SR2BackwardEulerTimeIntegration

# Or dump everything to a file (the catalog used by the doc generator).
./build/dev/src/tools/neml2-syntax --yaml /tmp/syntax.yml
```

The summary catalog catches utilities that `ls` misses (e.g., the `*BackwardEulerTimeIntegration` and `*VariableRate` helpers live under `models/common/`, not where you'd guess from a physics theme), and is small enough to read end-to-end without grep gymnastics. `--type` then gives you the per-option detail without dumping the whole registry.

**Why this step exists:** it is far cheaper to revise an ASCII spec in chat than to revise five `.h`/`.cxx`/`.i` triples that already compile. Skipping the spec step doesn't make you faster; it makes you re-do work. *Do not start writing code until the user has signed off on the spec.*

### Example spec (from a real session)

```
Model: ZenerViscoelasticity
Physics:        Standard Linear Solid — equilibrium spring in parallel with a Maxwell branch
Inputs:         strain : SR2, viscous_strain : SR2 (Maxwell-branch internal state)
Outputs:        stress : SR2, viscous_strain_rate : SR2
Parameters:     equilibrium_modulus, maxwell_modulus, maxwell_viscosity (all Scalar)
Internal state: viscous_strain (the framework will time-integrate viscous_strain_rate via SR2BackwardEulerTimeIntegration)
Equation:       sigma_M = E_M * (eps - eps_v)
                sigma   = E_inf * eps + sigma_M
                d(eps_v)/dt = sigma_M / eta_M

Composition (no new pieces beyond this Model):
  - SR2BackwardEulerTimeIntegration on viscous_strain
  - NonlinearSystem unknown = viscous_strain, residual = viscous_strain_residual
  - ImplicitUpdate + Newton + DenseLU + ConstantExtrapolationPredictor
  - TransientDriver prescribes strain over time
```

If the user provides handwavy physics ("just add Maxwell"), translate it into a spec like the above and confirm. If they provide an explicit governing equation, lift it directly into the spec. Either way, get sign-off before scaffolding.

---

## Step 2 — Scaffold each model

For each model in the spec, follow the `add-model` skill — do not duplicate its file templates here. The high-level shape is unchanged: `include/neml2/models/<submodule>/<domain>/<Name>.h`, `src/.../<Name>.cxx`, `tests/unit/models/<submodule>/<domain>/<Name>.i`. CMake `file(GLOB_RECURSE)` picks up new files automatically — no CMake edits.

Two things specific to a multi-model domain:

- **Use the same naming and parameter conventions across the family.** If one model in the domain calls its dashpot parameter `viscosity`, every model in the family should call it `viscosity` (not `eta` or `damping`). Inputs and parameter names show up in user-facing input files; inconsistency forces users to context-switch.
- **Derive rate, history, and residual variable names — never request them as input options.** When a model produces (or consumes) a *rate* of an existing state variable, the rate name must come from `rate_name(<base>.name())` so it lines up with whatever `BackwardEulerTimeIntegration` and friends derive. Same for `history_name` (previous-step values) and `residual_name`. These helpers respect `[Settings]` overrides for the prefix/suffix/separator. If you `add_input`/`add_output` a literal name like `"viscous_strain_rate"`, the model agrees with the framework only by accident at default settings — and the user can supply a name that silently mismatches the integrator. See the `add-model` skill's "Naming convention for rates, histories, and residuals" section for the pattern. This is the most common naming bug across the family: review every `_dot` / `_rate` / `~1` / `_residual` declaration before you ship.
- **Provide analytic derivatives** unless you have a strong reason not to. The `ModelUnitTest` cross-checks them against finite differencing; getting them right pays off immediately and never has to be revisited. If you can't derive them quickly, scaffold with `check_first_derivatives = false` in the `.i`, but treat that as a TODO, not a long-term state.

---

## Step 3 — Unit-test every model

Each `.i` file under `tests/unit/models/<submodule>/<domain>/` is auto-discovered by the `unit_tests` binary. Use `ModelUnitTest` and provide:

- Hand-computed reference outputs (don't accept whatever the model produces — that defeats the test)
- `check_first_derivatives = true` (the default) so analytic derivatives are cross-checked against FD

After building (`cmake --build --preset dev`), run each test by section name:

```bash
./build/dev/tests/unit/unit_tests "models" -c "<submodule>/<domain>/<Name>.i"
```

The `-c` flag selects a Catch2 dynamic section. If your section path is wrong, you'll get "1 test case passed, 0 assertions" — the test silently no-ops instead of failing. If you see that, double-check the `-c` path matches the file's relative path from `tests/unit/models/`.

Run the whole family at the end to confirm nothing regressed.

---

## Step 4 — Verify the composition with `neml2-inspect` BEFORE running

This is the step most people skip and then regret. Once your unit tests pass, you have working pieces — but nothing has yet checked that they *fit together correctly*. `neml2-inspect` (built when `NEML2_TOOLS=ON`, i.e., almost always) prints what each Model declares as inputs, outputs, and parameters, and how a `ComposedModel` resolves the dependency graph.

```bash
./build/dev/src/tools/neml2-inspect <input-file.i>
```

Use it to confirm, before launching `neml2-run`:

- Each submodel sees the inputs it expects (no typos in variable names).
- Outputs of one submodel match the inputs of the next (e.g., `viscous_strain_rate` from your model matches what `SR2BackwardEulerTimeIntegration` expects, named `viscous_strain_rate`).
- The `NonlinearSystem`'s `unknowns` and `residuals` correspond to outputs your composed model actually produces.
- Nothing is left dangling — every input is sourced, every output is consumed or surfaced as `additional_outputs`.

If `neml2-inspect` shows a problem, you haven't started time-stepping yet — fix the wiring at zero cost. If you skip this step and run `neml2-run` instead, the failure mode is often a cryptic shape mismatch deep inside Newton, which is much harder to localize.

`neml2-diagnose <file.i>` is the companion check: it runs each model's `diagnose()` method to catch misconfigurations the parser cannot detect statically. (`neml2-syntax` is for dumping the registry — see Step 1; it doesn't take an input file.)

---

## Step 5 — Verify against ground truth (when feasible)

A verification test answers *did the model produce a physically correct result?* The reference data comes from outside the model — an analytical solution (e.g., Maxwell stress relaxation \f$ \sigma(t) = \sigma_0 e^{-t/\tau} \f$ for a step strain), an established benchmark code (NEML1, WARP3D, etc.), or a strong physical-intuition result when no closed form exists (long-time stress, monotonicity, recoverability after fully reversed loading).

Use the `add-verification` skill for the workflow — it covers the `.vtest` reference-data format, the `VTestVerification` driver, where the harness expects the driver to be named, and how to choose tolerances. For each qualitatively distinct model in the domain, pick the most defensible source of truth available; even a "weaker" intuition check is independent evidence the unit tests cannot provide on their own.

For the viscoelasticity domain we built as a worked example, none of the five models had `.vtest` benchmark data on hand — but each has a closed-form long-time / short-time response that's straightforward to check (Maxwell relaxes to zero, Zener relaxes to \f$ E_\infty \boldsymbol{\varepsilon} \f$, Kelvin-Voigt creeps to \f$ \boldsymbol{\sigma}/E \f$, etc.). Those analytical limits are the right verification reference.

**When verification is not feasible.** Sometimes a defensible reference genuinely doesn't exist — the model captures novel physics with no published benchmark, no closed-form limit you can isolate, and no intuition test that would distinguish a correct implementation from a plausible-looking wrong one. **Do not fabricate a reference to check a box.** A verification test that compares the model against itself, or against a hand-tuned tolerance on a number you pulled from running the model and squinting at the plot, is worse than no verification test at all — it gives false confidence and pins a guess.

When you are unsure what to verify against:

- Surface the uncertainty to the user *before* skipping. State concretely what you considered (analytical limits, benchmark codes, intuition checks) and why each one is too weak or unavailable for this model. Ask whether they know of a reference you don't, or whether to skip with a TODO.
- If the user consents, skip the verification step for that model. Note the gap in the regression scenario's input file (a `# TODO: no verification reference yet — gold reflects current behavior, not known-good behavior.` comment near the `TransientRegression` block is a useful signal to future readers) and proceed to step 6.
- If only *some* models in the domain have defensible references, do verification for those and skip the rest with the same surfacing-and-consent flow. Partial verification is much better than none.

---

## Step 6 — Pin the verified behavior with regression tests

A regression test answers *did the model's output drift from what it produced last time?* It's a change-detection tool — useful for catching unintended numerical changes from refactors, dependency bumps, or unrelated edits. The reference (`gold/result.pt`) is the model's own previous output, not external truth.

Use the `add-regression` skill for the workflow — it covers the directory layout (`<scenario>/model.i` + `gold/result.pt`), the `neml2-run` bootstrap with the benign `TransientRegression not registered` warning, the cleanup of stray `result.pt` files, and the Catch2 silent-pass-with-bad-section trap.

The reason verification (step 5) comes first: a regression test by itself only freezes whatever the model happens to produce on the day you bootstrap it — including bugs. Verifying first means the gold you commit is gold, not just lead.

For a multi-model domain, add at least one regression scenario per *qualitatively distinct* model — typically one per model with internal state (these exercise the `ImplicitUpdate` path), one for each "pure forward" model if any, and one for the typical composition pattern users will reach for (e.g., for Maxwell-style dashpots, regress the composition with elasticity rather than the bare rate equation).

---

## Step 7 — Document the new domain

Add a new `##` section to the appropriate `doc/content/modules/<submodule>.md` — for a domain inside an existing submodule (e.g. viscoelasticity inside `solid_mechanics`), no XML edits are needed; the in-page TOC builds itself. For a brand-new top-level physics module, both `doc/config/DoxygenLayout.xml` and `doc/config/DoxygenLayoutPython.xml` must be updated to make the page appear in the sidebar.

The full editorial guidance — what to write at what level of detail, the `@list-input:` directive and its `git ls-files` quirk, layout-XML registration, build verification gotchas — is in the `add-doc` skill. The short version: a domain doc should be a high-level intro + canonical names + pointer to `[Syntax Documentation](@ref syntax-models)` + one worked input-file example. Per-class detail belongs in `expected_options()` doc strings, not in the narrative — it ages out of sync with the registry otherwise.

The regression tests you wrote in step 5 are the right inputs to point `@list-input:` at — but `git add` them first, otherwise the doc build will fail to resolve the path.

---

## Pre-commit and CI

Before reporting done:

```bash
pre-commit run --files \
  include/neml2/models/<submodule>/<domain>/*.h \
  src/neml2/models/<submodule>/<domain>/*.cxx
```

`clang-format` and `check_copyright.py` are enforced by CI. `clang-format` *will* rewrite lines (e.g., wrap long doc strings); that's expected, just rebuild and rerun the tests after.

---

## Common failure modes and how to recognize them

- **"unknown type Foo"** when running an `.i` test — missing `register_NEML2_object(Foo);` in the `.cxx`, or the macro is inside a function instead of at namespace scope.
- **Catch2 reports "1 test case passed, 0 assertions"** — the `-c` section pattern didn't match anything. Check the path is the relative path from `tests/unit/models/`.
- **"unknown variable" or shape-mismatch deep inside Newton** — a composition wiring bug that `neml2-inspect` would have caught. Go back to step 4.
- **Regression test compares against an old `gold/result.pt`** that doesn't reflect a recent equation change — you have to regenerate with `neml2-run` and re-verify.
- **Model takes a tensor by value but only forward-declares the type in the header** — fine in the `.h`, but the `.cxx` needs the full `#include "neml2/tensors/<T>.h"`.

---

## Reference: anatomy of a domain (real example)

`models/solid_mechanics/viscoelasticity/` ships five Models — `MaxwellViscoelasticity`, `KelvinVoigtViscoelasticity`, `ZenerViscoelasticity`, `WiechertViscoelasticity`, `BurgersViscoelasticity` — with one `.i` unit test each and five regression scenarios under `tests/regression/solid_mechanics/viscoelasticity/{maxwell,kelvin_voigt,zener,burgers,wiechert}/`. Browse it for a working layout — the family-level naming conventions (`viscosity`, `modulus`, `viscous_strain`, `viscous_strain_rate`) are a good template for any new constitutive-theory domain.
