Audit NEML2-specific naming consistency in the files given in `$ARGUMENTS`.

If `$ARGUMENTS` is empty, audit all files changed relative to `main`:
```bash
git diff --name-only main
```

Ignore everything in `contrib/`.

---

## What to check

Run all four checks. Report only concrete mismatches — do not flag uncertain or subjective cases.

---

### 1. File name vs. primary class name

For each `.h` file, extract the primary class or struct name (the one that matches the filename stem).
Report if the file stem and the class name differ.

```
include/neml2/models/FooBar.h  defines class FooBaz
→ FILE  [class-file-mismatch]  file: FooBar  class: FooBaz
```

---

### 2. Registration macro consistency

For each `.cxx` file that contains `register_NEML2_object(...)`, check that the argument
matches the primary class name defined in the corresponding `.h` file.

```bash
grep -n "register_NEML2_object" src/neml2/models/Foo.cxx
```

Report if the macro argument does not match the class name.

```
src/neml2/models/Foo.cxx:42  [registration-mismatch]  macro: FooBaz  class: FooBar
```

---

### 3. HIT `type =` vs. registered class name

For each `.i` file in scope, extract every `type = Value` assignment. Check that `Value`
matches a class registered via `register_NEML2_object` somewhere in `src/neml2/`.

```bash
grep -rn "register_NEML2_object" src/neml2/ | sed 's/.*(\(.*\)).*/\1/'
grep -n "type\s*=" tests/ doc/ models/ -- *.i
```

Report any `type = Value` where `Value` is not in the registered class list.

```
models/foo.i:7  [unregistered-type]  type = FooBaz  (not found in register_NEML2_object calls)
```

---

### 4. Member variable `_` prefix

For each `.h` file in scope, check that protected and private member variable declarations
follow the `_snake_case` convention (leading underscore). Only flag clear violations where
a member is declared without a leading `_` and is not a type alias, `static`, or `constexpr`.

```
include/neml2/models/Foo.h:18  [member-prefix]  found: elasticModulus  expected: _elastic_modulus
```

---

## Output format

```
FILE:LINE  [rule]  found: `actual`  expected: `expected`
```

Group by file. Print a count of total violations at the end.

Do NOT auto-fix. Do NOT flag uncertain cases. If a check cannot be performed confidently
(e.g. a complex macro or template alias obscures the class name), skip it silently.
