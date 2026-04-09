# Input

## Current inputs
- `_interface_displacement_jump[_qp]` — displacement jump in local interface frame, ordered `[delta_n, delta_s1, delta_s2]`

## State inputs
- None

## Parameters
- `normal_gap_at_maximum_normal_traction` — raw input; stored as `_delta_u0(0)`
- `tangential_gap_at_maximum_shear_traction` — raw input; stored as `_delta_u0(1) = _delta_u0(2) = sqrt(2) * input`
- `maximum_normal_traction` — stored as `_max_allowable_traction(0)`
- `maximum_shear_traction` — stored as `_max_allowable_traction(1) = _max_allowable_traction(2)`

## Control inputs
- None

---

# Algorithm

## 1. Build scaled characteristic-gap and prefactor vectors (constructor, once)

```
_delta_u0(0) = normal_gap_at_maximum_normal_traction
_delta_u0(1) = sqrt(2) * tangential_gap_at_maximum_shear_traction
_delta_u0(2) = sqrt(2) * tangential_gap_at_maximum_shear_traction
```

Prefactor `a_i` (computed inline each call, not stored):
```
a_0 = exp(1)           * _max_allowable_traction(0)   [normal, alpha=1]
a_1 = sqrt(2*exp(1))   * _max_allowable_traction(1)   [shear,  alpha=2]
a_2 = sqrt(2*exp(1))   * _max_allowable_traction(2)   [shear,  alpha=2]
```

## 2. Compute exponent `x` (`computeTraction` / `computeTractionDerivatives`)

```
x = (delta_n  / _delta_u0(0))
  + (delta_s1 / _delta_u0(1))^2
  + (delta_s2 / _delta_u0(2))^2
```
- Normal component enters linearly (alpha=1); shear components enter quadratically (alpha=2).
- No guards; `x` can be zero (zero-gap state), giving `exp(-x) = 1`.

```
exp_x = exp(-x)
```

## 3. Compute traction components (`computeTraction`)

For each direction `i = 0, 1, 2`:
```
b_i           = _interface_displacement_jump[_qp](i) / _delta_u0(i)
traction(i)   = a_i * b_i * exp_x
```
Expanded:
```
T_n  = exp(1)         * _max_allowable_traction(0) * (delta_n  / _delta_u0(0)) * exp(-x)
T_s1 = sqrt(2*exp(1)) * _max_allowable_traction(1) * (delta_s1 / _delta_u0(1)) * exp(-x)
T_s2 = sqrt(2*exp(1)) * _max_allowable_traction(2) * (delta_s2 / _delta_u0(2)) * exp(-x)
```
- No branching on sign of normal gap (no compression/opening split).
- No state update.

## 4. Compute Jacobian (`computeTractionDerivatives`)

Recompute `x` and `exp_x` identically to step 2 (separate code path, no shared state).

For each row `i` and column `j`:
```
dTi_duj = a_i * exp_x * (dbi_duj - b_i * dx_duj)
```

where:
```
dbi_duj = 1 / _delta_u0(i)   if i == j
          0                   otherwise

dx_duj  = 1 / _delta_u0(0)                                          if j == 0  [alpha=1]
          2 * delta_j / (_delta_u0(j) * _delta_u0(j))               if j > 0   [alpha=2]
```

Full 3×3 Jacobian stored in `_dinterface_traction_djump[_qp](i, j)`.

---

# Output

## Primary outputs
- `_interface_traction[_qp]` — traction vector in local interface frame `[T_n, T_s1, T_s2]`
- `_dinterface_traction_djump[_qp]` — 3×3 Jacobian of traction w.r.t. displacement jump (fully coupled, not diagonal)

## Updated state outputs
- None

## Optional derived outputs
- None

---

## Source files read
- `modules/solid_mechanics/src/materials/cohesive_zone_model/SalehaniIrani3DCTraction.C`
- `modules/solid_mechanics/include/materials/cohesive_zone_model/SalehaniIrani3DCTraction.h`

## Open issues
- No compression handling: normal traction is negative for `delta_n < 0` (interpenetration produces attraction). If interface compression is possible, a split or contact penalty term is needed.
- `x` and `exp_x` are computed twice identically — once in `computeTraction` and once in `computeTractionDerivatives`. No correctness issue, but worth noting for any refactor.
- The `sqrt(2)` scaling baked into `_delta_u0(1,2)` at construction silently shifts the characteristic gap for shear components. Any user comparing raw input to stored state should be aware of this scaling.
- No zero-gap guard: division by `_delta_u0(i)` is safe only if the user never sets `normal_gap_at_maximum_normal_traction = 0` or `tangential_gap_at_maximum_shear_traction = 0`. No runtime check exists.
