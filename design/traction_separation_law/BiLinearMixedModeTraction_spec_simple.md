# Input

## Current inputs
- `_interface_displacement_jump[_qp]` — displacement jump in local interface frame, ordered `[delta_n, delta_s1, delta_s2]`

## State inputs
- `_d_old[_qp]` — damage variable from previous step (state input)
- `_interface_displacement_jump_old[_qp]` — displacement jump from previous step (state input); used when `lag_mode_mixity` or `lag_displacement_jump` is active

## Parameters
- `_K` — penalty elastic stiffness
- `_GI_c[_qp]` — Mode I critical energy release rate
- `_GII_c[_qp]` — Mode II critical energy release rate
- `_N[_qp]` — tensile (normal) strength
- `_S[_qp]` — shear strength
- `_eta` — power law exponent (BK or POWER_LAW criterion)
- `_viscosity` — viscous regularization coefficient (0 = no regularization)
- `_dt` — time step size (from MOOSE, used for viscous regularization)

## Control inputs
- `_criterion` — mixed mode propagation criterion: `BK` (default) or `POWER_LAW`
- `_lag_mode_mixity` — bool (default `true`): use `delta_old` when computing `_beta`, `_delta_init`, `_delta_final`
- `_lag_disp_jump` — bool (default `false`): use `delta_old` when computing `_delta_m`
- `_alpha` — regularization parameter for the Macaulay bracket (default `1e-10`)

---

# Algorithm

## 1. Compute mode mixity ratio (`computeModeMixity`)

Select displacement jump for mixity:
```
delta = lag_mode_mixity ? delta_old : delta_current
```

If `delta(0) > 0` (normal opening):
```
delta_s = sqrt(delta(1)^2 + delta(2)^2)
_beta   = delta_s / delta(0)
```
If not lagged:
- if `delta_s ≈ 0`: `_dbeta_ddelta = (-delta_s/delta(0)^2,  0, 0)`
- else:             `_dbeta_ddelta = (-delta_s/delta(0)^2,  delta(1)/(delta_s*delta(0)),  delta(2)/(delta_s*delta(0)))`

Else (`delta(0) <= 0`, no opening):
```
_beta = 0,  _dbeta_ddelta = 0
```

## 2. Compute damage-initiation displacement jump (`computeCriticalDisplacementJump`)

Select displacement jump (same flag as step 1):
```
delta_normal0 = N / K
delta_shear0  = S / K
```

Default (pure shear or no normal opening):
```
_delta_init = delta_shear0
_ddelta_init_ddelta = 0
```

If `delta(0) > 0`:
```
delta_mixed  = sqrt(delta_shear0^2 + (beta * delta_normal0)^2)
_delta_init  = delta_normal0 * delta_shear0 * sqrt(1 + beta^2) / delta_mixed
```
If not lagged:
```
ddelta_init_dbeta    = _delta_init * beta * (1/(1+beta^2) - delta_normal0^2/delta_mixed^2)
_ddelta_init_ddelta  = ddelta_init_dbeta * _dbeta_ddelta
```

## 3. Compute full-degradation displacement jump (`computeFinalDisplacementJump`)

Select displacement jump (same flag):

Default (pure shear):
```
_delta_final = sqrt(2) * 2 * GII_c / S
_ddelta_final_ddelta = 0
```

If `delta(0) > 0`, branch on criterion:

### BK:
```
beta_sq_ratio  = beta^2 / (1 + beta^2)
_delta_final   = 2 / (K * delta_init) * (GI_c + (GII_c - GI_c) * beta_sq_ratio^eta)
```
If not lagged:
```
ddelta_final_ddelta_init  = -delta_final / delta_init
dbeta_sq_ratio_dbeta      = 2*beta / (1 + beta^2)^2
ddelta_final_dbeta        = 2/(K*delta_init) * (GII_c - GI_c) * eta
                            * beta_sq_ratio^(eta-1) * dbeta_sq_ratio_dbeta
_ddelta_final_ddelta      = ddelta_final_ddelta_init * _ddelta_init_ddelta
                          + ddelta_final_dbeta       * _dbeta_ddelta
```

### POWER_LAW:
```
Gc_mixed      = (1/GI_c)^eta + (beta^2/GII_c)^eta
_delta_final  = (2 + 2*beta^2) / (K * delta_init) * Gc_mixed^(-1/eta)
```
If not lagged:
```
ddelta_final_ddelta_init = -delta_final / delta_init
dGc_mixed_dbeta          = eta * (beta^2/GII_c)^(eta-1) * (2*beta/GII_c)
prefactor                = (2 + 2*beta^2) / (K * delta_init)
dprefactor_dbeta         = 4*beta / (K * delta_init)
dGc_term_dbeta           = (-1/eta) * Gc_mixed^(-1/eta - 1) * dGc_mixed_dbeta
ddelta_final_dbeta       = dprefactor_dbeta * Gc_mixed^(-1/eta) + prefactor * dGc_term_dbeta
_ddelta_final_ddelta     = ddelta_final_ddelta_init * _ddelta_init_ddelta
                         + ddelta_final_dbeta       * _dbeta_ddelta
```

## 4. Compute effective displacement jump (`computeEffectiveDisplacementJump`)

Select displacement jump:
```
delta = lag_disp_jump ? delta_old : delta_current
```

Regularized Macaulay bracket for normal component:
```
delta_n_pos = regularizedHeavyside(delta(0), alpha) * delta(0)
_delta_m    = sqrt(delta(1)^2 + delta(2)^2 + delta_n_pos^2)
_ddelta_m_ddelta = 0
```

If not lagged and `delta_m != 0`:
```
ddelta_n_pos_ddelta_n = regularizedHeavysideDerivative(delta(0), alpha)*delta(0)
                      + regularizedHeavyside(delta(0), alpha)
_ddelta_m_ddelta      = (delta_n_pos * ddelta_n_pos_ddelta_n,  delta(1),  delta(2)) / delta_m
```

## 5. Compute damage (`computeDamage`)

Bilinear law:
```
if   delta_m < delta_init:   d_trial = 0
elif delta_m > delta_final:  d_trial = 1
else:
    d_trial = delta_final * (delta_m - delta_init) / (delta_m * (delta_final - delta_init))
```

`_dd_ddelta = 0` (reset)

Irreversibility:
```
if d_trial < d_old:
    _d = d_old              # no derivative contribution
elif delta_m in [delta_init, delta_final]:
    _dd_ddelta = [
        _ddelta_final_ddelta * (delta_m - delta_init)
      + delta_final * (_ddelta_m_ddelta - _ddelta_init_ddelta)
    ] / (delta_m * (delta_final - delta_init))
    -
    delta_final * (delta_m - delta_init) * [
        _ddelta_m_ddelta * (delta_final - delta_init)
      + delta_m * (_ddelta_final_ddelta - _ddelta_init_ddelta)
    ] / (delta_m * (delta_final - delta_init))^2
```

Viscous regularization:
```
_d          = (d_trial + viscosity * d_old / dt) / (viscosity/dt + 1)
_dd_ddelta /= (viscosity/dt + 1)
```

## 6. Compute traction (`computeTraction`)

Regularized normal split (Macaulay):
```
H           = regularizedHeavyside(delta(0), alpha)
delta_n_pos = H * delta(0)
delta_n_neg = delta(0) - delta_n_pos

delta_active   = (delta_n_pos, delta(1), delta(2))
delta_inactive = (delta_n_neg,        0,        0)

traction = (1 - d) * K * delta_active + K * delta_inactive
```

## 7. Compute traction Jacobian (`computeTractionDerivatives`)

```
H    = regularizedHeavyside(delta(0), alpha)
dH   = regularizedHeavysideDerivative(delta(0), alpha)

ddelta_n_pos_ddelta_n = H + delta(0)*dH
ddelta_n_neg_ddelta_n = 1 - ddelta_n_pos_ddelta_n

ddelta_active_ddelta   = diag(ddelta_n_pos_ddelta_n, 1, 1)     [3x3 diagonal]
ddelta_inactive_ddelta = diag(ddelta_n_neg_ddelta_n, 0, 0)     [3x3 diagonal]

dtraction_ddelta = (1-d)*K*ddelta_active_ddelta + K*ddelta_inactive_ddelta
```

Chain-rule damage contribution:
```
A                 = outer(delta_active, _dd_ddelta)    [outer product]
dtraction_ddelta -= K * A
```

---

# Output

## Primary outputs
- `_interface_traction[_qp]` — traction vector in local interface frame `[T_n, T_s1, T_s2]`
- `_dinterface_traction_djump[_qp]` — 3×3 Jacobian of traction w.r.t. displacement jump

## Updated state outputs
- `_d[_qp]` — damage variable after irreversibility and viscous regularization (updated state output)

## Optional derived outputs
- `_beta[_qp]` — mode mixity ratio `delta_s / delta_n`
- `_delta_init[_qp]` — effective displacement jump at damage initiation
- `_delta_final[_qp]` — effective displacement jump at full degradation
- `_delta_m[_qp]` — current effective mixed-mode displacement jump

---

## Source files read
- `modules/solid_mechanics/src/materials/cohesive_zone_model/BiLinearMixedModeTraction.C`
- `modules/solid_mechanics/include/materials/cohesive_zone_model/BiLinearMixedModeTraction.h`
- `modules/solid_mechanics/include/materials/cohesive_zone_model/CZMComputeLocalTractionTotalBase.h`
- `modules/solid_mechanics/include/materials/cohesive_zone_model/CZMComputeLocalTractionBase.h`

## Open issues
- `delta_final` default (pure-shear path) uses `sqrt(2)*2*GII_c/S` — the `sqrt(2)` factor origin is not documented in code comments; cross-check with reference (Camanho & Davila NASA/TM-2002-211737).
- When `lag_mode_mixity=true` and `lag_displacement_jump=false`, `_delta_init`/`_delta_final` are computed from `delta_old` while `_delta_m` uses `delta_current`, creating a time-lag split; ensure this is intentional.
- `_dbeta_ddelta` is set to `(-delta_s/delta(0)^2, 0, 0)` when `delta_s ≈ 0` — the normal component is non-zero despite pure normal loading; verify sign/intent.
- Jacobian is not computed for lagged quantities (`_lag_mode_mixity=true`, `_lag_disp_jump=true`); those branches leave `_ddelta_final_ddelta`, `_ddelta_init_ddelta`, `_ddelta_m_ddelta` as zero, which is correct but worth confirming in convergence tests.
