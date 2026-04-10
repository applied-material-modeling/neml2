# Input

## Current inputs
- `_interface_displacement_jump[_qp]` — displacement jump vector in interface coordinates `[delta_n, delta_t1, delta_t2]` (AD material property)

## State inputs
- `_effective_displacement_jump_scalar_max_old[_qp]` — maximum effective scalar jump from previous step (state input, non-AD `Real`)

## Parameters
- `_Gc` — fracture energy G_c
- `_delta0` — softening length scale δ₀
- `_beta` — tangential weighting factor β
- `_eps` — small regularizer for norm (default 1e-16)

## Control inputs
- `_irreversible_damage` — bool; if true, use historical max δ_eff (no healing)

---

# Algorithm

1. **Extract jump components**
   - `delta_n = jump_on_GP(0)`
   - `delta_t_sq = jump_on_GP(1)² + jump_on_GP(2)²`
   - `delta_n_sq = delta_n²`

2. **Compute effective scalar jump**
   - `delta_eff = sqrt(delta_n_sq + beta * delta_t_sq + eps)`
   - Note: `delta_t_sq` is used directly (no intermediate sqrt) to avoid AD singularity at zero tangential jump

3. **Store effective displacement jump vector** (raw values, no AD)
   - `_interface_effective_displacement_jump[_qp] = [raw(delta_n), sqrt(beta)*raw(delta_t1), sqrt(beta)*raw(delta_t2)]`

4. **Irreversibility branch** (only if `_irreversible_damage == true`)
   - if `_effective_displacement_jump_scalar_max_old[_qp] < delta_eff.value()`:
     - `_effective_displacement_jump_scalar_max[_qp] = delta_eff`  ← update max (AD)
   - else:
     - `delta_eff = ADReal(_effective_displacement_jump_scalar_max_old[_qp])`  ← freeze at old max
     - `_effective_displacement_jump_scalar_max[_qp]` is NOT written here (left at old value via stateful copy)

5. **Compute damage**
   - `d = 1 - exp(-delta_eff / delta0)`
   - `_damage[_qp] = d`

6. **Compute traction**
   - `c = Gc / delta0²`
   - `_interface_traction[_qp] = (1 - d) * c * jump_on_GP`
   - Applied uniformly to all three components (normal and tangential)

---

# Output

## Primary outputs
- `_interface_traction[_qp]` — traction vector in interface coordinates (AD, written to base class member)

## Updated state outputs
- `_effective_displacement_jump_scalar_max[_qp]` — updated max scalar jump (updated state output; only advances when `delta_eff > old_max` and `irreversible_damage == true`)

## Optional derived outputs
- `_damage[_qp]` — scalar damage variable d ∈ [0, 1)
- `_interface_effective_displacement_jump[_qp]` — effective jump vector (raw, non-AD; for post-processing)

---

## Source files read
- `src/materials/ExpTractionSeparation.C`
- `include/materials/ExpTractionSeparation.h`

## Open issues
- When `_irreversible_damage == false`, `_effective_displacement_jump_scalar_max[_qp]` is declared but never written in `computeInterfaceTraction()` (only initialized to 0 in `initQpStatefulProperties`); this is harmless but the property is dead in the reversible case.
- The irreversibility branch does not write `_effective_displacement_jump_scalar_max[_qp]` in the non-advancing case — it relies on MOOSE's stateful property copy. Verify this is correct if the property is ever consumed downstream as an AD quantity (it is declared AD but the old value is non-AD `Real`).
