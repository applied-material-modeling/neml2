# Input

## Current inputs
- `_interface_displacement_jump[_qp]` — displacement jump in local interface frame, ordered `[delta_n, delta_s1, delta_s2]`

## State inputs
- None

## Parameters
- `normal_stiffness` (`K_n`) — elastic stiffness in the normal direction
- `tangent_stiffness` (`K_t`) — elastic stiffness in both tangential directions (isotropic)

## Control inputs
- None

---

# Algorithm

## 1. Build stiffness tensor (constructor, once)

```
_K = diag(K_n, K_t, K_t)    [3x3 diagonal RankTwoTensor]
```
- source: constructor, `RankTwoTensor(std::vector<Real>{K_n, K_t, K_t})`

## 2. Compute traction (`computeInterfaceTractionAndDerivatives`)

```
_interface_traction[_qp] = _K * _interface_displacement_jump[_qp]
```
Expanded:
```
T_n  = K_n * delta_n
T_s1 = K_t * delta_s1
T_s2 = K_t * delta_s2
```
- No branching, no guards, no state update.

## 3. Compute Jacobian

```
_dinterface_traction_djump[_qp] = _K    [constant diagonal 3x3]
```
- Exact, no approximation.

---

# Output

## Primary outputs
- `_interface_traction[_qp]` — traction vector in local interface frame `[T_n, T_s1, T_s2]`
- `_dinterface_traction_djump[_qp]` — 3×3 stiffness tensor (constant, diagonal)

## Updated state outputs
- None

## Optional derived outputs
- None

---

## Source files read
- `modules/solid_mechanics/src/materials/cohesive_zone_model/PureElasticTractionSeparation.C`
- `modules/solid_mechanics/include/materials/cohesive_zone_model/PureElasticTractionSeparation.h`

## Open issues
- No contact/compression handling: the normal stiffness `K_n` is applied uniformly for both opening and closing (`delta_n > 0` and `delta_n < 0`). If used in problems with interface compression, verify this is intentional or pair with a penalty contact term.
- The header docstring references "exponential traction separation law (Salehani & Irani 2018)" — this is a copy-paste error; the implementation is purely linear elastic.
