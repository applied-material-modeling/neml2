(physics-modules)=
# Physics modules

NEML2's model catalog is organized by physics domain. Each module
ships a self-contained set of `Model` building blocks that compose
into full constitutive theories via `ComposedModel` and the input-file
wiring rules covered in [](tutorials-models-cross-referencing).

This section gives the per-module orientation: the physics or
numerical method the catalog covers, the key equations, the
canonical model composition, and the prose explanation that ties
them together. For the per-type option-by-option reference, see the
[](syntax-catalog).

```{toctree}
:maxdepth: 1

chemical_reactions
finite_volume
kwn/index
phase_field_fracture
porous_flow
solid_mechanics/index
```
