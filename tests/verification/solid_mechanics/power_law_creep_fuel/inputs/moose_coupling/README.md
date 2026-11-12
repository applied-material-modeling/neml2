# Custom MOOSE-side bridge material (NOT part of MOOSE or NEML2)

`ComputeNEML2StressOldSystem.{h,C}` is a material **we authored** to feed the NEML2
stress into MOOSE's legacy (pre-Lagrangian) tensor-mechanics system, so the original
`x447_dp11_fuel_moose.i` deck can take its creep stress from NEML2. It is not in
upstream MOOSE and not in NEML2 -- it is kept here only as reference so the coupled
run is reproducible.

To use it: drop the two files into `modules/solid_mechanics/{include,src}/materials/`
of a MOOSE app built against NEML2 v3 (we used the `hugary1995:neml2-v3-migration`
fork) and rebuild, then run `../fuel_old_coupled.i.txt`.
