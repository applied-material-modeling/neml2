@insert-title:tutorials-extension-model-visualization

[TOC]

## Motivation

During the process of model development, it is of general interest (and is also very much encouraged) to reuse existing models. This is particularly useful and effective owing to the extreme modular design of NEML2, i.e., model composition.

However, from times to times, the composed model may become overwhelmingly complicated even for an experienced developer. For example, given a single-crystal crystal plasticity model with decoupled time integration defined in an input file `input.i` with the following content
```python
[Tensors]
  [a]
    type = Scalar
    values = '1.0'
  []
  [sdirs]
    type = FillMillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = FillMillerIndex
    values = '1 1 1'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    max_linesearch_iterations = 5
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = 'a'
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Models]
  [euler_rodrigues_1]
    type = RotationMatrix
    from = 'forces/tmp/orientation'
    to = 'state/orientation_matrix'
  []
  [elasticity_1]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'state/elastic_strain'
    stress = 'state/internal/cauchy_stress'
  []
  [resolved_shear]
    type = ResolvedShear
  []
  [elastic_stretch]
    type = ElasticStrainRate
  []
  [plastic_deformation_rate]
    type = PlasticDeformationRate
  []
  [sum_slip_rates]
    type = SumSlipRates
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 8.0
    gamma0 = 2.0e-1
  []
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 500.0
    saturated_hardening = 50.0
  []
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/internal/slip_hardening'
  []
  [integrate_elastic_strain]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/elastic_strain'
  []
  [implicit_rate_1]
    type = ComposedModel
    models = "euler_rodrigues_1 elasticity_1 resolved_shear
              elastic_stretch plastic_deformation_rate
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain"
  []
  [subsystem1]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate_1'
    solver = 'newton'
  []
  [euler_rodrigues_2]
    type = RotationMatrix
    from = 'state/orientation'
    to = 'state/orientation_matrix'
  []
  [elasticity_2]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'forces/tmp/elastic_strain'
    stress = 'state/internal/cauchy_stress'
  []
  [orientation_rate]
    type = OrientationRate
    elastic_strain = 'forces/tmp/elastic_strain'
  []
  [plastic_spin]
    type = PlasticVorticity
  []
  [slip_strength_2]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
    slip_hardening = 'forces/tmp/internal/slip_hardening'
  []
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'state/orientation'
  []
  [implicit_rate_2]
    type = ComposedModel
    models = "euler_rodrigues_2 elasticity_2 resolved_shear
              plastic_deformation_rate plastic_spin
              slip_rule slip_strength_2 orientation_rate
              integrate_orientation"
  []
  [subsystem2]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate_2'
    solver = 'newton'
  []
  [cache_elastic_strain]
    type = CopySR2
    from = 'state/elastic_strain'
    to = 'forces/tmp/elastic_strain'
  []
  [cache_slip_hardening]
    type = CopyScalar
    from = 'state/internal/slip_hardening'
    to = 'forces/tmp/internal/slip_hardening'
  []
  [cache1]
    type = ComposedModel
    models = 'cache_elastic_strain cache_slip_hardening'
  []
  [cache2]
    type = CopyWR2
    from = 'state/orientation'
    to = 'forces/tmp/orientation'
  []
  [model]
    type = ComposedModel
    models = 'cache2 subsystem1 cache1 subsystem2'
    priority = 'cache2 subsystem1 cache1 subsystem2'
    additional_outputs = 'state/elastic_strain state/internal/slip_hardening'
  []
[]
```
It is undoubtedly difficult to understanding the model composition just by reading the input file. To address such challenge, the `visualization` module is provided as part of the NEML2 Python package to help extract, plot, and customize the dependency graph of a composed model. The generated dependency graph provides a visual understanding of the model composition, helps identify otherwise-difficult-to-find issues, and opens up opportunities for optimizing of the composition.

## Rendering non-composed model

NEML2 relies on the Python interface of [Graphviz](https://graphviz.readthedocs.io/en/stable/index.html) to render the dependency graph. To visualize a basic non-composed model, simply load the model from the input file and pass it to the `neml2.visualization.render` function.
```python
import neml2
from neml2.visualization import render

model = neml2.load_model("input.i", "euler_rodrigues_1")
render(model, outfile="graph.svg")
```
The `render` function calls [`graphviz.render`](https://graphviz.readthedocs.io/en/stable/api.html#graphviz.render) behind the scenes, and additional arguments and keyword arguments are forwarded without modification. By default, an intermediate graphviz source file (written in the DOT language) with suffix `.gz` and the rendered output specified by `outfile` are generated. Customization of the graph will be discussed later. NEML2 provides some sensible default settings which generates the graph below.

![Graph of a basic model](asset/graph_basic.svg){html: width=25%}

There are three components in the above image:
- The input variable represented as a gray rectangular node
- The model represented as a blue rectangular node
- The output variable represented as a green rectangular node

Additional information are also labeled on each node enclosed within square brackets, such as the variable type (e.g. `[Rot]`, `[R2]`) and the model type (e.g. `[RotationMatrix]`). Note that the rounded dashed border around the variables denote sub-axes, and the sub-axis name is also labeled. Edges (arrows) in this directed graph represent information flow, i.e., the input variable is passed into the model, the model performs a set of operations and produces the output variable.

## Rendering composed model

The dependency graph of a composed model can also be visualized using a similar routine,
```python
import neml2
from neml2.visualization import render

model = neml2.load_model("input.i", "implicit_rate_1")
render(model, outfile="graph.svg")
```
Note the model name is changed to "implicit_rate_1" which is a composed model. The generated graph is shown below.

![Graph of a composed model](asset/graph_composed.svg){html: width=85%}

Each shaded area represents a sub-model in the composed model. When graph constraints allow, the input variables of the composed model are shown at the very top, and the output variables of the composed model are shown at the very bottom.

\note
The svg and pdf output formats allow arbitrary zooming without loss of resolution.

For Visual Studio Code users, the [Graphviz Interactive Preview](https://marketplace.visualstudio.com/items?itemName=tintinweb.graphviz-interactive-preview) extension can be used to directly preview the graphviz source code. The extension supports useful postprocessing operations such as node highlighting and filtering. For example, the following screenshot demonstrates upstream highlight of an intermediate output variable `state/internal/sum_slip_rates`.

![Graph of a composed model with upstream highlighting](asset/graph_highlight.svg){html: width=85%}

## Customizing graph style

The generated graph is highly customizable. All controllable attributes are available in the data class `Configuration`. The configuration can be modified with an instance of the data class, i.e.
```python
import neml2
from neml2.visualization import render, Configuration

model = neml2.load_model("input.i", "implicit_rate_1")

config = Configuration()
config.global_attributes["fontname"] = "Times-Roman"
config.global_node_attributes["fontname"] = "Arial"
config.global_edge_attributes["arrowhead"] = "empty"
config.submodel_name_node_attributes["color"] = "#bf0036"
config.submodel_name_node_attributes["fillcolor"] = "#bf003650"
config.input_node_attributes["color"] = "#00540150"
config.input_node_attributes["fillcolor"] = "#00540115"
config.output_node_attributes["color"] = "#8800ff50"
config.output_node_attributes["fillcolor"] = "#8800ff15"

render(model, outfile="graph.svg")
```
The font names, arrow head style, and node fill colors are modified from their default values. The generated graph is shown below.

![Graph of a composed model with custom style](asset/graph_custom.svg){html: width=85%}

@insert-page-navigation
