@insert-title:tutorials-next-steps

Congratulations on completing all tutorials! \emoji wink

Here, we list some additional resources that can come in handy as you integrate NEML2 into your own applications/workflows.

- [Syntax documentation](#syntax-models) is a catalog for all available objects that can be created using input files, including tensors, models, solvers, drivers, etc. The documentation of each object provides a brief description, followed by a list of input file options. Each option has a short description right next to it, and can be expanded to show additional details. We recommend browsing this catalog before implementing custom models to avoid re-inventing the wheels.
- **Physics documentation** contains high-level overviews of each specific physics. For example, the [Solid Mechanics](#solid-mechanics) physics documentation provides an overview of available building blocks for solid materials, as well as material models that can be composed using these building blocks, covering a wide variety of elasticity and plasticity models.
- [pyzag](https://github.com/applied-material-modeling/pyzag) is a library for efficiently training generic models defined with a recursive nonlinear function. Although the basics of model calibration was introduced in [this](#tutorials-optimization-parameter-calibration) tutorial, we still highly recommend interested users to checkout pyzag which performs model training in a much more efficient fashion. pyzag fully supports NEML2 models. Its full documentation can be found [here](https://applied-material-modeling.github.io/pyzag/).

Last but not the least, **we'd love to hear your feedback!** During NEML2's early and rapid development stage, your feedback will steer the direction of future development.
- Questions and general discussions can be posted on the [Discussion](https://github.com/applied-material-modeling/neml2/discussions) tab.
- Bug reports and feature requests can be submitted at the [Issues](https://github.com/applied-material-modeling/neml2/issues) tab.
- To propose changes or contribute back to the NEML2 project, [pull requests](https://github.com/applied-material-modeling/neml2/pulls) are always welcome! The [Contributing](#tutorials-contributing) guide is a good place to start.
- For collaboration, please contact the lead developers
  - Mark Messner, [messner@anl.gov](mailto:messner@anl.gov)
  - Gary Hu, [thu@anl.gov](mailto:thu@anl.gov)

