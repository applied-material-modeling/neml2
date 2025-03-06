@insert-title:tutorials-optimization

The previous tutorials have illustrated the use of NEML2 constitutive models in the "feed-forward" setting, i.e., the model maps from input variables to output variables with a given parametrization, i.e.
\f[
  y = f(x; p, b).
\f]
Recall that \f$p\f$ and \f$b\f$ are respectively the parameters and the buffers of the model.

Another interesting use of NEML2 constitutive models is *parameter calibration*: With given input variables \f$x\f$, find the optimal parameter set \f$p^*\f$ such that
\f[
  p^* = \mathop{\mathrm{argmin}}\limits_{p} l(f(x; p, b)),
\f]
where \f$l\f$ is oftentimes referred to as the loss (or objective) function defining optimality.

This set of tutorials demonstrate the use of PyTorch [Autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) to calculate parameter derivatives (\f$\pdv{l}{p}\f$) which is a necessary ingredient in all gradient-based optimizers.

@insert-subsection-list
