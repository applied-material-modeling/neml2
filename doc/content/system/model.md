# Model {#system-models}

[TOC]

Refer to [Syntax Documentation](@ref syntax-models) for the list of available objects.

## Model definition {#model-definition}

A NEML2 model is a function (in the context of mathematics)
\f[
  f: \mathbb{R}^m \to \mathbb{R}^n
\f]
mapping from the input space \f$\mathbb{R}^m\f$ of dimension \f$m\f$ to the output space \f$\mathbb{R}^n\f$ of dimension \f$n\f$. \f$\left[ \cdot \right]\f$ be the flatten-concatenation operator, the input vector is the concatenation of \f$p\f$ flattened variables, i.e.,
\f[
  x = \left[ x_i \right]_{i=1}^p \in \mathbb{R}^m, \quad \sum_{i=1}^p \lvert x_i \rvert = m,
\f]
where \f$\lvert x \rvert\f$ denotes the modulus of flattened variable \f$x\f$. Similarly, the output vector is the concatenation of \f$q\f$ flattened variables, i.e.,
\f[
  y = \left[ y_i \right]_{i=1}^q \in \mathbb{R}^n, \quad \sum_{i=1}^q \lvert y_i \rvert = n.
\f]

Translating the above mathematical definition into NEML2 is straightforward.
- A model following this definition derives from [Model](@ref neml2::Model).
- [declare_input_variable](@ref neml2::Model::declare_input_variable) declares an input variable \f$x_i\f$ in the input space \f$\mathbb{R}^m\f$.
- [declare_output_variable](@ref neml2::Model::declare_output_variable) declares an output variable \f$y_i\f$ in the output space \f$\mathbb{R}^n\f$.
- [set_value](@ref neml2::Model::set_value) is a method defining the forward operator \f$f\f$ itself.

Both `declare_input_variable` and `declare_output_variable` are templated on the variable type -- recall that only a variable of the NEML2 primitive tensor type can be registered. Furthermore, both methods return a `Variable<T> &` used for retrieving and setting the variable value inside the forward operator, i.e. `set_value`. Note that the reference returned by `declare_input_variable` is writable, while the reference returned by `declare_output_variable` is read-only.

## Model composition {#model-composition}

Quoting [Wikipedia](https://en.wikipedia.org/wiki/Function_composition):
> In mathematics, function composition is an operation \f$\circ\f$ that takes two functions \f$f\f$ and \f$g\f$, and produces a function \f$h = g \circ f\f$ such that \f$h(x) = g(f(x))\f$.

Since NEML2 `Model` is a function (in the context of mathematics) at its core, it should be possible, in theory, to compose different NEML2 `Model`s into a new NEML2 `Model`. The [ComposedModel](@ref neml2::ComposedModel) is precisely for that purpose.

Similar to the statement "a composed function is a function" in the context of mathematics, in NEML2, the equivalent statement "a `ComposedModel` is a `Model`" also holds. In addition, the `ComposedModel` provides four key features to help simplify the composition and reduces computational cost:
- Automatic dependency registration
- Automatic input/output identification
- Automatic dependency resolution
- Automatic chain rule

### A symbolic example {#a-symbolic-example}

To demonstrate the utility of the four key features of `ComposedModel`, let us consider the composition of three functions \f$f\f$, \f$g\f$, and \f$h\f$:
\f{align*}
  y_1 &= f(x_1, x_2), \\
  y_2 &= g(y_1, x_3), \\
  y &= h(y_1, y_2, x_4).
\f}

### Automatic dependency registration {#automatic-dependency-registration}

It is obvious to us that the function \f$h\f$ _depends_ on functions \f$f\f$ and \f$g\f$ because the input of \f$h\f$ depends on the outputs of \f$f\f$ and \f$g\f$. Such dependency is automatically identified and registered while composing a `ComposedModel` in NEML2. This procedure is called "automatic dependency registration".

In order to identify dependencies among different `Model`s, we keep track of the set of _consumed_ variables, \f$\mathcal{I}_i\f$, and a set of _provided_ variables, \f$\mathcal{O}_i\f$, for each `Model` \f$f_i\f$. When a set of models (functions) are composed together, `Model` \f$f_i\f$ is said to _depend_ on \f$f_j\f$ if and only if \f$\exists x\f$ such that
\f[
  x \in \mathcal{I}_i \wedge x \in \mathcal{O}_j.
\f]

### Automatic input/output identification {#automatic-input-output-identification}

The only possible composition \f$r\f$ of these three functions is
\f[
  y = r(x_1, x_2, x_3, x_4) := h(f(x_1, x_2), g(f(x_1, x_2), x_3), x_4).
\f]
The input variables of the composed function \f$r\f$ are \f$[x_1, x_2, x_3, x_4]\f$ (or their flattened concatenation), and the output variable of the composed function is simply \f$y\f$. The input/output variables are automatically identified while composing a `ComposedModel` in NEML2. This procedure is referred to as "automatic input/output identification".

In a `ComposedModel`, a _leaf_ model is a model which does not depend on any other model, and a _root_ model is a model which is not depent upon by any other model. A `ComposedModel` may have multiple leaf models and multiple root models. An input variable is said to be a _root_ input variable if it is not consumed by any other model, i.e. \f$x \in \mathcal{I}_i\f$ is a root input variable if and only if
\f[
    x \notin \mathcal{O}_j, \quad \forall i \neq j.
\f]
Similarly, an output variable is said to be a _leaf_ output variable if it is not provided by any other model, i.e. \f$x \in \mathcal{O}_i\f$ is a leaf output variable if an only if
\f[
    x \notin \mathcal{I}_j, \quad \forall i \neq j.
\f]
The input variables of a `ComposedModel` is the union of the set of all the root input variables, and the output variables of a `ComposedModel` is the set of all the leaf output variables.

### Automatic dependency resolution {#automatic-dependency-resolution}

To evaluate the forward operator of the composed model \f$r\f$, one has to first evaluate model \f$f\f$, then model \f$g\f$, and finally model \f$h\f$. The process of sorting out such evaluation order is called "dependency resolution".

While it is possible to sort the evaluation order "by hand" for this simple example composition, it is generally not a trivial task for practical compositions with more involved dependencies. To that end, NEML2 uses [topological sort](https://en.wikipedia.org/wiki/Topological_sorting) to sort the model evaluation order, such that by the time each model is evaluated, all of its dependent models have already been evaluated.

### Automatic chain rule {#automatic-chain-rule}

Chain rule can be applied to evaluate the derivative of the forward operator with respect to the input variables, i.e.,
\f{align*}
  \frac{\partial y}{\partial x_1} &= \left( \frac{\partial y}{\partial y_1} + \frac{\partial y}{\partial y_2} \frac{\partial y_2}{\partial y_1} \right) \frac{\partial y_1}{\partial x_1}, \\
  \frac{\partial y}{\partial x_2} &= \left( \frac{\partial y}{\partial y_1} + \frac{\partial y}{\partial y_2} \frac{\partial y_2}{\partial y_1} \right) \frac{\partial y_1}{\partial x_2}, \\
  \frac{\partial y}{\partial x_3} &= \frac{\partial y}{\partial y_2} \frac{\partial y_2}{\partial x_3}, \\
  \frac{\partial y}{\partial x_4} &= \frac{\partial y}{\partial x_4}.
\f}
Spelling out this chain rule can be cumbersome and error-prone, especially for more complicated model compositions. The evaluation of the chain rule is automated in NEML2, and the user is only responsible for implementing the partial derivatives of each model. For example, in the implementation of `Model` \f$f\f$, the user only need to define the partial derivatives
\f[
  \frac{\partial y_1}{\partial x_1}, \quad \frac{\partial y_1}{\partial x_2};
\f]
similarly, `Model` \f$g\f$ only defines
\f[
  \frac{\partial y_2}{\partial y_1}, \quad \frac{\partial y_2}{\partial x_3}
\f]
and `Model` \f$h\f$ only defines
\f[
  \frac{\partial y}{\partial y_1}, \quad \frac{\partial y}{\partial y_2}, \quad \frac{\partial y}{\partial x_4}.
\f]
The assembly of the partial derivatives into the total derivative \f$\partial y / \partial \boldsymbol{x}\f$ using the chain rule is handled by NEML2. This design serves as the fundation for a modular model implementation:
- Each model _does not_ need to know its composition with others.
- The same model partial derivatives can be reused in _any_ composition.

## Automatic differentiation {#automatic-differentiation}

Deriving and implementing derivatives of the forward operator can be cubersome from times to times. NEML2 offers the option to use automatic differentiation to obtain derivatives. To enable automatic differentiation, simply set the `_use_AD_first_derivative` and/or the `_use_AD_second_derivative` options to `true`.

Since a composed model uses chain rule to efficiently evaluate the total derivatives, automatic differentiation is disabled for `ComposedModel`. However, each of the child model can still use AD to calculate the derivatives of its own forward operator. Moreover, AD and non-AD models can be composed together.
