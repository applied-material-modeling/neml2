@insert-title:tutorials-extension-argument-declaration

[TOC]

Recall that all NEML2 models can be written in the following form
\f[
  y = f(x; p, b).
\f]

Defining a custom model therefore involves
1. declaring the input variables \f$x\f$,
2. declaring the output variables \f$y\f$,
3. declaring the parameters (and buffers, if any), and
4. defining the forward operator \f$f\f$.

The first three are introduced in this tutorial.

## Variable declaration

The API neml2::Model::declare_input_variable can be used to declare input variables. This method should be called in the constructor of the model. It allocate storage for the underlying tensor and returns a const-reference i.e., `const Variable<T> &`, to the variable, where `T` is the variable tensor type. Similarly, the method neml2::Model::declare_output_variable is for declaring output variables which returns a non-const-reference i.e., `Variable<T> &`, to the variable.

In our example, the only input variable is \f$\boldsymbol{v}\f$, and the only output variable is \f$\boldsymbol{a}\f$. To declare them, first create the corresponding members in the class like
```cpp
// Velocity of the projectile (input variable)
const Variable<Vec> & _v;

// Acceleration of the projectile (output variable)
Variable<Vec> _a;
```
Then, use the variable declaration method in the constructor, i.e.
```cpp
namespace neml2
{
ProjectileAcceleration::ProjectileAcceleration(const OptionSet & options)
  : Model(options),
    _v(declare_input_variable<Vec>("velocity")),
    _a(declare_output_variable<Vec>("acceleration"))
{
}
}
```
NEML2 will look for an input option named "velocity" and declare a vector-valued input variable with the user-specified variable name. Similarly, another input option named "acceleration" will be used to declare a vector-valued output variable.

## Parameter declaration

Parameter (and buffer) declaration is semantically similarly to variable declaration.
- neml2::Model::declare_parameter is used to declare parameters, and
- neml2::Model::declare_buffer is used to declare buffers.

Recall that both parameters and buffers "travel with the model", meaning that when a model is sent to a different device, so are its parameters and buffers. The difference is that parameters are *trainable*. Therefore, in our example, we'd like to treat the dynamic viscosity as a trainable parameter and the gravitational acceleration as a buffer.

First, create the corresponding members in the class:
```cpp
// Gravitational acceleration (parameter)
const Vec & _g;

// Dynamic viscosity of the medium (parameter)
const Scalar & _mu;
```
Then, call the declaration methods in the constructor, i.e.
@source:src1
```cpp
namespace neml2
{
ProjectileAcceleration::ProjectileAcceleration(const OptionSet & options)
  : Model(options),
    _v(declare_input_variable<Vec>("velocity")),
    _a(declare_output_variable<Vec>("acceleration")),
    _g(declare_buffer<Vec>("g", "gravitational_acceleration")),
    _mu(declare_parameter<Scalar>("mu", "dynamic_viscosity"))
{
}
}
```
@endsource
The first argument of the parameter/buffer declaration method takes a string specifying the name of the parameter/buffer used internally. These names will then become the "handles" for retrieving and updating parameter/buffer values.

NEML2 will look for input options named "gravitational_acceleration" and "dynamic_viscosity" and declare the corresponding parameter and buffer.

## Model inspection

Once all arguments are successfully declared, the *structure* of the model is determined. NEML2 handles the allocation and arrangement of variables, parameters, buffers, and their underlying tensors. The structure of the model can be inspected by printing out the summary:
@source:src1
```cpp
int
main()
{
  auto & model = neml2::load_model("input.i", "accel");
  std::cout << model << std::endl;
}
```
@endsource

Output:
```
@attach-output:src1
```

@insert-page-navigation
