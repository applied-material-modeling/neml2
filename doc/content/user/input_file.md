# Input File {#input-file}

[TOC]

The user interface of NEML2 is designed in such a way that no programing experience is required to compose custom material models and define how they are solved. This is achieved using _input files_. The input files are simply text files with a specific format that NEML2 can understand. NEML2 can _deserialize_ an input file, i.e., parse and create material models specified within the input file.

Since the input files are nothing more than text files saved on the disk, they can be used in any application that supports standard IO, easily exchanged among different devices running different operating systems, and archived for future reference.

## Syntax {#input-file-syntax}

Input files use the Hierarchical Input Text (HIT) format. The syntax looks like this:
```python
# Comments look like this
[block1]
  # This is a comment
  foo = 1
  bar = 3.14159
  baz = 'string value'
  [nested_block]
    # ...
  []
[]
```
where key-value pairs are defined under (nested) blocks denoted by square brackets. A value can be an integer, floating-point number, string, or array (as indicated by single quotes). Note that the block indentation is recommended for clarity but is not required.

All NEML2 capabilities that can be defined through the input file fall under a number of _systems_. Names of the top-level blocks specify the systems. For example, the following input file
```python
[Tensors]
  [E]
    # ...
  []
[]

[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = 'E 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
```
defines a tensor named "E" under the `[Tensors]` block and a model named "elasticity" under the `[Models]` block. Notice that an object (in this case the tensor named "E" under the `[Tensors]` section) can be referenced by an input option (in this case "coefficients") by its name, and this mechanism is referred to as [cross-referencing](@ref cross-referencing). The [Syntax Documentation](@ref syntax-tensors) provides a complete list of objects that can be defined by an input file. The [System Documentation](@ref system-tensors) provides detailed explanation of each system.

\note
The ordering of objects, i.e., the sequence objects appear in the input file, does not change their behavior.

## Special syntax

**Boolean**: Oftentimes the behavior of the object is preferrably controlled by a boolean flag. However, since the HIT format only allows (array of) integer, floating-point number, and string, a special syntax shall be reserved for boolean values. In NEML2 input files, a string with value "true" can be parsed into a boolean `true`, and a string with value "false" can be parsed into a boolean `false`.

\note
On the other hand, other commonly used boolean flags such as "on"/"off", "1"/"0", "True"/"False", etc., cannot be parsed into boolean values.

**Variable name**: NEML2 material models work with named variables to assign physical meanings to different slices of a tensor. A fully qualified variable name can be parsed from a string, and the delimiter "/" signifies nested sub-axes. For example, the string "forces/t" can be parsed into a variable named "t" defined on the sub-axis named "forces".

**Tensor shape**: Shape of a tensor can also be parsed from a string. The string must start with "(" and end with ")". An array of comma-separated integers must be enclosed by the parentheses. For example, the string "(5,6,7)" can be parsed into a shape tuple of value `(5, 6, 7)`. Note that white spaces are not allowed between the parentheses and could lead to undefined behavior. An empty array, i.e. "()", however, is allowed and fully supported.

## Cross-Referencing {#cross-referencing}

The cross-referencing mechanism allows object options in the input file to _reference_ another object by name. All manufacturable objects, i.e., objects listed in the [syntax documentation](@ref syntax-tensors), can be referenced by name. An example of referencing object by name is shown below.
```python
[Solvers]
  [newton]
    type = Newton
  []
[]

[Models]
  # Other objects defining 'implicit_model' are omitted for brevity
  [model]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
[]
```
In the above example, the object of type `ImplicitUpdate` references an implicit model named "implicit_rate" and a solver named "newton".

In addition to directly referencing objects by their names, a few special types support more flexible referencing mechanisms. `ATensor`, `Tensor`, and all primitive tensor types with fixed base shapes such as `Scalar`, `SR2`, etc, can be referenced either by value or by name.

When a tensor is referenced by value, the parser will parse the input option value as a numeric literal and return a tensor filled with the specified value; when a tensor is referenced by name, the parser will look for and return the object under the `[Tensors]` section with the given name.

In the following example,
```python
[Tensors]
  [p0]
    type = LinspaceScalar
    start = 1
    end = 10
    nstep = 5
  []
[]

[Models]
  [p]
    type = ArrheniusParameter
    temperature = 'forces/T'
    reference_value = 'p0'
    activation_energy = 1e3
    ideal_gas_constant = 8.314
  []
[]
```
option "reference_value" is referenced by its name "p0", and hence an object with the corresponding name must exist under the `[Tensors]` section. The option "activation_energy" is referenced by value, and a `Scalar` filled with the given value `1e3` is returned.
