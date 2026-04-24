@insert-title:tutorials-models-input-file

[TOC]

The user interface of NEML2 is designed in such a way that no programing experience is required to compose custom material models and define how they are solved. This is achieved using _input files_. The input files are simply text files with a specific format that NEML2 can understand. NEML2 can _deserialize_ an input file, i.e., parse and create material models specified within the input file.

Since the input files are nothing more than text files saved on the disk, they can be used in any application that supports standard IO, easily exchanged among different devices running different operating systems, and archived for future reference. Refer to [the format documentation](https://github.com/applied-material-modeling/neml2-hit) for full details.

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
defines a tensor named "E" under the `[Tensors]` block and a model named "elasticity" under the `[Models]` block. Notice that an object (in this case the tensor named "E" under the `[Tensors]` section) can be referenced by an input option (in this case "coefficients") by its name, and this mechanism is referred to as cross-referencing. The [Syntax Documentation](@ref syntax-tensors) provides a complete list of objects that can be defined by an input file. The [System Documentation](@ref system-tensors) provides detailed explanation of each system.

\note
The ordering of objects, i.e., the sequence objects appear in the input file, does not change their behavior.

## Custom types

In addition to the native value types supported by the HIT format, such as integer, floating-point number, boolean, strings, and arrays, NEML2 defines a few custom value types listed below.

**Tensor shape**: A tensor shape must start with "(" and end with ")". An array of comma-separated integers must be enclosed by the parentheses. For example, "(5,6,7)" can be parsed into a shape tuple of value `(5, 6, 7)`. Note that white spaces are not allowed between the parentheses and could lead to undefined behavior. An empty array, i.e. "()", however, is allowed and fully supported. Refer to @ref tutorials-tensors for a more in-depth description of tensor shape.

**Device**: A device string must follow the schema: `(type)[:<device-index>]`, where type can be cpu, cuda, etc., and `:<device-index>` optionally specifies a device index. Some examples are "cpu", "cuda", "cuda:0", "cuda:1", "xpu", etc.

@insert-page-navigation
