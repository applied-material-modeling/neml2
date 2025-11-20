@insert-title:tutorials-extension-connection-to-input-files

[TOC]

NEML2 utilizes the well-established [registry-factory pattern](https://en.wikipedia.org/wiki/Factory_method_pattern) to allow for runtime object creation and distributed class registration. In NEML2, neml2::Parser, neml2::Registry, and neml2::Factory work together to create objects defined in the input file.

## Parser

A class is said to be runtime-manufacturable if its constructor can be retrieved at runtime using a string identification of its symbol. To allow for a unified object creation signature, runtime-manufacturable objects (e.g., models) are required to derive from neml2::NEML2Object and provide a constructor and a static method with the following signatures

@list:cpp:connection_to_input_files/ex1.cxx:signature

- The static method `expected_options` allows the NEML2 registry to record required input options *expected* from the user in order to create an instance of the class.
- The constructor is responsible for constructing an instance given input options *extracted* from the input file.
- The parser is responsible for bridging the gap between the expected input options and the user-specified options.

The following steps take place for each object during input file parsing:
1. Parser extracts object type from the reserved field `type`.
2. Parser retrieves *expected* input options from the registry.
3. Parser iterates over *user-specified* options and override default options.

At the end of this process, a set of input options are gathered for the subsequent object creation.

All of the gathered options are stored in the NEML2 factory so that objects can be created at runtime per the user's request.

## Input file

Before defining the input file options for our custom model, it is a good idea to first design the syntax. For example, in this equation we are mapping from the projectile's current velocity to its acceleration, therefore we'd like to allow user to specify the variable names corresponding to the velocity and acceleration. In addition, since the equation is parametrized by the gravitational acceleration \f$\boldsymbol{g}\f$ and the dynamic viscosity \f$\nu\f$, we should allow user to set their values directly inside the input file (without modifying the source code).

Therefore, the input file syntax should look something like
@list-input:connection_to_input_files/input.i:Models/accel

By going through this exercise, we have effectively identified the *expected* input file options:
@list:cpp:connection_to_input_files/ex1.cxx:expected_options

## Registry

The registry is a [RAII-style](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization) singleton responsible for storing the mapping from the string identification of each runtime-manufacturable object to the constructor pointer along with the expected input file options.

The registration is accomplished by the static method neml2::Registry::add templated on the class type. The static registration method combined with the singleton enables distributed class registration in each translation unit. The convenience macro register_NEML2_object and its variants wrap around neml2::Registry::add to provide syntatic simplification of the registration call.

An example usage of the macro is
@list:cpp:connection_to_input_files/ex1.cxx:registration


Note that a dummy static variable (unique to the class typename) is created to prevent duplicate registration.

## Factory

The NEML2 factory handles the creation of objects at runtime. The generic API for object creation and retrieval is neml2::Factory::get_object.

In our example, the parser will first extract input file options from the input file and use them to fill out the expected options (defined by the `ProjectileAcceleration::expected_options()` method). The factory then passes the collected options to the constructor to create this object. The constructor should accept an neml2::OptionSet and define how the collected options are used.
@list:cpp:connection_to_input_files/ex1.cxx:constructor

The following tutorials will explain how to declare variables and parameters in the constructor.

Suppose an input file named "input.i" has the following content
@list-input:connection_to_input_files/input.i

To verify that our custom model is correctly accepting these input file options, let us use the factory to create this model and examine the option values.

@list:cpp:connection_to_input_files/ex1.cxx:main

Output:
@list-output:ex1

@insert-page-navigation
