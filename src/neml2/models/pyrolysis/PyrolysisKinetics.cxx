// Copyright 2024, UChicago Argonne, LLC
// All Rights Reserved
// Software Name: NEML2 -- the New Engineering material Model Library, version 2
// By: Argonne National Laboratory
// OPEN SOURCE LICENSE (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "neml2/models/pyrolysis/PyrolysisKinetics.h"
#include "neml2/tensors/functions/exp.h"

namespace neml2
{
register_NEML2_object(PyrolysisKinetics);
OptionSet
PyrolysisKinetics::expected_options()
{
  OptionSet options = Model::expected_options();
  options.doc() = "Governing Pyrolysis Kinetics equations, takes the form of \\f$ x = A \\exp "
                  "\\left( -\\frac{Ea}{RT} \\right)f \\f$, where "
                  "\\f$ A \\f$ is the kinetics constant, \\f$ Ea \\f$ is the activation energy, "
                  "\\f$ R \\f$ is the ideal gas constant, \\f$ T \\f$ is the temperature, and \\f$ "
                  "f \\f$ is the reaction models";

  options.set_parameter<TensorName<Scalar>>("kinetic_constant");
  options.set("kinetic_constant").doc() = "Kinetic constant, A";

  options.set_parameter<TensorName<Scalar>>("activation_energy");
  options.set("activation_energy").doc() = "Activation energy, Ea";

  options.set_parameter<TensorName<Scalar>>("ideal_gas_constant");
  options.set("ideal_gas_constant").doc() = "Ideal gas constant, R";

  options.set_input("temperature") = VariableName(FORCES, "temperature");
  options.set("temperature").doc() = "Temperature of the reactions";

  options.set_input("reaction") = VariableName("state", "reaction");
  options.set("reaction").doc() = "Reaction function, f";

  options.set_output("out") = VariableName("state", "out");
  options.set("out").doc() = "Kinetics output, x.";

  return options;
}

PyrolysisKinetics::PyrolysisKinetics(const OptionSet & options)
  : Model(options),
    _A(declare_parameter<Scalar>("A", "kinetic_constant")),
    _Ea(declare_parameter<Scalar>("Ea", "activation_energy")),
    _R(declare_parameter<Scalar>("R", "ideal_gas_constant")),
    _T(declare_input_variable<Scalar>("temperature")),
    _f(declare_input_variable<Scalar>("reaction")),
    _x(declare_output_variable<Scalar>("out"))
{
}

void
PyrolysisKinetics::set_value(bool out, bool dout_din, bool d2out_din2)
{
  neml_assert_dbg(!d2out_din2, "Second derivative not implemented.");

  if (out)
  {
    _x = _A * exp(-_Ea / (_R * _T)) * _f;
  }

  if (dout_din)
  {
    _x.d(_T) = _A * exp(-_Ea / (_R * _T)) * _f * (_Ea / (_R * _T * _T));
    _x.d(_f) = _A * exp(-_Ea / (_R * _T));
  }
}
}