#include "neml2/models/solid_mechanics/crystal_plasticity/AthermalStress.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(AthermalStress);

OptionSet
AthermalStress::expected_options()
{
    OptionSet options = Model::expected_options();
    options.set_parameter<TensorName<Scalar>>("shear_modulus");
    options.set_parameter<TensorName<Scalar>>("alpha");
    options.set_parameter<TensorName<Scalar>>("b");
    options.set_input("L");
    options.set_output("athermal_stress");

    return options;
}
AthermalStress::AthermalStress(const OptionSet & options) : Model(options),
    _G(declare_parameter<Scalar>("shear_modulus", "shear_modulus", true)),
    _alpha(declare_parameter<Scalar>("alpha", "alpha")),
    _b(declare_parameter<Scalar>("b", "b")),
    _L(declare_input_variable<Scalar>("L")),
    _sigma_a(declare_output_variable<Scalar>("athermal_stress"))
{
}
void
AthermalStress::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{   
    auto sigma_a = (_alpha * _G * _b) / _L();

    if (out)
        _sigma_a = sigma_a;
    
    if (dout_din)
    {   
        if (_L.is_dependent())
            _sigma_a.d(_L) = -(_alpha * _G * _b) / neml2::pow(_L(), 2.0);
        
        if (const auto * const G = nl_param("shear_modulus"))
            _sigma_a.d(*G) = (_alpha * _b) / _L();

        if (const auto * const b = nl_param("b"))
            _sigma_a.d(*b) = (_alpha * _G) / _L();
    
        if (const auto * const alpha = nl_param("alpha"))
            _sigma_a.d(*alpha) = (_G * _b) / _L();
    }
}
}