#include "neml2/models/solid_mechanics/crystal_plasticity/MeanFreePath.h"
#include "neml2/tensors/functions/sqrt.h"
#include "neml2/tensors/functions/pow.h"

namespace neml2
{
register_NEML2_object(MeanFreePath);

OptionSet
MeanFreePath::expected_options()
{
    OptionSet options = Model::expected_options();
    options.set<bool>("use_L2") = true;
    options.set<bool>("use_L3") = true;
    options.set_input("rho_m");
    options.set<TensorName<Scalar>>("c_lath");
    options.set<TensorName<Scalar>>("d_lath");
    options.set<TensorName<Scalar>>("c_block");
    options.set<TensorName<Scalar>>("d_block");
    options.set<TensorName<Scalar>>("c_packet");
    options.set<TensorName<Scalar>>("d_packet");
    options.set<TensorName<Scalar>>("c_PAG");
    options.set<TensorName<Scalar>>("d_PAG");
    options.set<TensorName<Scalar>>("c_MX");
    options.set<TensorName<Scalar>>("d_MX");
    options.set<TensorName<Scalar>>("c_M23C6");
    options.set<TensorName<Scalar>>("d_M23C6");
    options.set_output("L");

    return options;
}
MeanFreePath::MeanFreePath(const OptionSet & options) : Model(options),
    _use_L2(options.get<bool>("use_L2")),
    _use_L3(options.get<bool>("use_L3")),
    _rho_m(declare_input_variable<Scalar>("rho_m")),
    _L(declare_output_variable<Scalar>("L"))
{
    if (_use_L2)
    {
        _c_lath     = &declare_parameter<Scalar>("c_lath", "c_lath");
        _d_lath     = &declare_parameter<Scalar>("d_lath", "d_lath");
        _c_block    = &declare_parameter<Scalar>("c_block", "c_block");
        _d_block    = &declare_parameter<Scalar>("d_block", "d_block");
        _c_packet   = &declare_parameter<Scalar>("c_packet", "c_packet");
        _d_packet   = &declare_parameter<Scalar>("d_packet", "d_packet");
        _c_PAG      = &declare_parameter<Scalar>("c_PAG", "c_PAG");
        _d_PAG      = &declare_parameter<Scalar>("d_PAG", "d_PAG");
    }
    
    if (_use_L3)
    {
        _c_MX = &declare_parameter<Scalar>("c_MX", "c_MX");
        _d_MX = &declare_parameter<Scalar>("d_MX", "d_MX");
        _c_M23C6 = &declare_parameter<Scalar>("c_M23C6", "c_M23C6");
        _d_M23C6 = &declare_parameter<Scalar>("d_M23C6", "d_M23C6");
        
    }
}

void
MeanFreePath::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
    const auto sqrt_rho_m       = neml2::sqrt(_rho_m());
    auto inv_L                  = sqrt_rho_m;

    if (_use_L2)
    {
        const auto inv_L2       = (*_c_lath) / (*_d_lath) + (*_c_block) / (*_d_block) + (*_c_packet) / (*_d_packet) + (*_c_PAG) / (*_d_PAG);
        inv_L                   = inv_L + inv_L2;
    }

    if (_use_L3)
    {
        const auto inv_L3       = (*_c_MX) / (*_d_MX) + (*_c_M23C6) / (*_d_M23C6);
        inv_L                   = inv_L + inv_L3;
    }
    
    const auto L = 1.0 / inv_L;

    if (out)
    {
        _L = L;
    }

    if (dout_din)
    {   
        //   Let S = inv_L = sqrt(rho_m) + [const L2 terms] + [const L3 terms]
        //   L = 1/S
        const auto dS_drho_m = 1 / (2.0 * neml2::sqrt(_rho_m()));
        const auto dL_drho_m = neml2::pow(-L, 2.0) * dS_drho_m;

        if (_rho_m.is_dependent())
            _L.d(_rho_m) = dL_drho_m;
    }

}
}
