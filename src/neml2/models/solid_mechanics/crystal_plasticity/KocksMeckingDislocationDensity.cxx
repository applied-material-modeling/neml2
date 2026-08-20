#include "neml2/models/solid_mechanics/crystal_plasticity/KocksMeckingDislocationDensity.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/exp.h"

namespace neml2
{
register_NEML2_object(KocksMeckingDislocationDensity);

OptionSet
KocksMeckingDislocationDensity::expected_options()
{
    OptionSet options = Model::expected_options();
    options.set_input("plastic_flow_rate");
    options.set_parameter<TensorName<Scalar>>("k1");
    options.set_input("L");
    options.set_parameter<TensorName<Scalar>>("k2_0");
    options.set_parameter<TensorName<Scalar>>("Q_d");
    options.set_buffer<TensorName<Scalar>>("k_B");
    options.set_input("temperature");
    options.set_input("dislocation_density");
    options.set_output("density_rate");

    return options;
}
KocksMeckingDislocationDensity::KocksMeckingDislocationDensity(const OptionSet & options) : Model(options),
    _p_dot(declare_input_variable<Scalar>("plastic_flow_rate")),
    _k1(declare_parameter<Scalar>("k1", "k1", true)),
    _L(declare_input_variable<Scalar>("L")),
    _k2_0(declare_parameter<Scalar>("k2_0", "k2_0", true)),
    _Q_d(declare_parameter<Scalar>("Q_d", "Q_d", true)),
    _k_B(declare_buffer<Scalar>("k_B", "k_B")),
    _T(declare_input_variable<Scalar>("temperature")),
    _rho_m(declare_input_variable<Scalar>("dislocation_density")),
    _rho_m_dot(declare_output_variable<Scalar>("density_rate"))
{
}
void
KocksMeckingDislocationDensity::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{   
    const auto k2 = _k2_0 * neml2::exp(- _Q_d / (_k_B * _T()));

    if (out)
        _rho_m_dot = (_k1/_L() - k2 * _rho_m()) * _p_dot();

    if (dout_din)
    {
        if (_p_dot.is_dependent())
            _rho_m_dot.d(_p_dot) = _k1/_L() - k2 * _rho_m();
        
        if (_L.is_dependent())
            _rho_m_dot.d(_L) = - (_k1 * _p_dot()) / neml2::pow(_L(), 2.0);

        if (_T.is_dependent())
            _rho_m_dot.d(_T) = - (_k2_0 * _Q_d) / (_k_B * neml2::pow(_T(), 2.0)) * neml2::exp(- _Q_d / (_k_B * _T())) * _rho_m() * _p_dot();
        
        if (_rho_m.is_dependent())
            _rho_m_dot.d(_rho_m) = - k2 * _p_dot();

        if (const auto * const k1 = nl_param("k1"))
            _rho_m_dot.d(*k1) = _p_dot() / _L();
        
        if (const auto * const k2_0 = nl_param("k2_0"))
            _rho_m_dot.d(*k2_0) = - neml2::exp(- _Q_d / (_k_B * _T())) * _rho_m() * _p_dot();
        
        if (const auto * const Q_d = nl_param("Q_d"))
            _rho_m_dot.d(*Q_d) = _k2_0 / (_k_B * _T()) * neml2::exp(- _Q_d / (_k_B * _T())) * _rho_m() * _p_dot();
    }
}
}