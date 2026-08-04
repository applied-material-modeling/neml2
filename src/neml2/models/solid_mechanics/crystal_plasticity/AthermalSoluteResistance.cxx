#include "neml2/models/solid_mechanics/crystal_plasticity/AthermalSoluteResistance.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/exp.h"

namespace neml2
{
register_NEML2_object(AthermalSoluteResistance);

OptionSet
AthermalSoluteResistance::expected_options()
{
    OptionSet options = Model::expected_options();
    options.set_parameter<TensorName<Scalar>>("shear_modulus");
    options.set_parameter<TensorName<Scalar>>("alpha");
    options.set_parameter<TensorName<Scalar>>("b");
    options.set_input("L");
    options.set<bool>("include_solid_solution") = true;
    options.set_output("athermal_solute_resistance");
    options.set<TensorName<Scalar>>("t_a0");
    options.set<TensorName<Scalar>>("Q_a");
    options.set<TensorName<Scalar>>("k_B");
    options.set_input("temperature");
    options.set<TensorName<Scalar>>("m");
    options.set_input("v_disl");
    options.set<TensorName<Scalar>>("tau_s0");
    options.set<TensorName<Scalar>>("p_ss");

    return options;
}
AthermalSoluteResistance::AthermalSoluteResistance(const OptionSet & options) : Model(options),
    _G(declare_parameter<Scalar>("shear_modulus", "shear_modulus", true)),
    _alpha(declare_parameter<Scalar>("alpha", "alpha")),
    _b(declare_buffer<Scalar>("b", "b")),
    _L(declare_input_variable<Scalar>("L")),
    _include_solid_solution(options.get<bool>("include_solid_solution")),
    _sigma_0(declare_output_variable<Scalar>("athermal_solute_resistance"))
{
    if (_include_solid_solution)
    {
        _t_a0   = &declare_parameter<Scalar>("t_a0", "t_a0", true);
        _Q_a    = &declare_parameter<Scalar>("Q_a", "Q_a", true);
        _k_B    = &declare_parameter<Scalar>("k_B", "k_B");
        _T      = &declare_input_variable<Scalar>("temperature");
        _m      = &declare_parameter<Scalar>("m", "m");
        _v_disl = &declare_input_variable<Scalar>("v_disl");
        _tau_s0 = &declare_parameter<Scalar>("tau_s0", "tau_s0", true);
        _p_ss   = &declare_parameter<Scalar>("p_ss", "p_ss", true);
    }
}

void
AthermalSoluteResistance::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{   
    const auto sigma_a  = (_alpha * _G * _b) / _L();
    auto sigma_0        = sigma_a;

    if (_include_solid_solution)
    {
        const auto t_w        = _L() / (*_v_disl)();
        const auto t_a        = (*_t_a0) * neml2::exp((*_Q_a) / ((*_k_B) * (*_T)()));
        const auto tau_ss     = (*_tau_s0) * neml2::exp(-neml2::pow((t_w / t_a), (*_p_ss)));
        const auto sigma_ss   = tau_ss / (*_m);
        sigma_0               = sigma_a + sigma_ss;

    }

    if (out)
        _sigma_0 = sigma_0;
    
    if (dout_din)
    {   
        if (_L.is_dependent())
            _sigma_0.d(_L) = -(_alpha * _G * _b) / neml2::pow(_L(), 2.0);
        
        if (const auto * const G = nl_param("shear_modulus"))
            _sigma_0.d(*G) = (_alpha * _b) / _L();
    
        if (const auto * const alpha = nl_param("alpha"))
            _sigma_0.d(*alpha) = (_G * _b) / _L();
        
        if (_include_solid_solution)
        {
            const auto t_w        = _L() / (*_v_disl)();
            const auto t_a        = (*_t_a0) * neml2::exp((*_Q_a) / ((*_k_B) * (*_T)()));
            const auto tau_ss     = (*_tau_s0) * neml2::exp(-neml2::pow((t_w / t_a), (*_p_ss)));
            const auto sigma_ss   = tau_ss / (*_m);
            
            if (_L.is_dependent())
                _sigma_0.d(_L) = -(_alpha * _G * _b) / neml2::pow(_L(), 2.0) - 
                                 (*_tau_s0) * (*_p_ss) / ((*_m) * (t_a) * (*_v_disl)()) * neml2::pow((t_w/t_a), ((*_p_ss) - 1.0)) *
                                 neml2::exp(-neml2::pow((t_w/t_a), (*_p_ss)));

            if ((*_v_disl).is_dependent())
                _sigma_0.d(*_v_disl) = (*_tau_s0) * (*_p_ss) * _L() / ((*_m) * (t_a) * neml2::pow((*_v_disl)(), 2.0)) *
                                       neml2::pow((t_w/t_a), ((*_p_ss) - 1.0)) * neml2::exp(-neml2::pow((t_w/t_a), (*_p_ss)));
            
            if ((*_T).is_dependent())
                _sigma_0.d(*_T) = -(*_tau_s0) * (*_p_ss) * t_w * (*_t_a0) * (*_Q_a) / ((*_m) * (*_k_B) * neml2::pow((*_T)(), 2.0) * neml2::pow(t_a, 2.0)) *
                                  neml2::pow((t_w/t_a), ((*_p_ss) - 1.0)) * neml2::exp((*_Q_a)/((*_k_B) * (*_T)())) * 
                                  neml2::exp(-neml2::pow((t_w/t_a), (*_p_ss)));
        }
    }
}
}