#include "neml2/models/solid_mechanics/crystal_plasticity/ThermallyActivatedDislocationMobility.h"
#include "neml2/tensors/Scalar.h"
#include "neml2/tensors/functions/pow.h"
#include "neml2/tensors/functions/exp.h"
#include "neml2/tensors/functions/macaulay.h"
#include "neml2/tensors/functions/heaviside.h"
#include "neml2/tensors/functions/log.h"
#include "neml2/tensors/functions/clamp.h"

namespace neml2
{
register_NEML2_object(ThermallyActivatedDislocationMobility);

OptionSet
ThermallyActivatedDislocationMobility::expected_options()
{
    OptionSet options = Model::expected_options();
    options.set_input("effective_shear");
    options.set_input("athermal_shear");
    options.set_input("L");
    options.set_input("temperature");
    options.set_buffer<TensorName<Scalar>>("h");
    options.set_buffer<TensorName<Scalar>>("b");
    options.set_buffer<TensorName<Scalar>>("a");
    options.set_parameter<TensorName<Scalar>>("Bk");
    options.set_parameter<TensorName<Scalar>>("tau_p");
    options.set_parameter<TensorName<Scalar>>("T_0");
    options.set_parameter<TensorName<Scalar>>("p");
    options.set_parameter<TensorName<Scalar>>("q");
    options.set_parameter<TensorName<Scalar>>("H_0");
    options.set_buffer<TensorName<Scalar>>("k_B");
    options.set_parameter<TensorName<Scalar>>("s");
    options.set_output("v_disl");

    return options;
}
ThermallyActivatedDislocationMobility::ThermallyActivatedDislocationMobility(const OptionSet & options) : Model(options),
    _tau_eff(declare_input_variable<Scalar>("effective_shear")),
    _tau_a(declare_input_variable<Scalar>("athermal_shear")),
    _L(declare_input_variable<Scalar>("L")),
    _T(declare_input_variable<Scalar>("temperature")),
    _h(declare_buffer<Scalar>("h", "h")),
    _b(declare_buffer<Scalar>("b", "b")),
    _a(declare_buffer<Scalar>("a", "a")),
    _Bk(declare_parameter<Scalar>("Bk", "Bk", true)),
    _tau_p(declare_parameter<Scalar>("tau_p", "tau_p", true)),
    _T_0(declare_parameter<Scalar>("T_0", "T_0", true)),
    _p(declare_parameter<Scalar>("p", "p", true)),
    _q(declare_parameter<Scalar>("q", "q", true)),
    _D_H(declare_parameter<Scalar>("H_0", "H_0", true)),
    _k_B(declare_buffer<Scalar>("k_B", "k_B")),
    _s(declare_parameter<Scalar>("s","s", true)),
    _v(declare_output_variable<Scalar>("v_disl"))

{
}
void
ThermallyActivatedDislocationMobility::set_value(bool out, bool dout_din, bool /*d2out_din2*/)
{
    // Precompute common subexpressions
    const auto K            = (_h * _L() * _b) / (pow(_a, 2.0) * _Bk);     // Kink-pair prefactor: K = h·L·b / (a²·Bk)
    const auto mcl_eff      = macaulay(_tau_eff());                         // Positive effective shear stress (for pre-exponential driving force)
    const auto tau_1        = macaulay(_tau_eff() - _tau_a());              // Excess stress above athermal threshold
    const auto tau_tilda    = tau_1 / _tau_p;
    const auto tau_ratio    = clamp(tau_tilda, 1.0e-30, 1.0 - 1.0e-6);
    const auto D_G          = pow(1.0 - pow(tau_ratio, _p), _q) - _T() / _T_0;
    const auto dg1          = macaulay(D_G);
    const auto exp_core     = -_D_H * dg1 / (2.0 * _k_B * _T());
    const auto exp_val      = exp(exp_core);
    const auto v_kp         = K * mcl_eff * exp_val;

    if (out)
    {
        _v = v_kp;
    }


    if (dout_din)
    {
        // -------- SHARED INTERMEDIATES ---------

        const auto inner    = 1.0 - pow(tau_ratio, _p);

        // -------- CHAIN RULE COMPUTATION for dv_dtau_eff --------

        const auto dtau1_dtau_eff       = heaviside(tau_1);
        const auto dtau_tilda_dtau_eff  = 1.0 / _tau_p * dtau1_dtau_eff;
        const auto dD_G_dtau_eff        = _q * pow(1.0 - pow(tau_ratio, _p), _q - 1.0) * _p * pow(tau_ratio, _p - 1.0) * dtau_tilda_dtau_eff;
        const auto dv_kp_dtau_eff       = K * exp_val * heaviside(_tau_eff()) + K * mcl_eff * _D_H / (2.0 * _k_B * _T()) * exp_val * dD_G_dtau_eff;

        if (_tau_eff.is_dependent())
            _v.d(_tau_eff) = dv_kp_dtau_eff;

        // -------- CHAIN RULE COMPUTATION for dv_dtau_a --------

        const auto dtau1_dtau_a         = -1.0 * heaviside(tau_1);
        const auto dtau_tilda_dtau_a    = 1 / _tau_p * dtau1_dtau_a;
        const auto dD_G_dtau_a          = _q * pow(1.0 - pow(tau_ratio, _p), _q - 1.0) * _p * pow(tau_ratio, _p - 1.0) * dtau_tilda_dtau_a;
        const auto dv_kp_dtau_a         = K * mcl_eff * _D_H / (2.0 * _k_B * _T()) * dD_G_dtau_a * exp_val;
        
        if (_tau_a.is_dependent())
            _v.d(_tau_a) = dv_kp_dtau_a;

        // -------- CHAIN RULE COMPUTATION for dv_dL --------

        const auto dv_kp_dL         = _h * _b / (pow(_a, 2.0) * _Bk) * mcl_eff * exp_val;

        if (_L.is_dependent())
            _v.d(_L) = dv_kp_dL;

        // -------- CHAIN RULE COMPUTATION for dv_dT --------

        const auto dexp_core_dT         = _D_H / (2.0 * _k_B * pow(_T(), 2.0)) * pow(1.0 - pow(tau_ratio, _p), _q);
        const auto dv_kp_dT             = K * mcl_eff * dexp_core_dT * exp_val;

        if (_T.is_dependent())
            _v.d(_T) = dv_kp_dT;

        // -------- CHAIN RULE COMPUTATION for dv_dBk --------

        const auto dv_kp_dBk            = -(_h * _L() * _b) / (pow(_a, 2.0) * pow(_Bk, 2.0)) * mcl_eff * exp_val;

        if (const auto * const Bk = nl_param("Bk"))
            _v.d(*Bk) = dv_kp_dBk;

        // -------- CHAIN RULE COMPUTATION for dv_dtau_p --------

        const auto dtau_ratio_dtau_p    = -tau_1 / pow(_tau_p, 2.0);
        const auto dD_G_dtau_p          = _q * pow(1.0 - pow(tau_ratio, _p), _q - 1.0) * _p * pow(tau_ratio, _p - 1.0) * dtau_ratio_dtau_p;
        const auto dv_kp_dtau_p         = K * mcl_eff * _D_H / (2.0 * _k_B * _T()) * dD_G_dtau_p * exp_val;

        if (const auto * const tau_p = nl_param("tau_p"))
            _v.d(*tau_p) = dv_kp_dtau_p;

        // -------- CHAIN RULE COMPUTATION for dv_dT_0 --------

        const auto dexp_core_dT_0       = -_D_H / (2.0 * _k_B * pow(_T_0, 2.0));
        const auto dv_kp_dT_0           = K * mcl_eff * dexp_core_dT_0 * exp_val;

        if (const auto * const T_0 = nl_param("T_0"))
            _v.d(*T_0) = dv_kp_dT_0;

        // -------- CHAIN RULE COMPUTATION for dv_dp --------

        const auto dtau_ratiop_dp       = -pow(tau_ratio, _p) * log(tau_ratio);
        const auto dD_G_dp              = _q * pow(1.0 - pow(tau_ratio, _p), _q - 1.0) * dtau_ratiop_dp;
        const auto dv_kp_dp             = -K * mcl_eff * _D_H / (2.0 * _k_B * _T()) * dD_G_dp * exp_val;

        if (const auto * const p = nl_param("p"))
            _v.d(*p) = dv_kp_dp;

        // -------- CHAIN RULE COMPUTATION for dv_dq --------

        const auto dD_G_dq              = pow(1.0 - pow(tau_ratio, _p), _q) * log(1.0 - pow(tau_ratio, _p));
        const auto dv_kp_dq             = -K * mcl_eff * _D_H / (2.0 * _k_B * _T()) * dD_G_dq * exp_val;

        if (const auto * const q = nl_param("q"))
            _v.d(*q) = dv_kp_dq;

        // -------- CHAIN RULE COMPUTATION for dv_dD_H --------

        const auto dexp_core_dD_H       = -1.0 * dg1 / (2.0 * _k_B * _T());
        const auto dv_kp_dD_H           = K * mcl_eff * dexp_core_dD_H * exp_val;

        if (const auto * const D_H = nl_param("H_0"))
            _v.d(*D_H) = dv_kp_dD_H;

        // -------- CHAIN RULE COMPUTATION for dv_ds --------

        // const auto dphi_ds              = pow(phi, 2.0) * tau_tilda * exp(-_s * tau_tilda);
        // const auto dtau_ratio_ds        = macaulay(tau_1) / _tau_p * dphi_ds - eps * dphi_ds;
        // const auto dD_G_ds              = -_D_H * _q * pow(inner_safe, _q - 1.0) * _p * pow(tau_ratio, _p - 1.0) * dtau_ratio_ds;
        // const auto dv_kp_ds             = -K * mcl_eff / (_k_B * _T()) * dD_G_ds * exp_val;
        // const auto dQ_ds                = (v_drag * dv_kp_ds * (v_kp + v_drag) - v_kp * v_drag * dv_kp_ds) / pow(v_kp + v_drag + 1.0e-30, 2.0);

        
        // if (const auto * const s = nl_param("s"))
        //     _v.d(*s) = Q * dphi_ds + dQ_ds * phi;
    }
}
}
