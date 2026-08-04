#pragma once
#include "neml2/models/Model.h"

namespace neml2
{
class Scalar;

class AthermalSoluteResistance : public Model
{
public:
    static OptionSet expected_options();

    AthermalSoluteResistance(const OptionSet & options);

protected:
    void set_value(bool out, bool dout_din, bool d2out_din2) override;

    // Inputs for Athermal Microstructural Term
    const Scalar & _G;              // Shear modulus
    const Scalar & _alpha;          // dislocation interaction constant
    const Scalar & _b;              // burger's vector
    const Variable<Scalar> & _L;    // mean free path

    // Inputs for Solute Resistance Term
    const bool _include_solid_solution;   // boolean logic gate for sigma_ss

    // Output (athermal solute resistance)
    Variable<Scalar> & _sigma_0;

    // -------- Solute Interaction Time (Arrhenius) t_a --------
    const Scalar * _t_a0 = nullptr;
    const Scalar * _Q_a = nullptr;
    const Scalar * _k_B = nullptr;
    const Variable<Scalar> * _T = nullptr;

    // -------- Waiting Time t_w --------
    const Scalar * _m = nullptr;                   // schmid factor
    const Variable<Scalar> * _v_disl = nullptr;    // dislocation velocity

    // -------- solute resistance --------
    const Scalar * _tau_s0 = nullptr;
    const Scalar * _p_ss = nullptr;

};
}