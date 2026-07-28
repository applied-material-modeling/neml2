#pragma once
#include "neml2/models/Model.h"

namespace neml2
{
class Scalar;

class KocksMeckingDislocationDensity : public Model
{
public:
    static OptionSet expected_options();

    KocksMeckingDislocationDensity(const OptionSet & options);

protected:
    void set_value(bool out, bool dout_din, bool d2out_din2) override;

    // Plastic Flow rate
    const Variable<Scalar> & _gamma_dot;
    // const parameter k1
    const Scalar & _k1;
    const Variable<Scalar> & _L;
    // temperature-dependent parameter k2
    const Scalar & _k2_0; // reference recovery coefficient
    const Scalar & _Q_d;  // activation energy for thermally-activated recovery
    const Scalar & _k_B;  // Boltzmann's constant
    const Variable<Scalar> & _T; // input temperature
    // dislocation density (input variable from backward euler integration)
    const Variable<Scalar> & _rho_m;
    // output: dislocation density rate
    Variable<Scalar> & _rho_m_dot;
};
}
