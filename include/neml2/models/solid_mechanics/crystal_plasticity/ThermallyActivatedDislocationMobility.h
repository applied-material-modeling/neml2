#pragma once
#include "neml2/models/Model.h"

namespace neml2
{
class Scalar;

class ThermallyActivatedDislocationMobility : public Model
{
public:
    static OptionSet expected_options();

    ThermallyActivatedDislocationMobility(const OptionSet & options);

protected:
    void set_value(bool out, bool dout_din, bool d2out_din2) override;

    // Input variables (effective shear and athermal shear)
    const Variable<Scalar> & _tau_eff;
    const Variable<Scalar> & _tau_a;
    const Variable<Scalar> & _L;
    const Variable<Scalar> & _T;

    // Variables from equation
    const Scalar & _h;      // buffer
    const Scalar & _b;      // buffer
    const Scalar & _a;      // buffer
    const Scalar & _Bk;     // parameter
    const Scalar & _tau_p;  // parameter
    const Scalar & _T_0;    // parameter
    const Scalar & _p;      // parameter
    const Scalar & _q;      // parameter
    const Scalar & _D_H;    // parameter
    const Scalar & _k_B;    // buffer
    const Scalar & _s;      // parameter

    // Output variable (dislocation velocity)
    Variable<Scalar> & _v;

};
}