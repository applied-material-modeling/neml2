#pragma once
#include "neml2/models/solid_mechanics/crystal_plasticity/ThermallyActivatedDislocationMobility.h"

namespace neml2
{
class Scalar;

class ThermallyActivatedDislocationMobility_diag : public ThermallyActivatedDislocationMobility
{
public:
    static OptionSet expected_options();

    ThermallyActivatedDislocationMobility_diag(const OptionSet & options);

protected:
    void set_value(bool out, bool dout_din, bool d2out_din2) override;

    // Additional Debugging Outputs
    Variable<Scalar> & _K;
    Variable<Scalar> & _K_mcl_eff;
    Variable<Scalar> & _tau_ratio;
    Variable<Scalar> & _D_G;
    Variable<Scalar> & _mclD_G;
    Variable<Scalar> & _exp_core;
    Variable<Scalar> & _exp_arg;

};
}
