#pragma once
#include "neml2/models/Model.h"

namespace neml2
{
class Scalar;

class AthermalStress : public Model
{
public:
    static OptionSet expected_options();

    AthermalStress(const OptionSet & options);

protected:
    void set_value(bool out, bool dout_din, bool d2out_din2) override;

    // Input parameters
    const Scalar & _G;
    const Scalar & _alpha;
    const Scalar & _b;

    // Input variable
    const Variable<Scalar> & _L;

    // Output (athermal stress)
    Variable<Scalar> & _sigma_a;
};
}