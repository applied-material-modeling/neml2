#pragma once
#include "neml2/models/Model.h"

namespace neml2
{
class Scalar;

class MeanFreePath : public Model
{
public:
    static OptionSet expected_options();

    MeanFreePath(const OptionSet & options);

protected:
    void set_value(bool out, bool dout_din, bool d2out_din2) override;

    // -----------------------------------------------------------------------
    // Feature Flags
    // -----------------------------------------------------------------------

    // Enable sub-grain boundary contribution (L2): lath / block / packet / PAG
    const bool _use_L2;

    // Enable precipitate contribution (l3): MX and M23C6
    const bool _use_L3;

    // -----------------------------------------------------------------------
    // Input Variable: mobile dislocation density rho_m [m^-2]
    // -----------------------------------------------------------------------

    const Variable<Scalar> & _rho_m;

    // -----------------------------------------------------------------------
    // Parameters for L2 (only used when _use_L2 == true)
    // -----------------------------------------------------------------------

    // Geometric factor for lath boundary
    const Scalar * _c_lath = nullptr;
    // Martensitic lath width [m]
    const Scalar * _d_lath = nullptr;

    // Geometric factor for block boundary
    const Scalar * _c_block = nullptr;
    // Mean block width [m]
    const Scalar * _d_block = nullptr;

    // Geometric factor for packet boundary
    const Scalar * _c_packet = nullptr;
    // Mean packet size [m]
    const Scalar * _d_packet = nullptr;

    // Geometric factor for prior-austenite grain (PAG) boundary
    const Scalar * _c_PAG = nullptr;
    // Mean PAG size [m]
    const Scalar * _d_PAG = nullptr;

    // -----------------------------------------------------------------------
    // Parameters for L3 (only used when _use_L3 == true)
    // -----------------------------------------------------------------------

    // Geometric factor for MX precipitate
    const Scalar * _c_MX = nullptr;
    // Mean MX precipitate spacing [m]
    const Scalar * _d_MX = nullptr;

    // Geometric factor for M23C6 precipitate
    const Scalar * _c_M23C6 = nullptr;
    // Mean M23C6 precipitate spacing [m]
    const Scalar * _d_M23C6 = nullptr;

    // -----------------------------------------------------------------------
    // Output variable: Mean free path [m]
    // -----------------------------------------------------------------------

    Variable<Scalar> & _L;

};
}