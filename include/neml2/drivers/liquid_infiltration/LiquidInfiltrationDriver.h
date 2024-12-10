#pragma once

#include "neml2/drivers/TransientDriver.h"

namespace neml2
{
/**
 * @brief The transient driver specialized for solid mechanics problems.
 *
 */
class LiquidInfiltrationDriver : public TransientDriver
{
public:
  static OptionSet expected_options();

  /**
   * @brief Construct a new LiquidInfiltrationDriver object
   *
   * @param options The options extracted from the input file
   */
  LiquidInfiltrationDriver(const OptionSet & options);

  void setup() override;

  void diagnose(std::vector<Diagnosis> &) const override;

protected:
  virtual void update_forces() override;

  /**
   * The value of the driving force, depending on `_control` this is either the prescribed strain or
   * the prescribed stress.
   */
  Scalar _driving_force;
  VariableName _driving_force_name;
};
}
