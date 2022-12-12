#pragma once

#include <string>

#include <torch/torch.h>

#include "models/Model.h"

/// Simple object to read in a verification test
class VerificationTest
{
public:
  VerificationTest(std::string fname);

  /// Evaluate the comparison between the two models
  bool compare(const neml2::Model & model) const;

  /// Driving time data
  torch::Tensor time() const { return _time; };

  /// Driving strain data
  torch::Tensor strain() const { return _strain; };

  /// Driving temperature data
  torch::Tensor temperature() const { return _temperature; };

  /// Stress data, for comparison
  torch::Tensor stress() const { return _stress; };

  /// Does this test have actual temperature data?
  bool with_temperature() const { return _with_temperature; };

  /// Relative tolerance
  double rtol() const { return _rtol; };

  /// Absolute tolerance
  double atol() const { return _atol; };

private:
  /// Read the test data
  void parse();

private:
  const std::string _filename;
  std::string _neml_model_file;
  std::string _neml_model_name;
  std::string _neml2_model_file;
  std::string _neml2_model_name;
  double _rtol, _atol;
  std::string _description;
  bool _with_temperature;
  torch::Tensor _time;
  torch::Tensor _strain;
  torch::Tensor _temperature;
  torch::Tensor _stress;
};

/// Split a string separated by some delimiter character
std::vector<std::string> split_string(const std::string & input, const char * delimiter = " ");
