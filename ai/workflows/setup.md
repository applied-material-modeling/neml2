# Workflow: SETUP

Purpose: set up the Python development environment.

## Procedure

1. Use the requested Python executable or default to `python3`; refer to the chosen interpreter as `<py>`.
2. Install:
   - `pip install ".[dev]" -v`
3. Verify:
   - `<py> -c "import neml2; print('neml2', neml2.__version__, 'installed OK')"`
4. Confirm pytest discovery:
   - `<py> -m pytest --collect-only python/tests -q`
5. If extension compilation fails, recommend doing a C++ build first with the `cc` preset.
