Set up the NEML2 Python development environment.

Parse `$ARGUMENTS` for an optional Python executable path (default: `python3` / `pip3` on PATH).

**Steps:**

1. Install the Python package with development extras (builds the pybind11 C++ extension):
   ```bash
   pip install ".[dev]" -v
   ```
   If a custom Python path was given (e.g., `/path/to/venv/bin/python`), use:
   ```bash
   /path/to/venv/bin/pip install ".[dev]" -v
   ```

2. Verify the installation:
   ```bash
   python3 -c "import neml2; print('neml2', neml2.__version__, 'installed OK')"
   ```

3. Confirm pytest can discover the test suite:
   ```bash
   pytest --collect-only python/tests -q 2>&1 | tail -5
   ```

**Notes:**
- The `[dev]` extra pulls in `pytest`, `black`, and other developer tools.
- A working libtorch installation must be discoverable via `CMAKE_PREFIX_PATH` or the PyTorch Python package; the build will fail otherwise.
- If the extension fails to compile, suggest running the C++ build first with `/build cc` to get compile commands for debugging.
