Verify that the NEML2 documentation builds cleanly and check for errors.

Run from the repository root.

---

## Steps

1. **Generate examples and build HTML:**
   ```bash
   ./doc/scripts/examples.py --log-level INFO --cmake-configure-args="-GNinja"
   ./doc/scripts/genhtml.py --log-level INFO
   ```

2. **Check logs:**
   ```bash
   cat build/doc/doxygen.html.log
   cat build/doc/doxygen.python.log
   cat build/doc/syntax.err
   ```

3. **Report:**
   - If `syntax.err` is non-empty, show its full contents.
   - List any errors or warnings from the doxygen logs.
   - If all logs are clean, report: "Documentation builds cleanly."

---

Do not auto-fix errors found in the logs — present the findings and the affected file/docstring for review.
