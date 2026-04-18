# Workflow: DOCS-VERIFY

Purpose: Verify documentation builds cleanly.

## Procedure

1. Run:
   - `./doc/scripts/examples.py --log-level INFO --cmake-configure-args="-GNinja"`
   - `./doc/scripts/genhtml.py --log-level INFO`
2. Inspect:
   - `build/doc/doxygen.html.log`
   - `build/doc/doxygen.python.log`
   - `build/doc/syntax.err`
3. Report:
   - full `syntax.err` contents if non-empty
   - any warnings or errors from the doxygen logs
   - otherwise state that documentation builds cleanly
4. Do not auto-fix doc build failures in this procedure.
