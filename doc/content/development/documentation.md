(building-docs)=
# Documentation

The doc pipeline is `sphinx-build` driven by `doc/conf.py`, wrapped by
`doc/scripts/build.sh`:

```shell
pip install -e ".[dev]" -v               # sphinx + extensions land here
doc/scripts/build.sh                     # build to doc/_build/html
xdg-open doc/_build/html/index.html      # macOS: `open ...`
```

The wrapper runs `sphinx-build -j auto -W --keep-going` with the
jupyter-cache pre-create workaround that lets parallel myst-nb workers
build without racing. `--help` lists every flag; the useful ones:
`--clean` for a cold rebuild, `--serve` to start `python -m http.server`
on `127.0.0.1:8765` (prints the SSH-tunnel command), `--port`,
`--dest`, `--no-strict`.

For live preview during editing:

```shell
sphinx-autobuild doc doc/_build/html
```

This serves at `http://127.0.0.1:8000/` and rebuilds whenever anything
under `doc/` changes.

The syntax catalog under `/generated/syntax/` is regenerated from
`neml2-syntax --json` on every build via a dedicated Sphinx extension.
