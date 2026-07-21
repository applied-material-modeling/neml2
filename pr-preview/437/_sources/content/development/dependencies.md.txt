(contributing-deps)=
# Dependency pinning

Versions of third-party dependencies are tracked in
`scripts/dependencies.yaml` and propagated into source files
(`pyproject.toml`, `.github/workflows/*.yaml`, doc snippets, …) via
inline annotations:

```python
# dependencies: torch.version_min
"torch>=2.10.0",
```

The annotation line above the version literal tells the
`scripts/dep_manager.py` tool which `scripts/dependencies.yaml` entry
owns the value. Use the tool rather than editing version strings by
hand:

```shell
python scripts/dep_manager.py check                   # CI-equivalent verify
python scripts/dep_manager.py list                    # show all tracked deps
python scripts/dep_manager.py bump torch.version_min 2.11.0   # update + propagate
```

The torch compatibility matrix in
`doc/content/installation/compatibility.yaml` is governed by its own
tool, `scripts/compat_matrix.py`, which validates the matrix and renders
it into `doc/content/installation/compatibility.md`. Both tools are
wired into pre-commit so a typo will be caught at commit time.
