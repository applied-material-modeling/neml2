# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

# Local sphinx extensions live under doc/_ext/.
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project = "NEML2"
author = "Argonne National Laboratory"
copyright = "2024, UChicago Argonne, LLC"

try:
    release = _pkg_version("neml2")
except PackageNotFoundError:
    # neml2 must be installed (non-editable preferred) for autodoc to work,
    # but we still want `sphinx-build` to surface a useful message rather
    # than crash on import.
    release = "unknown"
version = release

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "myst_nb",  # markdown + executable notebooks (supersedes myst_parser for .md/.ipynb)
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinxcontrib.programoutput",
    # Local: autogenerates doc/generated/syntax/** from `neml2-syntax --json`
    # before sphinx reads any source files.
    "neml2_syntax",
    # Local: injects an "Open in Colab" badge on the executable tutorial
    # notebooks (doc/_ext/colab_button.py).
    "colab_button",
]

# sphinx-copybutton: strip common shell / REPL prompts from the copied
# text so a user pasting a shell snippet doesn't paste the `$ ` along
# with it. The regex covers `$`, `>>> `, `... `, and `In [N]: `.
copybutton_prompt_text = r"^\$ |^>>> |^\.\.\. |^In \[\d+\]: "
copybutton_prompt_is_regexp = True
# Skip the line-numbers gutter and any output blocks that myst-nb
# renders next to code cells.
copybutton_exclude = ".linenos, .gp, .go"

# MyST-NB owns the markdown / notebook suffixes; rst is also enabled so
# autosummary-generated stubs (which use rst directive syntax) parse with
# the built-in reStructuredText parser instead of being misread as Markdown.
# `.rst` is FIRST because autosummary defaults to the first registered
# suffix for stub files; the .md ordering is otherwise irrelevant since
# every existing source file already names its own extension.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
# Make the new stub-suffix default explicit so future refactors don't
# silently flip it back to .md.
autosummary_generate_overwrite = True

master_doc = "index"

# Suppress noisy warning categories that the strict CI build (`-W`) would
# otherwise reject:
#   misc.highlighting_failure -- HIT files lex with the `ini` lexer, which
#       trips on `!include` / `${...}`; Pygments falls back to relaxed mode
#       and renders correctly.
#   ref.python -- same root cause: ambiguous cross-references when both the
#       re-export and the submodule definition document the same name.
#   docutils -- a handful of legacy docstrings have rst-formatting issues
#       (indentation, undefined directives) that are docstring-fix follow-ups,
#       not blocking issues for the API ref scaffold.
suppress_warnings = [
    "misc.highlighting_failure",
    # Some legacy class docstrings have rst-formatting quirks
    # (indentation, undefined directives) that surface as docutils
    # warnings under autodoc. These are docstring-fix follow-ups, not
    # blocking issues for the API ref scaffold.
    "docutils",
    "ref.python",
]
# Tutorials are notebook-only: Sphinx renders each `main.ipynb` directly.
# The reference-only tutorial `.md` pages (no `.ipynb` sibling) render as
# ordinary markdown. Nothing to exclude beyond the usual build cruft.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# ---------------------------------------------------------------------------
# Numbering for figures / tables / equations
# ---------------------------------------------------------------------------
# `numfig` enables sphinx-side figure / table counters. Sphinx's equation
# counters (`math_numfig` + `math_number_all`) only reach equations created
# through the `{math}` directive — they do NOT touch MyST's `$$ ... $$`
# dollarmath blocks. To get a per-page counter on every displayed equation
# (including the bare dollarmath we use throughout the tutorials), let
# MathJax 3 do the numbering: `tags: "all"` tells MathJax to number every
# displayed equation, and the counter naturally resets per page since each
# HTML page loads MathJax independently.
numfig = True
math_numfig = True
math_number_all = True
mathjax3_config = {
    "tex": {"tags": "all", "tagSide": "right"},
}

# ---------------------------------------------------------------------------
# MyST + MyST-NB
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Execute the cheap tutorial notebooks at build time and cache the results
# (cache lives under `_build/.jupyter_cache/`), so editing prose around a
# stable code cell does not re-execute it. The executable notebooks live
# under `content/tutorials/` and `content/modules/`, so a global `cache`
# only ever runs those; the syntax-catalog and reference pages have no
# executable cells and cost nothing.
#
# `nb_execution_in_temp` runs each notebook in a throwaway directory: the
# Colab-runnable tutorials create their own input files via `%%writefile`,
# so executing in a temp dir keeps those writes (and any `neml2-compile`
# artifacts) out of the source tree while relative `load_model("input.i")`
# still resolves against the same cwd.
#
# The expensive notebooks below are excluded from execution and instead
# render from their committed, pre-baked outputs (kept executed by the
# scoped `check-notebook-executed` pre-commit hook); executing them needs a
# GPU and minutes-to-hours per run. These are the two pyzag calibration
# tutorials plus the physics-module KWN worked examples (316H precipitation,
# Al-Cu TTP) re-homed from v2's `python/examples/`. Patterns are matched
# against the path tail (`PurePosixPath.match`).
nb_execution_mode = "cache"
nb_execution_timeout = 60
nb_execution_in_temp = True
nb_execution_excludepatterns = [
    "optimization/deterministic/main.ipynb",
    "optimization/statistical/main.ipynb",
    "modules/kwn/precipitation_316h.ipynb",
    "modules/kwn/al_cu_ttp.ipynb",
    "modules/solid_mechanics/crystal_plasticity/formulations.ipynb",
    "modules/solid_mechanics/crystal_plasticity/polefigures.ipynb",
]

# ---------------------------------------------------------------------------
# Autodoc / autosummary
# ---------------------------------------------------------------------------
autosummary_generate = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "shibuya"
html_title = f"NEML2 {release}"

# Sphinx serves anything in html_static_path under /_static/. We mount the
# existing convention dirs (doc/asset, doc/js) verbatim so paths in source
# markdown can read `../../_static/asset/...` and so the per-page raw-HTML
# mounts in references/cpp_dispatch.md (the `.scheduler-demo` divs) keep working.
html_static_path = ["asset", "js"]

# CSS overrides loaded on every page; same is-it-needed logic as html_js_files
# (inert when no matching markup exists, so global-load cost is zero).
html_css_files = [
    "css/custom.css",
]

# Custom JS, loaded on every page. Each is inert when its trigger is absent, so
# the cost of loading globally is zero on pages that don't use it. All are
# *classic* scripts wrapped in an IIFE -- NOT ES modules, because the doc-render
# validator (check_doc_render.py) opens pages over file://, where browsers block
# module loading under the CORS null-origin rule.
#   - scheduler-demos.js: animates `.scheduler-demo` divs (Web Animations API).
#   - version-banner.js: on the multi-version gh-pages site, prepends an
#     "old/dev documentation" banner by comparing the URL's version segment
#     against /<repo>/versions.json; no-op on local/file:// builds.
html_js_files = [
    "components/scheduler-demos.js",
    "components/version-banner.js",
]

html_logo = "asset/logo_light.png"
# No favicon: the project logo is a wide wordmark and does not crop
# cleanly into a 32x32 square.

# shibuya theme options. See https://shibuya.lepture.com/customisation/
# The light/dark logo paths are URLs under the html_baseurl, not source-tree
# paths. Sphinx flattens html_static_path = ["asset", "js"] so files from
# doc/asset/ land directly under /_static/ -- no extra "asset/" segment.
html_theme_options = {
    "accent_color": "iris",
    "github_url": "https://github.com/applied-material-modeling/neml2",
    "color_mode": "auto",
    "light_logo": "_static/logo_light.png",
    "dark_logo": "_static/logo_dark.png",
}


def _write_nojekyll(app, exception):
    # GitHub Pages serves any branch through Jekyll by default, which strips
    # files and directories whose names start with `_` -- exactly the prefix
    # Sphinx uses for `_static/`, `_sources/`, `_images/`. A `.nojekyll`
    # marker at the site root disables Jekyll for the whole branch. Sphinx
    # has no native option for this; emit it from the build-finished hook
    # so every artifact carries it and the eventual `gh-pages` root gets
    # one when the main-deploy uploads `doc/_build/html/` verbatim.
    #
    # Intentionally NOT gated on `exception is None`: under `-W --keep-going`
    # Sphinx still emits the full HTML tree even when it finally raises a
    # warning-as-error, and we want `.nojekyll` to land alongside that tree.
    if app.builder.name == "html" and getattr(app, "outdir", None):
        (Path(app.outdir) / ".nojekyll").touch()


def setup(app):
    app.connect("build-finished", _write_nojekyll)
