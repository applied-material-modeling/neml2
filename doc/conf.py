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
# Jupytext keeps `.md` mirrors next to every executable `.ipynb` tutorial
# so they round-trip cleanly through the jupytext pre-commit hook. Sphinx
# must not pick those up — they would double-register against the
# notebooks and the source-of-truth for the rendered output is the
# `.ipynb` (the only one with cell metadata + persisted execution
# counts). Add a new exclude entry whenever a new jupytext-paired
# notebook tutorial lands.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "content/tutorials/optimization/deterministic/main.md",
    "content/tutorials/optimization/statistical/main.md",
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

# Default: don't re-execute notebooks at build time. The executable
# `.ipynb` tutorials are committed with pre-baked outputs (enforced by
# the `check-notebook-executed` pre-commit hook), and the syntax-catalog
# pages have no executable cells, so a global "off" makes the usual
# rebuild instant.
#
# Tutorial pages that embed live `{code-cell}` directives opt in to
# execution via per-page front-matter:
#
#   ---
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   mystnb:
#     execution_mode: cache
#   ---
#
# `cache` reuses jupyter-cache results when the cell source is unchanged
# (cache lives under `_build/.jupyter_cache/`), so editing prose around a
# stable code cell does not re-execute it.
nb_execution_mode = "off"
nb_execution_timeout = 60

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
# markdown can read `../../_static/asset/...` or so the per-page raw-HTML
# in scheduler.md keeps working unchanged.
html_static_path = ["asset", "js"]

# CSS overrides loaded on every page; same is-it-needed logic as html_js_files
# (inert when no matching markup exists, so global-load cost is zero).
html_css_files = [
    "css/custom.css",
]

# Custom JS components, loaded on every page. They are inert until a div
# with the matching class exists in the rendered HTML, so the cost of
# loading them globally is zero on pages that don't use them.
html_js_files = [
    "components/work-dispatcher.js",
    "components/static-hybrid-scheduler.js",
    "components/simple-scheduler-demo.js",
    "components/static-hybrid-scheduler-demo.js",
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
