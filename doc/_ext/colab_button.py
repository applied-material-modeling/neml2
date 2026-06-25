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

"""Sphinx extension: inject an "Open in Colab" badge on tutorial notebooks.

Every executable tutorial under ``content/tutorials/`` whose source is a
Jupyter ``.ipynb`` (the cheap, Colab-runnable notebooks) gets a clickable
Colab badge inserted just below its page title. The badge links to

    https://colab.research.google.com/github/<repo>/blob/<ref>/doc/<docname>.ipynb

The git ``<ref>`` comes from the ``NEML2_DOC_GIT_REF`` environment variable
(default ``main``); the release-deploy CI job sets it to the release tag so a
launched notebook always matches the ``pip install neml2`` release it runs
against. The badge is injected into the rendered HTML only -- the committed
``.ipynb`` never contains it, so opening the notebook in Colab shows no
self-referential badge.

The two *expensive* jupytext notebooks (``optimization/deterministic`` and
``optimization/statistical``) are skipped: they are not Colab-ready (heavy
pyzag runs, a shared ``demo_model.i`` symlink that does not survive a
single-file Colab open) and render from pre-baked outputs.
"""

from __future__ import annotations

import os

from docutils import nodes

#: ``owner/repo`` slug used to build the Colab GitHub URL.
DEFAULT_REPO = "applied-material-modeling/neml2"

#: Tutorial docnames (relative to ``doc/``, no suffix) that must NOT get a
#: badge even though they are ``.ipynb`` -- the expensive, non-Colab notebooks.
SKIP_DOCNAMES = frozenset(
    {
        "content/tutorials/optimization/deterministic/main",
        "content/tutorials/optimization/statistical/main",
    }
)

_TUTORIAL_PREFIX = "content/tutorials/"


def _colab_url(repo: str, ref: str, docname: str) -> str:
    return f"https://colab.research.google.com/github/{repo}/blob/{ref}/doc/{docname}.ipynb"


def _badge_html(url: str) -> str:
    return (
        f'<a class="colab-badge" href="{url}" target="_blank" rel="noopener" '
        f'title="Run this tutorial in Google Colab">'
        f'<img alt="Open in Colab" '
        f'src="https://colab.research.google.com/assets/colab-badge.svg"></a>'
    )


def _inject_badge(app, doctree):
    env = app.env
    docname = env.docname
    if not docname.startswith(_TUTORIAL_PREFIX) or docname in SKIP_DOCNAMES:
        return
    # Only notebooks (the Colab-runnable tutorials), not the reference-only
    # markdown pages that live alongside them.
    if not str(env.doc2path(docname)).endswith(".ipynb"):
        return

    repo = getattr(app.config, "colab_repo", DEFAULT_REPO)
    ref = os.environ.get("NEML2_DOC_GIT_REF", "main")
    raw = nodes.raw("", _badge_html(_colab_url(repo, ref, docname)), format="html")

    # Place the badge immediately after the page title (index 0 of the first
    # top-level section), so it sits right under the H1.
    for section in doctree.findall(nodes.section):
        insert_at = 1 if section.children and isinstance(section[0], nodes.title) else 0
        section.insert(insert_at, raw)
        break


def setup(app):
    app.add_config_value("colab_repo", DEFAULT_REPO, "env")
    app.connect("doctree-read", _inject_badge)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
