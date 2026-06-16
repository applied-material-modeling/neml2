#!/usr/bin/env python3
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

"""Headless-browser integrity check for every page in the built sphinx site.

What sphinx + ``-W`` can't see: anything that goes wrong *after* the page
is delivered to the browser. This script loads every built HTML page in
parallel headless Chromium tabs and collects four categories of failure:

* **MathJax render errors.** ``<mjx-merror>`` nodes left in the DOM after
  MathJax's first typeset pass -- TeX MathJax couldn't parse. The classic
  trigger is a MyST display-math context (``$$`` or ```` ```{math} ````)
  wrapping an amsmath display environment like ``\\begin{align}``; MyST
  auto-wraps in ``\\begin{split}`` and ``split`` containing ``align`` is
  invalid LaTeX.
* **Uncaught JavaScript exceptions** (``page.on("pageerror")``). Any
  unhandled JS error during page load -- typically broken theme JS, a
  missing custom-component script, or a 3rd-party library throwing.
* **Console errors** (``console.error(...)``). The page's own scripts
  logged a hard error. Console warnings/info are intentionally skipped
  (too noisy across third-party JS) -- only true errors fail the build.
* **Failed local-asset requests** (``page.on("requestfailed")`` filtered
  to ``file://``). A ``<link>``/``<script>``/``<img>`` pointing at a
  missing file under the built site -- always a sphinx pipeline bug.

Errors land in stderr per page, and a markdown summary is written to
``$GITHUB_STEP_SUMMARY`` when running in CI so the list appears in the
job's Summary tab.

CI-only by design: Playwright + Chromium are ~150 MB and not in
``[dev]`` extras. The matching GitHub Actions step installs them on the
runner. Local debugging: ``pip install playwright && playwright install
chromium``, then ``python doc/scripts/check_math.py doc/_build/html``.

Usage:
    python doc/scripts/check_math.py BUILD_DIR [--concurrency N]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

# MathJax DOM-scan harness. We don't hook into the (version-fragile)
# ``formatError`` callback -- we just wait for the startup promise to
# resolve, then count error nodes. Catches every TeX MathJax couldn't
# typeset regardless of its internal config layout.
_MATHJAX_HARNESS = r"""
() => new Promise((resolve) => {
  const errors = [];
  const finishUp = () => {
    document.querySelectorAll("mjx-merror, .mjx-error").forEach((node) => {
      const title = (node.getAttribute("title") || "").trim();
      const visible = (node.textContent || "").trim().replace(/\s+/g, " ");
      errors.push({
        message: title || "MathJax rendered an error node",
        expr: visible,
      });
    });
    resolve(errors);
  };
  const settle = () => setTimeout(finishUp, 250);
  if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(settle).catch((e) => {
      errors.push({
        message: `MathJax.startup.promise rejected: ${e}`,
        expr: "",
      });
      settle();
    });
  } else {
    finishUp();
  }
});
"""


@dataclass
class PageReport:
    rel: str
    mathjax: list[dict] = field(default_factory=list)
    pageerrors: list[str] = field(default_factory=list)
    console_errors: list[str] = field(default_factory=list)
    failed_assets: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return (
            len(self.mathjax)
            + len(self.pageerrors)
            + len(self.console_errors)
            + len(self.failed_assets)
        )


def _iter_html_pages(build_dir: Path):
    """Yield (relative path string, absolute Path) for every .html under *build_dir*."""
    skip = {"_static", "_sources", "_images", "search.html", "genindex.html"}
    for p in sorted(build_dir.rglob("*.html")):
        rel = p.relative_to(build_dir)
        if any(seg in skip for seg in rel.parts):
            continue
        yield str(rel).replace(os.sep, "/"), p


async def _check_one(context, abs_path: Path, rel: str) -> PageReport:
    """Load one page, attach event hooks, collect all failure categories."""
    report = PageReport(rel=rel)
    page = await context.new_page()

    def on_console(msg):
        # Only severe console messages fail the build. MathJax + shibuya
        # together emit a fair amount of warning/info noise on every page.
        if msg.type == "error":
            report.console_errors.append(msg.text)

    def on_page_error(exc):
        report.pageerrors.append(str(exc))

    def on_request_failed(req):
        # Only flag misses on assets we ourselves shipped. Anything served
        # from the local site is under ``file://`` (since we load via
        # ``file://`` URLs); external HTTP failures are network noise and
        # don't reflect a docs-pipeline bug.
        if req.url.startswith("file://"):
            failure = req.failure or "unknown reason"
            report.failed_assets.append(f"{req.url} ({failure})")

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    page.on("requestfailed", on_request_failed)

    url = "file://" + urllib.parse.quote(str(abs_path))
    try:
        await page.goto(url, wait_until="networkidle", timeout=30_000)
        report.mathjax = await page.evaluate(_MATHJAX_HARNESS)
    except Exception as e:  # noqa: BLE001
        # A navigation timeout or harness-eval failure is itself a page
        # error -- record it under pageerrors so the build fails.
        report.pageerrors.append(f"goto/evaluate failed: {e}")
    finally:
        await page.close()
    return report


async def _run(build_dir: Path, concurrency: int) -> list[PageReport]:
    from playwright.async_api import async_playwright  # noqa: PLC0415

    pages = list(_iter_html_pages(build_dir))
    if not pages:
        return []

    sem = asyncio.Semaphore(concurrency)
    reports: list[PageReport] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        context = await browser.new_context()

        async def bounded(abs_path, rel):
            async with sem:
                report = await _check_one(context, abs_path, rel)
                reports.append(report)

        try:
            await asyncio.gather(*(bounded(p, r) for r, p in pages))
        finally:
            await context.close()
            await browser.close()
    return reports


def _emit_stderr(report: PageReport) -> None:
    if report.total == 0:
        return
    print(f"\n{report.rel}: {report.total} error(s)", file=sys.stderr)
    for err in report.mathjax:
        msg = err.get("message", "").strip()
        expr = err.get("expr", "").strip()
        if expr:
            print(f"  [mathjax] {msg}\n      in: {expr}", file=sys.stderr)
        else:
            print(f"  [mathjax] {msg}", file=sys.stderr)
    for e in report.pageerrors:
        print(f"  [pageerror] {e}", file=sys.stderr)
    for e in report.console_errors:
        print(f"  [console] {e}", file=sys.stderr)
    for e in report.failed_assets:
        print(f"  [asset] {e}", file=sys.stderr)


def _emit_summary(reports: list[PageReport]) -> str:
    """Build the ``$GITHUB_STEP_SUMMARY`` markdown for failing pages."""
    failing = [r for r in reports if r.total > 0]
    if not failing:
        return ""
    total = sum(r.total for r in failing)
    lines: list[str] = [
        "## Documentation render errors\n",
        f"Found **{total} error(s)** across **{len(failing)} page(s)**. ",
        "Build the docs locally with `doc/scripts/build.sh --clean` to ",
        "reproduce, then re-run `python doc/scripts/check_math.py doc/_build/html`.\n",
    ]
    for r in sorted(failing, key=lambda r: r.rel):
        lines.append(f"\n### `{r.rel}` ({r.total} error(s))\n")
        for err in r.mathjax:
            msg = err.get("message", "").strip()
            expr = err.get("expr", "").strip()
            if expr:
                lines.append(f"- **mathjax:** {msg}\n  ```\n  {expr}\n  ```\n")
            else:
                lines.append(f"- **mathjax:** {msg}\n")
        for e in r.pageerrors:
            lines.append(f"- **pageerror:** {e}\n")
        for e in r.console_errors:
            lines.append(f"- **console:** {e}\n")
        for e in r.failed_assets:
            lines.append(f"- **missing asset:** {e}\n")
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Headless-browser integrity check for the built doc site."
    )
    parser.add_argument("build_dir", help="Sphinx output dir, e.g. doc/_build/html.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of pages to load in parallel (default: 8).",
    )
    args = parser.parse_args(argv)

    build_dir = Path(args.build_dir).resolve()
    if not build_dir.is_dir():
        print(f"BUILD_DIR does not exist: {build_dir}", file=sys.stderr)
        return 2

    try:
        # Import lazily so the script's --help works without playwright.
        import playwright  # noqa: F401, PLC0415
    except ImportError:
        print(
            "playwright is not installed. Run `pip install playwright && "
            "playwright install chromium` and re-invoke.",
            file=sys.stderr,
        )
        return 2

    reports = asyncio.run(_run(build_dir, args.concurrency))

    for r in sorted(reports, key=lambda r: r.rel):
        _emit_stderr(r)

    failing = [r for r in reports if r.total > 0]
    if failing:
        total = sum(r.total for r in failing)
        print(
            f"\n{total} error(s) across {len(failing)} page(s).",
            file=sys.stderr,
        )
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            Path(summary_path).write_text(_emit_summary(reports), encoding="utf-8")
        return 1
    print(f"OK — no errors across {len(reports)} page(s) under {build_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
