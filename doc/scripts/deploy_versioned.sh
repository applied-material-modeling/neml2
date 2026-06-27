#!/usr/bin/env bash
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

# Deploy a freshly built documentation site into the versioned gh-pages layout.
#
#   /<repo>/
#     stable/         the newest release (the "stable" pointer)
#     v3.0/ v2.1/ ..  frozen older minors (one folder per MAJOR.MINOR)
#     dev/            main-branch build (managed elsewhere; never touched here)
#     pr-preview/N/   PR previews        (managed elsewhere; never touched here)
#     versions.json   { "stable": "v3.0", "versions": ["v3.0", ...] }
#     index.html      redirect -> stable/
#     .nojekyll
#
# To save a duplicate copy, the newest minor's content lives ONLY in stable/. On
# a minor (or major) bump we rename the current stable/ to its own v<minor>/
# folder (freezing it -- its baked version label and Colab tag are already
# correct) and drop the new build into stable/. A patch of the current stable
# overwrites stable/ in place; a patch of an older minor overwrites only that
# v<minor>/ folder and leaves stable/ alone.
#
# Usage:
#   deploy_versioned.sh --site DIR --version X.Y.Z --pages DIR [options]
#
#   --site DIR        freshly built HTML (e.g. the downloaded artifact)
#   --version X.Y.Z   released project version (from `dep_manager.py get neml2.version`)
#   --pages DIR       a checkout/working tree of the gh-pages branch (committed here)
#   --no-push         commit but do not push (for local dry-runs / tests)
#   --remote NAME     git remote to push to (default: origin)
#   --branch NAME     branch to push (default: gh-pages)
#
# Version reads go through dep_manager.py in the caller; nothing is hardcoded here.

set -euo pipefail

SITE=""
VERSION=""
PAGES=""
PUSH=1
REMOTE="origin"
BRANCH="gh-pages"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --site) SITE="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --pages) PAGES="$2"; shift 2 ;;
    --no-push) PUSH=0; shift ;;
    --remote) REMOTE="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    *) echo "deploy_versioned.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$SITE" && -d "$SITE" ]] || { echo "error: --site must be an existing directory" >&2; exit 2; }
[[ -n "$PAGES" && -d "$PAGES" ]] || { echo "error: --pages must be an existing directory" >&2; exit 2; }
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]] || { echo "error: --version must be X.Y.Z, got '$VERSION'" >&2; exit 2; }

# Derived minor folder name, e.g. 3.0.4 -> v3.0
MINOR="v$(echo "$VERSION" | cut -d. -f1,2)"

# Absolute path to the built site so we can `cp` after cd'ing into the pages dir.
SITE="$(cd "$SITE" && pwd)"

# True when $1 (a vX.Y) is strictly newer than $2 (a vX.Y).
version_gt() {
  local a="${1#v}" b="${2#v}"
  [[ "$a" != "$b" ]] && [[ "$(printf '%s\n%s\n' "$a" "$b" | sort -V | tail -1)" == "$a" ]]
}

# Replace the contents of directory $1 with the built site (sans build metadata).
populate() {
  local dest="$1"
  rm -rf "$dest"
  mkdir -p "$dest"
  cp -R "$SITE"/. "$dest"/
  rm -f "$dest"/_buildinfo.json   # internal routing metadata, not for publishing
}

cd "$PAGES"

# Current stable minor (empty on a fresh / freshly migrated branch).
STABLE=""
if [[ -f versions.json ]]; then
  STABLE="$(python3 -c 'import json,sys; print(json.load(open("versions.json")).get("stable",""))')"
fi

if [[ -z "$STABLE" ]]; then
  echo "No stable recorded yet -> publishing $MINOR as stable/"
  populate stable
elif [[ "$MINOR" == "$STABLE" ]]; then
  echo "Patch of current stable ($STABLE) -> overwriting stable/"
  populate stable
elif version_gt "$MINOR" "$STABLE"; then
  echo "Minor/major bump $STABLE -> $MINOR: freezing stable/ as $STABLE/, new build to stable/"
  # Defensive: under the rename scheme a v<oldstable> folder should not exist
  # yet, but clear any stray one (working tree + index) so the rename can't fail.
  rm -rf "$STABLE"
  git rm -r --cached --ignore-unmatch "$STABLE" >/dev/null 2>&1 || true
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1 && git ls-files --error-unmatch stable >/dev/null 2>&1; then
    git mv stable "$STABLE"
  else
    mv stable "$STABLE"
  fi
  populate stable
else
  echo "Patch of older minor $MINOR (stable is $STABLE) -> overwriting $MINOR/ only"
  populate "$MINOR"
fi

# Root files: redirect + Jekyll opt-out. Kept on every deploy so they survive.
cat > index.html <<'HTML'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url=./stable/">
  <link rel="canonical" href="./stable/">
  <title>NEML2 documentation</title>
  <script>location.replace("./stable/" + location.search + location.hash);</script>
</head>
<body>
  <p>Redirecting to the <a href="./stable/">latest NEML2 documentation</a>.</p>
</body>
</html>
HTML
touch .nojekyll

# Regenerate versions.json from the actual folders on disk + the stable minor,
# so the manifest can never drift from reality. (The stable minor has no folder
# of its own while it is the pointer, hence it is unioned in explicitly.)
python3 - "$MINOR" "$STABLE" <<'PY'
import json, os, re, sys
released, prev_stable = sys.argv[1], sys.argv[2]
# stable is the newest of: the just-released minor, and any prior stable.
stable = released
if prev_stable and not (
    [int(x) for x in released[1:].split(".")] > [int(x) for x in prev_stable[1:].split(".")]
):
    # released is a patch of (or older than) the existing stable -> keep stable.
    stable = prev_stable
folders = [d for d in os.listdir(".") if re.match(r"^v\d+\.\d+$", d) and os.path.isdir(d)]
allv = sorted(set(folders) | {stable}, key=lambda s: [int(x) for x in s[1:].split(".")], reverse=True)
json.dump({"stable": stable, "versions": allv}, open("versions.json", "w"), indent=2)
open("versions.json", "a").write("\n")
print(f"versions.json: stable={stable} versions={allv}")
PY

git add -A
if git diff --cached --quiet; then
  echo "No changes to deploy."
  exit 0
fi

git -c user.name="github-actions[bot]" \
    -c user.email="github-actions[bot]@users.noreply.github.com" \
    commit -q -m "Deploy docs $VERSION ($MINOR)"

if [[ "$PUSH" == "1" ]]; then
  # Releases are rare but the deploy can race a concurrent gh-pages write; retry
  # by rebasing onto the latest remote tip before giving up.
  for attempt in 1 2 3; do
    if git push "$REMOTE" "$BRANCH"; then
      echo "Pushed to $REMOTE/$BRANCH."
      exit 0
    fi
    echo "Push rejected (attempt $attempt) -> fetch + rebase + retry"
    git fetch "$REMOTE" "$BRANCH"
    git rebase "$REMOTE/$BRANCH"
  done
  echo "error: failed to push after retries" >&2
  exit 1
fi

echo "Committed (push skipped)."
