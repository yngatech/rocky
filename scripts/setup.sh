#!/usr/bin/env bash
# Set up a clone or worktree: pinned Python and uv via mise, dependencies, and the live checkout's files.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v mise >/dev/null; then
  mise trust --quiet mise.toml
  mise install
  eval "$(mise env --shell bash)"
  # Put the pinned uv and Python ahead of any others already on PATH (e.g. a standalone uv in ~/.local/bin).
  PATH="$(mise bin-paths | paste -sd: -):$PATH"
else
  echo "warning: mise not installed; using $(uv --version); pins are in mise.toml and .python-version" >&2
fi

uv run --with-requirements requirements.txt python -c ''

# In a worktree, link .env from the main clone (a symlink: never write test data to it).
main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
[ -e .env ] || [ -L .env ] || [ ! -e "$main/.env" ] || ln -s "$main/.env" .env
