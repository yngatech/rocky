#!/usr/bin/env bash
#
# Prepare a clone or agent worktree for development:
#   - in a worktree, link .env and custom_vocab.txt from the main clone
#   - install the pinned toolchain (Python, uv) with mise
#   - fetch the dependencies in requirements.txt into uv's cache
#
# Idempotent, safe to re-run: `scripts/setup.sh`.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

main_worktree=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")

warnings=()

step() { printf '\n==> %s\n' "$1"; }
warn() {
  printf '!!  %s\n' "$1"
  warnings[${#warnings[@]}]="$1"
}

if [ "$main_worktree" != "$repo_root" ]; then
  # Symlinks, not copies: the live bot's files, never written to from a worktree (see AGENTS.md).
  step "Linking gitignored files from $main_worktree"
  for file in .env custom_vocab.txt; do
    if [ -e "$file" ] && [ ! -L "$file" ]; then
      warn "./$file is a real file; leaving it alone"
    elif [ -e "$main_worktree/$file" ]; then
      ln -sfn "$main_worktree/$file" "$file"
      echo "    $file -> $main_worktree/$file"
    else
      warn "no $file at $main_worktree/$file"
    fi
  done
fi

step "Installing the pinned toolchain"
if command -v mise >/dev/null 2>&1; then
  mise trust --quiet mise.toml
  mise install
  # A uv or python earlier on PATH than mise's shims (such as uv's own installer
  # puts in ~/.local/bin) shadows the pin in a plain shell.
  for tool in python uv; do
    pinned=$(mise current "$tool")
    actual=$("$tool" --version 2>/dev/null | awk '{print $2}') || true
    if [ "$actual" != "$pinned" ]; then
      warn "$(command -v "$tool" || echo "$tool") is ${actual:-missing}, not the pinned $pinned; put mise first on PATH or use \`mise exec --\`"
    fi
  done
  run=(mise exec --)
else
  warn "mise is not installed; using the uv on PATH instead of the pin in mise.toml (see README.md#setup)"
  run=()
fi

step "Fetching dependencies"
"${run[@]}" uv --version
"${run[@]}" uv run --with-requirements requirements.txt python --version

if [ ${#warnings[@]} -ne 0 ]; then
  printf '\n==> Warnings\n'
  for warning in "${warnings[@]}"; do
    printf '!!  %s\n' "$warning"
  done
fi

printf '\nSetup complete.\n'
