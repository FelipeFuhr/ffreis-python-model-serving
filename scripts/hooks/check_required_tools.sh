#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <tool> [tool...]" >&2
  exit 1
fi

missing=0

install_hint() {
  case "$1" in
    ruff)
      echo "pip install ruff  or  uv tool install ruff"
      ;;
    mypy)
      echo "pip install mypy  or  uv tool install mypy"
      ;;
    gitleaks)
      echo "https://github.com/gitleaks/gitleaks#installing"
      ;;
    uv)
      echo "https://docs.astral.sh/uv/getting-started/installation/"
      ;;
    *)
      echo "Install '$1' and ensure it is available in PATH."
      ;;
  esac
}

for tool in "$@"; do
  if command -v "$tool" >/dev/null 2>&1; then
    continue
  fi
  echo "Missing required tool: $tool" >&2
  echo "Install hint: $(install_hint "$tool")" >&2
  missing=1
done

exit "$missing"
