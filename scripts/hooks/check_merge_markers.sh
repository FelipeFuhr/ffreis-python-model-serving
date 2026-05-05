#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

source "$(dirname "$0")/../lib/common.sh"

has_error=0
found_any=0
tmp_output="$(mktemp)"
trap 'rm -f "${tmp_output}"' EXIT

common_require_git_repo

while IFS= read -r -d '' file; do
  found_any=1
  if git show ":${file}" | grep -nE '^(<<<<<<< |=======|>>>>>>> )' >"${tmp_output}" 2>/dev/null; then
    common_err "Merge conflict markers detected in staged file: ${file}"
    sed 's/^/  /' "${tmp_output}" >&2
    has_error=1
  fi
done < <(git diff --cached --name-only --diff-filter=ACM -z)

if [[ "$found_any" -eq 0 ]]; then
  exit 0
fi

if [[ "$has_error" -ne 0 ]]; then
  common_err "Resolve conflict markers before committing."
  exit 1
fi
