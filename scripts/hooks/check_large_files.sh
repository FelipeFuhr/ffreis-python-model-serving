#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

source "$(dirname "$0")/../lib/common.sh"

MAX_SIZE_BYTES="${MAX_SIZE_BYTES:-1048576}"
has_error=0

common_require_git_repo

is_allowlisted() {
  local path="$1"
  common_is_allowlisted_path "$path"
}

while IFS= read -r -d '' file; do
  size="$(git cat-file -s ":${file}")"
  if [[ -z "$size" ]]; then
    continue
  fi

  if [[ "$size" -le "$MAX_SIZE_BYTES" ]]; then
    continue
  fi

  if is_allowlisted "$file"; then
    continue
  fi

  common_err "Staged file exceeds ${MAX_SIZE_BYTES} bytes: ${file} (${size} bytes)"
  has_error=1
done < <(git diff --cached --name-only --diff-filter=ACM -z)

if [[ "$has_error" -ne 0 ]]; then
  common_err "Large staged files must be split, compressed, or explicitly allowlisted."
  exit 1
fi
