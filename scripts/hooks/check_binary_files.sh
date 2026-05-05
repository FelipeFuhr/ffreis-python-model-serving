#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

source "$(dirname "$0")/../lib/common.sh"

has_error=0

common_require_git_repo

is_allowlisted() {
  local path="$1"
  common_is_allowlisted_path "$path"
}

while IFS= read -r -d '' entry; do
  IFS=$'\t' read -r added deleted file <<<"$entry"

  if [[ "$added" != "-" || "$deleted" != "-" ]]; then
    continue
  fi

  if is_allowlisted "$file"; then
    continue
  fi

  common_err "Unexpected staged binary file: ${file}"
  has_error=1
done < <(git diff --cached --numstat --diff-filter=ACM -z)

if [[ "$has_error" -ne 0 ]]; then
  common_err "Binary files are blocked unless they match allowlisted paths/extensions."
  exit 1
fi
