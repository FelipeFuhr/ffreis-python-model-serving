#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

commit_msg_file="${1:-}"
if [ -z "${commit_msg_file}" ] || [ ! -f "${commit_msg_file}" ]; then
  echo "Commit message file is required." >&2
  echo "Usage: $0 <commit-msg-file>" >&2
  exit 1
fi

subject="$(grep -m1 -vE '^\s*(#|$)' "${commit_msg_file}" || true)"
if [ -z "${subject}" ]; then
  echo "Commit message subject cannot be empty." >&2
  exit 1
fi

if [[ "${subject}" =~ ^(<<<<<<<|=======|>>>>>>>) ]]; then
  echo "Commit message contains merge conflict markers." >&2
  exit 1
fi

if [[ "${subject}" =~ ^Merge[[:space:]] ]]; then
  exit 0
fi

if [ "${#subject}" -gt 72 ]; then
  echo "Commit subject is too long (${#subject} > 72)." >&2
  exit 1
fi

pattern='^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-z0-9._/-]+\))?: .+'
if ! [[ "${subject}" =~ ${pattern} ]]; then
  echo "Commit message must follow Conventional Commits:" >&2
  echo "  <type>(optional-scope): <subject>" >&2
  echo "Allowed types: feat fix docs style refactor perf test build ci chore revert" >&2
  echo "Example: feat(converter): add ONNX opset 18 support" >&2
  exit 1
fi
