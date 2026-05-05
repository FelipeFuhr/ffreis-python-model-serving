#!/usr/bin/env sh
set -eu

LEFTHOOK_VERSION="${LEFTHOOK_VERSION:-1.7.10}"
BIN_DIR="${BIN_DIR:-.bin}"
BIN="$BIN_DIR/lefthook"

mkdir -p "$BIN_DIR"

if [ ! -x "$BIN" ]; then
  echo "Downloading lefthook v$LEFTHOOK_VERSION ..."

  OS="$(uname -s)"
  ARCH="$(uname -m)"

  case "$OS" in
    Linux) OS=Linux ;;
    Darwin) OS=Darwin ;;
    *)
      echo "Unsupported OS: $OS" >&2
      exit 2
      ;;
  esac

  case "$ARCH" in
    x86_64|amd64) ARCH=x86_64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *)
      echo "Unsupported arch: $ARCH" >&2
      exit 2
      ;;
  esac

  URL="https://github.com/evilmartians/lefthook/releases/download/v${LEFTHOOK_VERSION}/lefthook_${LEFTHOOK_VERSION}_${OS}_${ARCH}"

  EXPECTED_CHECKSUM=""
  case "${OS}_${ARCH}" in
    Linux_x86_64) EXPECTED_CHECKSUM="33e1a7a8af4bcb0215d54ecdbc78e82b125d00e46e95a779838ec81ea39136d2" ;;
    Linux_arm64)  EXPECTED_CHECKSUM="ea20bc6c9055b45f640104e860e0beb83368ad9efcda85efef513fc70f3992c3" ;;
    Darwin_x86_64) EXPECTED_CHECKSUM="a6423e3efc6e48fdf0b43b81ff63d9e57b3d3e175bd22cf5e880b9a8b779eb8d" ;;
    Darwin_arm64)  EXPECTED_CHECKSUM="05cab0767664461aff1e86b4df9402565b81825fac574ccfcac822c60c8bd015" ;;
    *)
      echo "No checksum available for ${OS}_${ARCH}" >&2
      exit 2
      ;;
  esac

  TMP_BIN="${BIN}.tmp"
  curl --fail --show-error --silent --location \
    --proto '=https' \
    --tlsv1.2 \
    "$URL" -o "$TMP_BIN"

  if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL_CHECKSUM="$(sha256sum "$TMP_BIN" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    ACTUAL_CHECKSUM="$(shasum -a 256 "$TMP_BIN" | awk '{print $1}')"
  else
    echo "Neither sha256sum nor shasum found; cannot verify binary integrity." >&2
    rm -f "$TMP_BIN"
    exit 2
  fi

  if [ "$ACTUAL_CHECKSUM" != "$EXPECTED_CHECKSUM" ]; then
    echo "Checksum mismatch for lefthook v${LEFTHOOK_VERSION} ${OS}_${ARCH}:" >&2
    echo "  expected: $EXPECTED_CHECKSUM" >&2
    echo "  actual:   $ACTUAL_CHECKSUM" >&2
    rm -f "$TMP_BIN"
    exit 1
  fi

  mv "$TMP_BIN" "$BIN"
  chmod +x "$BIN"
fi

echo "Lefthook available at: $BIN"
