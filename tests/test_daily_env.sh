#!/usr/bin/env bash
# daily_env.sh must exit before any HTML write when config.env is missing,
# and must export PY and PLOTLY_ROOT when a temp config exists.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_SH="${ROOT}/scripts/daily_env.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# Isolate from the real repo config by running a copy of the script layout.
mkdir -p "${TMP}/scripts"
cp "${ENV_SH}" "${TMP}/scripts/daily_env.sh"
# Point PROJECT_ROOT at the temp tree: the script derives it from its own path.

set +e
bash -c "source '${TMP}/scripts/daily_env.sh'" >"${TMP}/out.txt" 2>"${TMP}/err.txt"
code=$?
set -e
if [[ "${code}" -eq 0 ]]; then
  echo "expected non-zero exit when config.env is missing" >&2
  exit 1
fi
if ! grep -q "config.env.example" "${TMP}/err.txt"; then
  echo "error message should name config.env.example" >&2
  cat "${TMP}/err.txt" >&2
  exit 1
fi

cat > "${TMP}/config.env" << 'EOF'
PY=/usr/bin/python3
PLOTLY_ROOT=/tmp/ashare-plotly-test
EOF
# shellcheck disable=SC1091
eval "$(bash -c "source '${TMP}/scripts/daily_env.sh'; printf 'PY=%q\nPLOTLY_ROOT=%q\nPROJECT_ROOT=%q\n' \"\$PY\" \"\$PLOTLY_ROOT\" \"\$PROJECT_ROOT\"")"
if [[ "${PY}" != "/usr/bin/python3" ]]; then
  echo "PY not exported from config.env, got ${PY}" >&2
  exit 1
fi
if [[ "${PLOTLY_ROOT}" != "/tmp/ashare-plotly-test" ]]; then
  echo "PLOTLY_ROOT not exported, got ${PLOTLY_ROOT}" >&2
  exit 1
fi
if [[ "${PROJECT_ROOT}" != "${TMP}" ]]; then
  echo "PROJECT_ROOT should be the temp repo, got ${PROJECT_ROOT}" >&2
  exit 1
fi
echo "test_daily_env.sh ok"
