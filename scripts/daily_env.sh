# Shared startup for the three stock daily shells.
# Reads only PY and PLOTLY_ROOT from config.env. A value already set in the
# environment wins, so `PY=... ./scripts/daily_*.sh` still works.
_DAILY_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_DAILY_ENV_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

_CONFIG_ENV="${PROJECT_ROOT}/config.env"
if [[ ! -f "${_CONFIG_ENV}" ]]; then
  echo "[ERROR] missing ${_CONFIG_ENV}. Copy ${_CONFIG_ENV}.example and set PY and PLOTLY_ROOT." >&2
  exit 1
fi

_PRE_PY="${PY:-}"
_PRE_PLOTLY="${PLOTLY_ROOT:-}"
set -a
# shellcheck disable=SC1090
source "${_CONFIG_ENV}"
set +a
if [[ -n "${_PRE_PY}" ]]; then
  PY="${_PRE_PY}"
fi
if [[ -n "${_PRE_PLOTLY}" ]]; then
  PLOTLY_ROOT="${_PRE_PLOTLY}"
fi
if [[ -z "${PY:-}" || -z "${PLOTLY_ROOT:-}" ]]; then
  echo "[ERROR] ${_CONFIG_ENV} must set PY and PLOTLY_ROOT. See ${_CONFIG_ENV}.example." >&2
  exit 1
fi
