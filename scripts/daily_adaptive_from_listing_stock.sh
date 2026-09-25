#!/usr/bin/env bash
# 股票侧：每日生成 from_listing 自适应 HTML（上市日 → AS_OF）
#
# EOD 增量（默认：复用最近一个 *_from_listing_stock，按缺失交易日逐根补 K 线；
# 隔多日也不会整图重画。上一份正好是前一交易日时，与 ETF 一样只补一根）:
#   PY=${PY:-python} ./scripts/daily_adaptive_from_listing_stock.sh
#   AS_OF=2026-09-23 JOBS=8 PY=... ./scripts/daily_adaptive_from_listing_stock.sh
#
# 强制全量重画:
#   INCREMENTAL=0 AS_OF=2026-09-23 PY=... ./scripts/daily_adaptive_from_listing_stock.sh
#
# 手动指定昨日目录（跳过自动发现）:
#   PREV_LISTING_DIR=${PLOTLY_ROOT}/20260922_from_listing_stock \
#     AS_OF=2026-09-23 PY=... ./scripts/daily_adaptive_from_listing_stock.sh
#
# 单票调试:
#   CODES=SH600519 AS_OF=2026-09-23 PY=... ./scripts/daily_adaptive_from_listing_stock.sh
#
# 共享训练缓存 / 上市日缓存（默认开启）:
#   TRAIN_CACHE_DIR=off LISTING_CACHE=off 可关
#
# 图12 六状态背景（默认开，后处理重画；FIG12=0 可关）:
#   FIG12=0 AS_OF=... PY=... ./scripts/daily_adaptive_from_listing_stock.sh
#
set -euo pipefail

source "$(dirname "$0")/daily_env.sh"
# 新目录还没有昨日 HTML 时，回退到仓库里已有的 from_listing（只读，不往仓库写）。
LEGACY_PLOTLY_ROOT="${LEGACY_PLOTLY_ROOT:-${PROJECT_ROOT}/workspace/plotly_outputs}"
PY="${PY:-python}"
AS_OF="${AS_OF:-$(date +%Y-%m-%d)}"
TRAIN_CUTOFF="${TRAIN_CUTOFF:-2026-04-01}"
VALIDATION_DIR="${VALIDATION_DIR:-${PROJECT_ROOT}/workspace/decision_packs/20260901_stock/regime_transition_validation_q90_stock}"
CONFIG_SOURCE_DIR="${CONFIG_SOURCE_DIR:-}"
HTML_OUT_DIR="${HTML_OUT_DIR:-}"
CODES="${CODES:-}"
MAX_CODES="${MAX_CODES:-}"
RETRAIN="${RETRAIN:-0}"
JOBS="${JOBS:-8}"
LIVE_SNAPSHOT="${LIVE_SNAPSHOT:-}"
# Stock: no ETF HRP membership by default → RU diag off.
DISABLE_RU_DIAG="${DISABLE_RU_DIAG:-1}"
# INCREMENTAL=1 (default) → html mode; INCREMENTAL=0 → full rebuild.
INCREMENTAL="${INCREMENTAL:-1}"
INCREMENTAL_MODE="${INCREMENTAL_MODE:-}"
INCREMENTAL_FROM="${INCREMENTAL_FROM:-}"
PREV_LISTING_DIR="${PREV_LISTING_DIR:-}"
FAIR_PATH_EXTRA_TRAIL_YEARS="${FAIR_PATH_EXTRA_TRAIL_YEARS:-1,0.5,0.25,1/12}"
TRAIN_CACHE_DIR="${TRAIN_CACHE_DIR:-${PLOTLY_ROOT}/_train_cache}"
LISTING_CACHE="${LISTING_CACHE:-${PLOTLY_ROOT}/_listing_dates_stock.csv}"
FIG12="${FIG12:-1}"

if [[ ! "${AS_OF}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "[ERROR] AS_OF must be YYYY-MM-DD, got: ${AS_OF}" >&2
  exit 2
fi
if ! date -d "${AS_OF}" >/dev/null 2>&1; then
  echo "[ERROR] AS_OF is not a valid date: ${AS_OF}" >&2
  exit 2
fi
AS_OF_TAG="$(date -d "${AS_OF}" +%Y%m%d)"
CONFIG_SOURCE_DIR="${CONFIG_SOURCE_DIR:-${PLOTLY_ROOT}/${AS_OF_TAG}all_adaptive_stock}"
HTML_OUT_DIR="${HTML_OUT_DIR:-${PLOTLY_ROOT}/${AS_OF_TAG}_from_listing_stock}"

# Resolve incremental mode: INCREMENTAL_MODE wins if set; else INCREMENTAL=1 → html.
if [[ -z "${INCREMENTAL_MODE}" ]]; then
  if [[ "${INCREMENTAL}" == "0" || "${INCREMENTAL}" == "false" || "${INCREMENTAL}" == "off" || "${INCREMENTAL}" == "no" ]]; then
    INCREMENTAL_MODE="off"
  else
    INCREMENTAL_MODE="html"
  fi
fi

find_prev_listing_dir() {
  local as_of_tag="$1"
  local root
  local best=""
  local best_tag=""
  shopt -s nullglob
  for root in "${PLOTLY_ROOT}" "${LEGACY_PLOTLY_ROOT}"; do
    [[ -n "${root}" && -d "${root}" ]] || continue
    for d in "${root}"/*_from_listing_stock; do
      [[ -d "${d}" ]] || continue
      [[ -f "${d}/listing_dates.csv" ]] || continue
      local base
      base="$(basename "${d}")"
      # expect YYYYMMDD_from_listing_stock
      local tag="${base%_from_listing_stock}"
      [[ "${tag}" =~ ^[0-9]{8}$ ]] || continue
      if [[ "${tag}" < "${as_of_tag}" ]]; then
        if [[ -z "${best_tag}" || "${tag}" > "${best_tag}" ]]; then
          best_tag="${tag}"
          best="${d}"
        fi
      fi
    done
  done
  shopt -u nullglob
  echo "${best}"
}

if [[ "${INCREMENTAL_MODE}" == "html" ]]; then
  if [[ -n "${INCREMENTAL_FROM}" ]]; then
    PREV_LISTING_DIR="${INCREMENTAL_FROM}"
  elif [[ -n "${PREV_LISTING_DIR}" ]]; then
    :
  else
    PREV_LISTING_DIR="$(find_prev_listing_dir "${AS_OF_TAG}")"
  fi
  if [[ -z "${PREV_LISTING_DIR}" || ! -d "${PREV_LISTING_DIR}" ]]; then
    echo "[WARN] no previous *_from_listing_stock found; falling back to full rebuild"
    INCREMENTAL_MODE="off"
    PREV_LISTING_DIR=""
  fi
fi

cd "${PROJECT_ROOT}"

ARGS=(
  decision_pack/scripts/run_from_listing_sharded.py
  --as-of "${AS_OF}"
  --train-cutoff "${TRAIN_CUTOFF}"
  --validation-dir "${VALIDATION_DIR}"
  --config-source-dir "${CONFIG_SOURCE_DIR}"
  --html-out-dir "${HTML_OUT_DIR}"
  --jobs "${JOBS}"
  --incremental-mode "${INCREMENTAL_MODE}"
)

_tc="$(echo "${TRAIN_CACHE_DIR}" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${TRAIN_CACHE_DIR}" && "${_tc}" != "off" && "${_tc}" != "none" && "${_tc}" != "0" ]]; then
  ARGS+=(--train-cache-dir "${TRAIN_CACHE_DIR}")
else
  ARGS+=(--no-train-cache)
fi
_lc="$(echo "${LISTING_CACHE}" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${LISTING_CACHE}" && "${_lc}" != "off" && "${_lc}" != "none" && "${_lc}" != "0" ]]; then
  ARGS+=(--listing-cache "${LISTING_CACHE}")
else
  ARGS+=(--no-listing-cache)
fi
if [[ "${RETRAIN}" == "1" || "${RETRAIN}" == "true" || "${RETRAIN}" == "yes" ]]; then
  ARGS+=(--retrain)
fi
if [[ "${DISABLE_RU_DIAG}" == "1" || "${DISABLE_RU_DIAG}" == "true" || "${DISABLE_RU_DIAG}" == "yes" ]]; then
  ARGS+=(--disable-ru-diag)
fi
if [[ -n "${LIVE_SNAPSHOT}" ]]; then
  ARGS+=(--live-snapshot "${LIVE_SNAPSHOT}")
fi
if [[ -n "${MAX_CODES}" ]]; then
  ARGS+=(--max-codes "${MAX_CODES}")
fi
if [[ -n "${CODES}" ]]; then
  for c in ${CODES}; do
    ARGS+=(--code "${c}")
  done
fi
if [[ "${INCREMENTAL_MODE}" == "html" && -n "${PREV_LISTING_DIR}" ]]; then
  ARGS+=(--incremental-from "${PREV_LISTING_DIR}")
fi
_fp_extra="$(echo "${FAIR_PATH_EXTRA_TRAIL_YEARS}" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${FAIR_PATH_EXTRA_TRAIL_YEARS}" && "${_fp_extra}" != "off" && "${_fp_extra}" != "none" && "${_fp_extra}" != "0" ]]; then
  ARGS+=(--fair-path-extra-trail-years "${FAIR_PATH_EXTRA_TRAIL_YEARS}")
fi

echo "[INFO] AS_OF=${AS_OF} TRAIN_CUTOFF=${TRAIN_CUTOFF} JOBS=${JOBS} RETRAIN=${RETRAIN}"
echo "[INFO] CONFIG_SOURCE_DIR=${CONFIG_SOURCE_DIR}"
echo "[INFO] HTML_OUT_DIR=${HTML_OUT_DIR}"
echo "[INFO] INCREMENTAL_MODE=${INCREMENTAL_MODE} PREV_LISTING_DIR=${PREV_LISTING_DIR:-}"
echo "[INFO] TRAIN_CACHE_DIR=${TRAIN_CACHE_DIR}"
echo "[INFO] LISTING_CACHE=${LISTING_CACHE}"
echo "[INFO] FIG12=${FIG12}"
PYTHONPATH=. "${PY}" "${ARGS[@]}"

EWMA_ARGS=(
  decision_pack/scripts/add_ewma_rails.py
  --html-dir "${HTML_OUT_DIR}"
  --jobs "${JOBS}"
)
if [[ -n "${CODES}" ]]; then
  for c in ${CODES}; do
    EWMA_ARGS+=(--code "${c}")
  done
fi
echo "[INFO] ewma rails html-dir=${HTML_OUT_DIR} jobs=${JOBS}"
PYTHONPATH=. "${PY}" "${EWMA_ARGS[@]}"

_fig12="$(echo "${FIG12}" | tr '[:upper:]' '[:lower:]')"
if [[ "${_fig12}" != "0" && "${_fig12}" != "false" && "${_fig12}" != "off" && "${_fig12}" != "no" ]]; then
  FIG12_ARGS=(
    decision_pack/scripts/add_fig12_six_states.py
    --html-dir "${HTML_OUT_DIR}"
    --jobs "${JOBS}"
  )
  if [[ -n "${CODES}" ]]; then
    for c in ${CODES}; do
      FIG12_ARGS+=(--code "${c}")
    done
  fi
  echo "[INFO] fig12 post-process html-dir=${HTML_OUT_DIR} jobs=${JOBS}"
  PYTHONPATH=. "${PY}" "${FIG12_ARGS[@]}"
fi
echo "[OK] 输出目录: ${HTML_OUT_DIR}"
