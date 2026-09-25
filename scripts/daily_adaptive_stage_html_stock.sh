#!/usr/bin/env bash
# 股票侧：待选池「按标的自适应」趋势阶段 HTML + ticket_card
#
# 依赖：先跑 daily_regime_transition_validation_stock.sh（至少 ①②，SKIP_HTML=1 即可）
#
# EOD 示例:
#   AS_OF=2026-09-01 PY=${PY:-python} \
#     ./scripts/daily_adaptive_stage_html_stock.sh
#
# 单票调试:
#   CODES=SH600519 AS_OF=2026-09-01 PY=... ./scripts/daily_adaptive_stage_html_stock.sh
#
# 跨日共享训练缓存（默认 ${PLOTLY_ROOT}/_train_cache；关: TRAIN_CACHE_DIR=off）:
#   TRAIN_CACHE_DIR=/path/to/cache AS_OF=... PY=... ./scripts/daily_adaptive_stage_html_stock.sh
#
# 关掉票卡:
#   TICKET_CARD=0 AS_OF=... PY=... ./scripts/daily_adaptive_stage_html_stock.sh
#
set -euo pipefail

source "$(dirname "$0")/daily_env.sh"
PY="${PY:-python}"
VALIDATION_DIR="${VALIDATION_DIR:-${PROJECT_ROOT}/workspace/decision_packs/20260901_stock/regime_transition_validation_q90_stock}"
PLOT_START="${PLOT_START:-2026-04-01}"
TRAIN_CUTOFF="${TRAIN_CUTOFF:-${PLOT_START}}"
LIVE_SNAPSHOT="${LIVE_SNAPSHOT:-}"
CODES="${CODES:-}"
MAX_CODES="${MAX_CODES:-}"
RETRAIN="${RETRAIN:-0}"
JOBS="${JOBS:-8}"
PRIOR_K="${PRIOR_K:-5}"
SCORE_MODE="${SCORE_MODE:-expect_no_trade}"
REGIME_MODE="${REGIME_MODE:-legacy_score_method}"
TRADE_MODE="${TRADE_MODE:-hold_up}"
# Stock: no ETF HRP membership by default → RU diag off.
DISABLE_RU_DIAG="${DISABLE_RU_DIAG:-1}"
FAIR_PATH_EXTRA_TRAIL_YEARS="${FAIR_PATH_EXTRA_TRAIL_YEARS:-1,0.5,0.25,1/12}"
TRAIN_CACHE_DIR="${TRAIN_CACHE_DIR:-${PLOTLY_ROOT}/_train_cache}"

AS_OF="${AS_OF:-$(date +%Y-%m-%d)}"
if [[ ! "${AS_OF}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "[ERROR] AS_OF must be YYYY-MM-DD, got: ${AS_OF}" >&2
  exit 2
fi
if ! date -d "${AS_OF}" >/dev/null 2>&1; then
  echo "[ERROR] AS_OF is not a valid date: ${AS_OF}" >&2
  exit 2
fi
AS_OF_TAG="$(date -d "${AS_OF}" +%Y%m%d)"
HTML_OUT_DIR="${HTML_OUT_DIR:-${PLOTLY_ROOT}/${AS_OF_TAG}all_adaptive_stock}"

cd "${PROJECT_ROOT}"

ARGS=(
  decision_pack/scripts/plot_adaptive_stage_pool.py
  --start-date "${PLOT_START}"
  --end-date "${AS_OF}"
  --train-cutoff "${TRAIN_CUTOFF}"
  --validation-dir "${VALIDATION_DIR}"
  --html-out-dir "${HTML_OUT_DIR}"
  --prior-k "${PRIOR_K}"
  --score-mode "${SCORE_MODE}"
  --regime-mode "${REGIME_MODE}"
  --trade-mode "${TRADE_MODE}"
  --jobs "${JOBS}"
  --continue-on-error
)

_tc="$(echo "${TRAIN_CACHE_DIR}" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${TRAIN_CACHE_DIR}" && "${_tc}" != "off" && "${_tc}" != "none" && "${_tc}" != "0" ]]; then
  ARGS+=(--train-cache-dir "${TRAIN_CACHE_DIR}")
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
_fp_extra="$(echo "${FAIR_PATH_EXTRA_TRAIL_YEARS}" | tr '[:upper:]' '[:lower:]')"
if [[ -n "${FAIR_PATH_EXTRA_TRAIL_YEARS}" && "${_fp_extra}" != "off" && "${_fp_extra}" != "none" && "${_fp_extra}" != "0" ]]; then
  ARGS+=(--fair-path-extra-trail-years "${FAIR_PATH_EXTRA_TRAIL_YEARS}")
fi

echo "[INFO] AS_OF=${AS_OF} PLOT_START=${PLOT_START} TRAIN_CUTOFF=${TRAIN_CUTOFF} RETRAIN=${RETRAIN} SCORE_MODE=${SCORE_MODE} REGIME_MODE=${REGIME_MODE} TRADE_MODE=${TRADE_MODE}"
echo "[INFO] VALIDATION_DIR=${VALIDATION_DIR}"
echo "[INFO] FAIR_PATH_EXTRA_TRAIL_YEARS=${FAIR_PATH_EXTRA_TRAIL_YEARS}"
echo "[INFO] TRAIN_CACHE_DIR=${TRAIN_CACHE_DIR}"
echo "[INFO] JOBS=${JOBS}"
echo "[INFO] HTML_OUT_DIR=${HTML_OUT_DIR}"
"${PY}" "${ARGS[@]}"
echo "[OK] 输出目录: ${HTML_OUT_DIR}"
if [[ "${TICKET_CARD:-1}" != "0" ]]; then
  echo "[INFO] building research ticket_card.csv (TICKET_CARD=0 to skip)"
  "${PY}" decision_pack/scripts/build_daily_ticket_card.py \
    --as-of "${AS_OF}" \
    --adaptive-dir "${HTML_OUT_DIR}" \
    --out "${HTML_OUT_DIR}/ticket_card.csv"
fi
