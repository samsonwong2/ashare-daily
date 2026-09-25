#!/usr/bin/env bash
# 股票侧：每日 regime transition 验证（①信号 + ②反转标注 + 可选③非自适应 HTML）
#
# EOD 完整链路:
#   PY=${PY:-python} ./scripts/daily_regime_transition_validation_stock.sh
#   AS_OF=2026-09-01 PY=... ./scripts/daily_regime_transition_validation_stock.sh
#
# 只跑 ①②、跳过 ③（给 adaptive HTML 用）:
#   SKIP_HTML=1 AS_OF=2026-09-01 PY=... ./scripts/daily_regime_transition_validation_stock.sh
#
# 仅跑 ①（当月按天增量：保留 signals.csv，只清 .done）:
#   SKIP_REBUILD=1 SKIP_HTML=1 AS_OF=2026-09-23 JOBS=8 PY=... \
#     ./scripts/daily_regime_transition_validation_stock.sh
#
# 整月重算兜底:
#   FORCE_MONTH_REBUILD=1 AS_OF=2026-09-23 PY=... ./scripts/daily_regime_transition_validation_stock.sh
#
set -euo pipefail

source "$(dirname "$0")/daily_env.sh"
FORMAL_OUT_DIR="${PROJECT_ROOT}/workspace/decision_packs/20260901_stock/regime_transition_validation_q90_stock"
CACHE_DIR="${PROJECT_ROOT}/workspace/decision_packs/regime_transition_model_cache_stock"
PY="${PY:-python}"
PLOT_START="${PLOT_START:-2026-04-01}"
SKIP_REBUILD="${SKIP_REBUILD:-0}"
SKIP_HTML="${SKIP_HTML:-0}"
JOBS="${JOBS:-8}"
FORCE_MONTH_REBUILD="${FORCE_MONTH_REBUILD:-0}"
HTML_POOL_DIR="${HTML_POOL_DIR:-}"

AS_OF="${AS_OF:-$(date +%Y-%m-%d)}"
if [[ ! "${AS_OF}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
  echo "[ERROR] AS_OF must be YYYY-MM-DD, got: ${AS_OF}" >&2
  exit 2
fi
if ! date -d "${AS_OF}" >/dev/null 2>&1; then
  echo "[ERROR] AS_OF is not a valid date: ${AS_OF}" >&2
  exit 2
fi
MONTH="${AS_OF:0:7}"
AS_OF_TAG="$(date -d "${AS_OF}" +%Y%m%d)"

cd "${PROJECT_ROOT}"

OUT_DIR="${FORMAL_OUT_DIR}"
HTML_OUT_DIR="${HTML_POOL_DIR:-${PLOTLY_ROOT}/${AS_OF_TAG}all_stock}"
mkdir -p "${OUT_DIR}/shards" "${CACHE_DIR}"

echo "[INFO] EOD stock mode AS_OF=${AS_OF} MONTH=${MONTH} OUT_DIR=${OUT_DIR} JOBS=${JOBS}"
echo "[INFO] HTML_OUT_DIR=${HTML_OUT_DIR} PLOT_START=${PLOT_START}"

if [[ "${FORCE_MONTH_REBUILD}" == "1" || "${FORCE_MONTH_REBUILD}" == "true" || "${FORCE_MONTH_REBUILD}" == "yes" ]]; then
  rm -f "${OUT_DIR}/shards/${MONTH}.done"
  rm -f "${OUT_DIR}/shards/${MONTH}.signals.csv"
  echo "[INFO] FORCE_MONTH_REBUILD=1: cleared ${MONTH} shards under ${OUT_DIR}/shards"
else
  rm -f "${OUT_DIR}/shards/${MONTH}.done"
  echo "[INFO] day-resume: cleared ${MONTH}.done (kept ${MONTH}.signals.csv if present)"
fi

echo ""
echo "=== ① regime transition validation (stock) ==="
"${PY}" decision_pack/scripts/backtest_regime_transition_signals.py \
  --output-dir "${OUT_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --eval-start 2024-06-01 \
  --eval-end "${AS_OF}" \
  --horizon 10 \
  --jobs "${JOBS}"

if [[ "${SKIP_REBUILD}" != "1" ]]; then
  echo ""
  echo "=== ② rebuild reversal labels + event match ==="
  "${PY}" decision_pack/scripts/rebuild_reversal_event_metrics.py \
    --validation-dir "${OUT_DIR}"
else
  echo "[SKIP] ② rebuild_reversal_event_metrics (SKIP_REBUILD=1)"
fi

if [[ "${SKIP_HTML}" != "1" ]]; then
  echo ""
  echo "=== ③ plot pool HTML (non-adaptive) ==="
  mkdir -p "${HTML_OUT_DIR}"
  "${PY}" decision_pack/scripts/plot_regime_transition_example.py \
    --validation-dir "${OUT_DIR}" \
    --start-date "${PLOT_START}" \
    --end-date "${AS_OF}" \
    --html-out-dir "${HTML_OUT_DIR}" \
    --continue-on-error
else
  echo "[SKIP] ③ plot HTML (SKIP_HTML=1)"
fi

echo ""
echo "=== 运行完成 ==="
echo "mode:     EOD (stock)"
echo "AS_OF:    ${AS_OF}"
echo "输出目录: ${OUT_DIR}"
echo "信号文件: ${OUT_DIR}/signals_oos.csv"
echo "报告:     ${OUT_DIR}/REGIME_TRANSITION_VALIDATION.md"
if [[ -d "${HTML_OUT_DIR}" ]]; then
  echo "HTML目录: ${HTML_OUT_DIR}"
  echo "HTML数量: $(find "${HTML_OUT_DIR}" -maxdepth 1 -name 'regime_transition_*.html' 2>/dev/null | wc -l)"
fi
echo "go/no-go: $("${PY}" -c "import json; print(json.load(open('${OUT_DIR}/go_nogo.json'))['pass'])" 2>/dev/null || echo 'N/A')"
