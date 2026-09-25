"""Hardcoded lists, output paths and numeric thresholds.

Extracted from ``2_filter/0.2_cluster_select_from_dendrogram.py``. Values are
kept byte-identical so the refactored pipeline reproduces the legacy output.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


from ashare_daily.paths import ALL_INSTRUMENTS_TXT, CLUSTER_MAPPING_SELECTED_TXT, TEMP_DIR, repo_root

_REPO = repo_root()

# ── Output paths ────────────────────────────────────────────────────────────
OUT_DIR = str(TEMP_DIR)
CSV_OUT = os.path.join(OUT_DIR, "stock_cluster_mapping.csv")
CSV_SELECTED_OUT = os.path.join(OUT_DIR, "stock_cluster_mapping_selected.csv")
DIAGNOSTICS_CSV = os.path.join(OUT_DIR, "stock_cluster_mapping_diagnostics.csv")


def dated_selected_csv_path(suffix: str) -> str:
    """Machine-generated selected pool; does not overwrite a maintained CSV."""
    return os.path.join(OUT_DIR, f"stock_cluster_mapping_selected_{suffix.strip()}.csv")


def dated_mapping_csv_path(suffix: str) -> str:
    """Full cluster mapping for an A/B suffix; does not overwrite production CSV."""
    return os.path.join(OUT_DIR, f"stock_cluster_mapping_{suffix.strip()}.csv")


def dated_metadata_path(suffix: str) -> str:
    """Generation metadata for an A/B suffix."""
    return os.path.join(OUT_DIR, f"stock_cluster_mapping_selected_metadata_{suffix.strip()}.json")


CSV_BENCHMARK_OUT = os.path.join(OUT_DIR, "stock_cluster_mapping_exp.csv")
CSV_SELECTED_BENCHMARK_OUT = os.path.join(OUT_DIR, "stock_cluster_mapping_selected_exp.csv")
METADATA_OUT = os.path.join(OUT_DIR, "stock_cluster_mapping_selected_metadata.json")
IMG_OUT = os.path.join(OUT_DIR, "stock_dendrogram_selected_reps.png")
SVG_OUT = os.path.join(OUT_DIR, "stock_dendrogram_selected_reps.svg")

CLUSTER_REFERENCE_MAP_CSV = CSV_OUT
STOCK_EXCLUDE_CODE_PREFIXES = ("BJ",)

# ── Hand-curated whitelists / blacklists ────────────────────────────────────
ALWAYS_KEEP_CODES = [
]

ALWAYS_DROP_CODES = [
]

# Put a seed code here to drop the whole cluster before clustering.
ALWAYS_DROP_CLUSTER_CODES = [
]

# Canonical txt output (consumed by pipeline orchestrator as --cluster-mapping-path).
DEFAULT_ALL_TXT = str(ALL_INSTRUMENTS_TXT)
DEFAULT_CANONICAL_TXT = str(CLUSTER_MAPPING_SELECTED_TXT)

# ── Clustering / selection thresholds ───────────────────────────────────────
# DIST_T 说明：distance threshold (dist = 1 - corr)。生产默认 0.60（≈ corr>=0.40 并簇）。
# 回滚 B0：POOL_DIST_T=0.70（见 experiments/cluster_tuning/rollback.env.example）。
DIST_T = float(os.environ.get("POOL_DIST_T", 0.60))
PER_CLUSTER = 1
RULE = "multi_rep_equal_mainline"
BENCHMARK_RULE = "multi_rep_exp_benchmark"
PRODUCTION_MULTI_REP_MAINLINE_SCHEME = "equal"
PRODUCTION_MULTI_REP_BENCHMARK_SCHEME = "exp"
DEPRECATED_MULTI_REP_WEIGHT_SCHEMES = ("harmonic",)
MULTI_REP_EXP_BASE = 0.65
# 生产默认 130（2026-06-03 cluster tuning B4）。回滚 B0：POOL_TARGET_COUNT=100。
TARGET_SELECTED_COUNT = int(os.environ.get("POOL_TARGET_COUNT", 100))
MAX_UNIVERSE_SIZE = int(os.environ.get("POOL_MAX_UNIVERSE_SIZE", 1200))
CLUSTER_LOOKBACK_DAYS = 252
MIN_CLUSTER_HISTORY_DAYS = 120
MIN_PAIR_OVERLAP_DAYS = 120
MIN_SELECTION_HISTORY_DAYS = 180
LOW_OVERLAP_DISTANCE = 0.95
# Final 去相关阈值：从最严 0.70 起步，以便跨簇近克隆在 trim 阶段被剐掉
# (Tier1 整改：0.75→0.70，进一步压制组合内跨簇冗余)
FINAL_MAX_ABS_CORR_THRESHOLDS = (0.70, 0.75, 0.80)

# 簇内（同 cluster）近克隆阈值：比跨簇阈值更严（同簇成员本来就共享主题，
# 保留多个 rank 只应在它们显著分化时才允许）。默认 0.60；
# 环境变量 POOL_INTRA_CLUSTER_CORR_LIMIT 可覆盖；设为 >= 1.0 即等效关闭。
INTRA_CLUSTER_MAX_ABS_CORR = float(os.environ.get("POOL_INTRA_CLUSTER_CORR_LIMIT", 0.60))

# average / ward / complete / single linkage for hierarchical clustering (POOL_LINKAGE).
# 生产默认 average：与 dist=1-corr + dist_t 阈值标定一致。
# ward 需要欧氏距离，直接喂相关距离时极易退化成「一股一簇」（2026-08 银行案例）。
# 回滚：export POOL_LINKAGE=ward
_CLUSTER_LINKAGE_RAW = os.environ.get("POOL_LINKAGE", "average").strip().lower()
CLUSTER_LINKAGE = (
    _CLUSTER_LINKAGE_RAW if _CLUSTER_LINKAGE_RAW in ("ward", "average", "complete", "single") else "average"
)

# Representative pick within each cluster cap: return | center_corr | blend.
# 生产默认 blend（B4）：收益排名与簇中心相关度各半，低中心相关惩罚；勿用纯 return 挑族内涨幅最高。
# 临时纯涨幅：export POOL_REP_PICK=return
_REP_PICK_RAW = os.environ.get("POOL_REP_PICK", "blend").strip().lower()
REP_PICK_MODE = _REP_PICK_RAW if _REP_PICK_RAW in ("return", "center_corr", "blend") else "blend"
REP_MIN_CENTER_CORR = float(os.environ.get("POOL_REP_MIN_CENTER_CORR", "0.55"))

# Within-cluster scoring: cumulative_return | sharpe_mom | legacy_mean.
# 生产默认 sharpe_mom：Sharpe 与 12-1 动量各半排名，避免纯累计涨幅挑代表。
# 临时用 cumulative_return：export POOL_SCORING_MODE=cumulative_return
_SCORING_MODE_RAW = os.environ.get("POOL_SCORING_MODE", "sharpe_mom").strip().lower()
SCORING_MODE = (
    _SCORING_MODE_RAW
    if _SCORING_MODE_RAW in ("cumulative_return", "sharpe_mom", "legacy_mean")
    else "sharpe_mom"
)

# 注：2026-04-24 移除了 dual-tier 收益预筛 + diversifier 分支。
# 单一路径：层级聚类 → 簇内 blend/sharpe_mom 挑代表 → 跨簇 low_corr 争席。

# ── MLFAM 结构性预筛（第4章）────────────────────────────────────────────────
PRESCREEN_ENABLED = os.environ.get("POOL_PRESCREEN", "1").strip().lower() in ("1", "true", "yes")
PRESCREEN_MAX_UNIVERSE = int(os.environ.get("POOL_PRESCREEN_MAX_UNIVERSE", "2200"))
PRESCREEN_VOL_PCTL = float(os.environ.get("POOL_PRESCREEN_VOL_PCTL", "20"))
PRESCREEN_MIN_PER_BUCKET = int(os.environ.get("POOL_PRESCREEN_MIN_PER_BUCKET", "2"))
PRESCREEN_DIAGNOSTICS_CSV = os.path.join(OUT_DIR, "stock_universe_prescreen_diagnostics.csv")

# Detone 会剥掉银行等高β行业的共同因子，残差相关下难以并簇。
# 生产默认关闭；需要 MLFAM 去市场因子时：export POOL_DETONE_CORR=1
DETONE_CORR_ENABLED = os.environ.get("POOL_DETONE_CORR", "0").strip().lower() in ("1", "true", "yes")

MERGE_LOW_COHESION_ENABLED = os.environ.get("POOL_MERGE_LOW_COhesion", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
MERGE_CORR_FLOOR = float(os.environ.get("POOL_MERGE_CORR_FLOOR", "0.35"))
MERGE_MAX_CLUSTER_SIZE = int(os.environ.get("POOL_MERGE_MAX_CLUSTER_SIZE", "2"))

COVERAGE_RESCUE_ENABLED = os.environ.get("POOL_COVERAGE_RESCUE", "1").strip().lower() in ("1", "true", "yes")
COVERAGE_RESCUE_MAX = int(os.environ.get("POOL_COVERAGE_RESCUE_MAX", "30"))

# Cross-cluster rank-1 ordering for Plan B quota cut (decoupled from within-cluster rep pick).
# 生产默认 low_corr_return：低相关 + 累计涨幅混合（要分散也要涨）。
# 纯分散：export POOL_CROSS_CLUSTER_RANK=low_corr
# 纯涨幅：export POOL_CROSS_CLUSTER_RANK=cumulative_return
_CROSS_CLUSTER_RANK_ALLOWED = (
    "low_corr_return",
    "low_corr",
    "cumulative_return",
    "sharpe_mom",
    "within_blend",
)
_CROSS_CLUSTER_RANK_RAW = os.environ.get("POOL_CROSS_CLUSTER_RANK", "low_corr_return").strip().lower()
CROSS_CLUSTER_RANK = (
    _CROSS_CLUSTER_RANK_RAW
    if _CROSS_CLUSTER_RANK_RAW in _CROSS_CLUSTER_RANK_ALLOWED
    else "low_corr_return"
)
# low_corr_return 中 low_corr 一侧权重（其余给 cumulative_return）。
# 默认 0.35：偏涨幅但仍惩罚高相关；回滚对半：export POOL_CROSS_LOW_CORR_WEIGHT=0.5
CROSS_LOW_CORR_WEIGHT = float(os.environ.get("POOL_CROSS_LOW_CORR_WEIGHT", "0.35"))
# 跨簇争席时，簇内按 cross_score 最优成员出战（而不是 blend 中心代表）。
# 避免「blend 钉死高相关中心股 → 整族低相关+高涨幅成员进不了」的情况（如 121 族寒武纪 vs 海光）。
# 关闭：export POOL_CROSS_REP_BY_SCORE=0
CROSS_REP_BY_SCORE = os.environ.get("POOL_CROSS_REP_BY_SCORE", "1").strip().lower() in ("1", "true", "yes")

MERGE_SINGLETONS_ENABLED = os.environ.get("POOL_MERGE_SINGLETONS", "1").strip().lower() in ("1", "true", "yes")
MERGE_SINGLETON_CORR = float(os.environ.get("POOL_MERGE_SINGLETON_CORR", "0.50"))

# Selection-stage liquidity floor (median dollar volume = volume * close).
SELECTION_LIQUIDITY_GATE_ENABLED = os.environ.get("POOL_SELECTION_LIQUIDITY_GATE", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
# apply: ``singleton`` (default) stricter gate for n=1 clusters; ``all`` for every selected code.
_SELECTION_LIQ_APPLY_RAW = os.environ.get("POOL_SELECTION_LIQUIDITY_APPLY", "singleton").strip().lower()
SELECTION_LIQUIDITY_APPLY = _SELECTION_LIQ_APPLY_RAW if _SELECTION_LIQ_APPLY_RAW in ("singleton", "all") else "singleton"
SELECTION_MIN_DOLLAR_VOLUME = float(os.environ.get("POOL_SELECTION_MIN_DOLLAR_VOLUME", "5.5e7"))
SINGLETON_MIN_DOLLAR_VOLUME = float(os.environ.get("POOL_SINGLETON_MIN_DOLLAR_VOLUME", "2.45e7"))
SINGLETON_LIQUIDITY_MULT = float(os.environ.get("POOL_SINGLETON_LIQUIDITY_MULT", "1.0"))
# Only rank-1 reps in the top-N cross_cluster_score queue face the liquidity floor
# (avoids shrinking a singleton-only universe below TARGET when most names are illiquid).
LIQUIDITY_HOT_RANK_CUTOFF = int(os.environ.get("POOL_LIQUIDITY_HOT_RANK_CUTOFF", "100"))
LIQUIDITY_DIAGNOSTICS_CSV = os.path.join(OUT_DIR, "stock_liquidity_gate_diagnostics.csv")

os.makedirs(OUT_DIR, exist_ok=True)
