# ashare-daily

A-share daily research workflow for people who already have a Qlib `cn_data` directory. This repo does not ship market data, HTML, or training caches.

Without a Qlib database the commands below cannot run.

## Setup

Python 3.12. Imports used by the daily commands: `qlib`, `pandas`, `numpy`, `plotly`, `scipy`, `skfolio`.

```bash
cp config.env.example config.env
cp configs/production_regime_switch_ewma_shrink.json.example configs/production_regime_switch_ewma_shrink.json
cp fund_pool_builder/shared_filter_config.json.example fund_pool_builder/shared_filter_config.json
```

Edit the two json files and `config.env`. `config.env` only has `PY` and `PLOTLY_ROOT`. Python reads the local json through `runtime_paths.py`, not `config.env`. If either file is missing, the daily commands exit and name the example.

`PLOTLY_ROOT` is where HTML and the train cache go. Keep it outside the git checkout.

Command 2 also needs `paths.etf_risk_report_script` in the local json, pointed at `generate_fat_tail_risk_report.py` from a separate ETF checkout. If that file is missing, the command stops when it reaches the risk CSV.

## Commands

Run from the repo root with `PYTHONPATH=.` for the Python entry points. The shells set that themselves.

1. Merge CSI800 and CSI1000 into `csi1800.txt`:

```bash
PYTHONPATH=. python fund_pool_builder/merge_csi1800.py
```

2. Build the selected cluster mapping:

```bash
PYTHONPATH=. python fund_pool_builder/生成cluster_mapping_selected_最短命令清单.py
```

3. Review the cluster mapping. CSV paths come from your local config, not from this file:

```bash
PYTHONPATH=. python fund_pool_builder/每日审查cluster_mapping_最短命令清单.py
```

4. Regime validation, resume the current month, skip HTML:

```bash
SKIP_REBUILD=1 SKIP_HTML=1 ./scripts/daily_regime_transition_validation_stock.sh
```

5. Adaptive HTML for the selected pool:

```bash
./scripts/daily_adaptive_stage_html_stock.sh
```

6. From-listing HTML, incremental (does not force a full redraw):

```bash
./scripts/daily_adaptive_from_listing_stock.sh
```

7. HRP dendrogram, default cut, then distance 0.8:

```bash
PYTHONPATH=. python workspace/scripts/generate_hrp_dendrogram_html.py
PYTHONPATH=. python workspace/scripts/generate_hrp_dendrogram_html.py --dist-t 0.8
```

8. Next-day trigger scan. Point `--listing-dir` at an existing `*_from_listing_stock` directory under `PLOTLY_ROOT`:

```bash
PYTHONPATH=. python decision_pack/scripts/scan_next_day_trigger_prices.py --listing-dir "$PLOTLY_ROOT/YYYYMMDD_from_listing_stock"
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_runtime_paths.py fund_pool_builder/tests
bash tests/test_daily_env.sh
```

## License

MIT. See `LICENSE`.
