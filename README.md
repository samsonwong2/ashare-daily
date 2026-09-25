# ashare-daily

[中文](README.zh-CN.md)

Daily A-share commands: pool merge, cluster map, cluster review, regime validation, adaptive HTML, from-listing HTML, HRP dendrogram, and next-day triggers.

Qlib `cn_data` is required and is not downloaded by `pip`. If `pip install` cannot find a package named `qlib`, install Qlib the way this machine already has it, then install this checkout.

## Install

```bash
pip install -e .
cp config.env.example config.env
cp configs/production_regime_switch_ewma_shrink.json.example configs/production_regime_switch_ewma_shrink.json
cp configs/shared_filter_config.json.example configs/shared_filter_config.json
```

Edit the three copies. `config.env` only needs `PLOTLY_ROOT`. The json files hold data paths. A missing file exits and names the example. `cluster-map` also needs `paths.etf_risk_report_script` in the local json when the risk step runs.

Point two symlinks at your regime shard directory and model cache. Do not copy those CSVs into git:

```bash
mkdir -p runtime/decision_packs
ln -sfn /path/to/20260901_stock runtime/decision_packs/20260901_stock
ln -sfn /path/to/regime_transition_model_cache_stock runtime/decision_packs/regime_transition_model_cache_stock
```

## Daily order

Run these in order. `adaptive` reads the validation directory written by `regime`. `listing` reads that day's `all_adaptive_stock` directory written by `adaptive`. If that directory is missing, `listing` exits and prints the path. It does not run `adaptive` for you.

Only `listing` writes the HTML under `$PLOTLY_ROOT/20260924_from_listing_stock`. The date comes from `--as-of 2026-09-24`. The other commands write a pool file, cluster tables, a regime signal shard, a different HTML directory, an HRP page, or a trigger table.

```bash
ashare-daily pool-merge
ashare-daily cluster-map
ashare-daily cluster-review
ashare-daily regime --skip-rebuild --skip-html
ashare-daily adaptive --as-of 2026-09-24
ashare-daily listing --as-of 2026-09-24
ashare-daily hrp
ashare-daily hrp --dist-t 0.8
ashare-daily triggers --listing-dir "$PLOTLY_ROOT/YYYYMMDD_from_listing_stock"
```

`listing` is incremental unless you pass `--no-incremental`. EWMA rails on figures 7 through 11, including the 1-month panel, always run after listing HTML. Figure 12 runs unless `--no-fig12`.

### `ashare-daily pool-merge`

Reads `csi800.txt` and `csi1000.txt` from the Qlib instruments directory in the local json. Keeps each file's latest end date, unions the codes, and writes `csi1800.txt` next to them. This does not draw HTML.

### `ashare-daily cluster-map`

Builds the production cluster map from that universe. Defaults are distance `0.60`, 100 names, Sharpe-momentum scoring, and detone off. It writes `cluster_mapping.csv`, `cluster_mapping_selected.csv`, and `cluster_mapping_selected.txt`. The risk step runs only when `paths.etf_risk_report_script` in the local json points at a real file. Output stays under the local json `TEMP_DIR`, not under `PLOTLY_ROOT`.

### `ashare-daily cluster-review`

Audits `stock_cluster_mapping_selected.csv` against the full `stock_cluster_mapping.csv` from the same select run. `--future-end` defaults to today. This writes a review under the review output directory. It does not refresh the listing HTML.

### `ashare-daily regime --skip-rebuild --skip-html`

Updates the regime-transition signal shard for the month of `--as-of`. With no `--as-of`, that date is today, not `2026-09-24`. The command checks that the validation directory exists, deletes that month's `.done`, and keeps the `.signals.csv`. It then runs the signal backtest (`--eval-start 2024-06-01`, `--horizon 10`). `--skip-rebuild` skips the reversal-metric rebuild. `--skip-html` skips the regime HTML, so nothing is written to `$PLOTLY_ROOT/{YYYYMMDD}all_stock`. September 2026 shares one shard, `2026-09.signals.csv`, which `adaptive` and `listing` both require.

### `ashare-daily adaptive --as-of 2026-09-24`

Draws the adaptive-stage HTML for the 100 cluster-selected names, ending 2026-09-24. Plot start and train cutoff default to `2026-04-01`. It reads `2026-09.signals.csv` and refuses to start if that file is missing. HTML goes to `$PLOTLY_ROOT/20260924all_adaptive_stock`. Training reuses `$PLOTLY_ROOT/_train_cache`. A ticket card is written into the same directory unless `--no-ticket-card` is set. This directory is the config source for `listing`. It is not the from-listing HTML folder.

### `ashare-daily listing --as-of 2026-09-24`

Draws one HTML per name from listing date through 2026-09-24. It first requires both `2026-09.signals.csv` and `$PLOTLY_ROOT/20260924all_adaptive_stock`. If the adaptive directory is missing, it exits and prints that path. HTML goes to `$PLOTLY_ROOT/20260924_from_listing_stock`. Incremental mode appends each missing session after the newest older `*_from_listing_stock` directory under `PLOTLY_ROOT`. After the HTML exists, EWMA rails are added to figures 7 through 11, including the 1-month panel, and figure 12 is added unless `--no-fig12` is set.

### `ashare-daily hrp`

Builds the HRP dendrogram for the cluster pool. Lookback is 252 days. Distance stays at the production default `0.60`. With no `--asof-date`, the end date is today. The page is `$TEMP_DIR/hrp_dendrogram_{YYYYMMDD}.html`. The representative table beside it is `cluster_representatives_{YYYYMMDD}.csv`. Pass `--asof-date 2026-09-24` when the page should match that listing day.

### `ashare-daily hrp --dist-t 0.8`

Same dendrogram with a coarser distance cut of `0.8`. The page is `$TEMP_DIR/hrp_dendrogram_{YYYYMMDD}_d080.html`, so it does not replace the `0.60` page. The representative table is `cluster_representatives_{YYYYMMDD}_d080.csv`. The end date is still today until you pass `--asof-date`.

### `ashare-daily triggers --listing-dir "$PLOTLY_ROOT/YYYYMMDD_from_listing_stock"`

Scans the from-listing HTML and writes next-session trigger prices back into that same directory: `next_day_triggers_{next-day}.md`, `.csv`, and `_hard_touch.csv`. The anchor is `SH000300`. `YYYYMMDD` in the line above is a placeholder. For the 2026-09-24 batch, replace it with `20260924`. If the shell did not expand `$PLOTLY_ROOT`, the command still looks up that directory name under `PLOTLY_ROOT` from `config.env`. When `--as-of` and `--next-day` are omitted, the date in the directory name is T, and T+1 is the next weekday. `20260924_from_listing_stock` becomes `--as-of 2026-09-24 --next-day 2026-09-25`. This skip does not know exchange holidays. Pass `--next-day` yourself when the next session is not the next weekday.

## Example page

[examples/查看HTML.md](examples/查看HTML.md) explains how to open the BYD from-listing sample, `examples/regime_transition_SZ002594_比亚迪_20110630_20260924_adaptive.html`.

## Tests

```bash
pip install -e .
python -m pytest tests/test_paths.py tests/test_config_env.py tests/test_cli_argv.py tests/test_regime_resume.py tests/test_validation_precheck.py tests/test_listing_worker.py tests/test_as_of.py tests/test_filter_config.py tests/pool
```

`tests/pool` may still fail `test_export_mapping_columns_and_no_diagnostics_on_main_csv` with `KeyError: ma_stack`. That failure existed before this layout.
