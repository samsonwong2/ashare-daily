# ashare-daily

[English](README.md)

A 股日常命令：股票池合并、聚类映射、聚类复查、状态验证、自适应 HTML、上市以来 HTML、HRP 树状图、次日触发价。

需要 Qlib 的 `cn_data`。`pip` 不会下载这份行情。如果 `pip install` 找不到名为 `qlib` 的包，先按本机已有的方式装好 Qlib，再安装本仓库。

## 安装

```bash
pip install -e .
cp config.env.example config.env
cp configs/production_regime_switch_ewma_shrink.json.example configs/production_regime_switch_ewma_shrink.json
cp configs/shared_filter_config.json.example configs/shared_filter_config.json
```

改这三份拷贝。`config.env` 只需要 `PLOTLY_ROOT`。两个 json 放数据路径。缺文件时命令会退出，并写出对应的 example 文件名。`cluster-map` 跑到风险步骤时，还需要本地 json 里的 `paths.etf_risk_report_script`。

用两条符号链接指向状态分片目录和模型缓存。不要把那些 CSV 拷进 git：

```bash
mkdir -p runtime/decision_packs
ln -sfn /path/to/20260901_stock runtime/decision_packs/20260901_stock
ln -sfn /path/to/regime_transition_model_cache_stock runtime/decision_packs/regime_transition_model_cache_stock
```

## 日常顺序

按下面顺序跑。`adaptive` 读 `regime` 写出的验证目录。`listing` 读当天 `adaptive` 写出的 `all_adaptive_stock` 目录。这个目录不存在时，`listing` 退出并打印该路径，不会替你去跑 `adaptive`。

只有 `listing` 会把 HTML 写进 `$PLOTLY_ROOT/20260924_from_listing_stock`。日期来自 `--as-of 2026-09-24`。其余命令分别写股票池、聚类表、状态信号分片、另一个 HTML 目录、HRP 页面或触发价表。

```bash
ashare-daily pool-merge
ashare-daily cluster-map
ashare-daily cluster-review
ashare-daily regime --skip-rebuild --skip-html
ashare-daily adaptive --as-of 2026-09-24
ashare-daily listing --as-of 2026-09-24
ashare-daily hrp --asof-date 2026-09-24
ashare-daily hrp --asof-date 2026-09-24 --dist-t 0.8
ashare-daily triggers --listing-dir "$PLOTLY_ROOT/20260924_from_listing_stock"
```

`listing` 默认增量。加上 `--no-incremental` 才全量重算。listing 的 HTML 生成之后，图 7 到图 11 的 EWMA 上下轨一定会画，含 1 个月那一块。图 12 默认会画，加上 `--no-fig12` 才跳过。

### `ashare-daily pool-merge`

从本地 json 里的 Qlib instruments 目录读取 `csi800.txt` 和 `csi1000.txt`。各取最新结束日的成分，按代码取并集，在同一目录写出 `csi1800.txt`。这条不画 HTML。

### `ashare-daily cluster-map`

用这批股票做生产聚类。默认距离 `0.60`、选出 100 只、评分用夏普动量、去噪关闭。写出 `cluster_mapping.csv`、`cluster_mapping_selected.csv` 和 `cluster_mapping_selected.txt`。风险步骤只有在本地 json 的 `paths.etf_risk_report_script` 指向真实文件时才会跑。产物在本地 json 的 `TEMP_DIR` 下，不在 `PLOTLY_ROOT` 下。

### `ashare-daily cluster-review`

拿同一次选出的 `stock_cluster_mapping_selected.csv`，对照完整的 `stock_cluster_mapping.csv` 做质量复查。`--future-end` 默认是今天。复查结果写到复查输出目录。这条不刷新 listing HTML。

### `ashare-daily regime --skip-rebuild --skip-html`

按 `--as-of` 所在月份更新状态切换信号分片。不写 `--as-of` 时日期是今天，不是 `2026-09-24`。命令先确认验证目录存在，删掉该月的 `.done`，保留 `.signals.csv`，再跑信号回测（`--eval-start 2024-06-01`，`--horizon 10`）。`--skip-rebuild` 跳过反转指标重算。`--skip-html` 跳过状态 HTML，所以不会写出 `$PLOTLY_ROOT/{YYYYMMDD}all_stock`。2026 年 9 月共用一份分片 `2026-09.signals.csv`，后面的 `adaptive` 和 `listing` 都要这份文件。

### `ashare-daily adaptive --as-of 2026-09-24`

给聚类选出的 100 只股票画自适应阶段 HTML，截止日期是 2026-09-24。画图起点和训练截止默认都是 `2026-04-01`。它读取 `2026-09.signals.csv`，文件不在就直接退出。HTML 写到 `$PLOTLY_ROOT/20260924all_adaptive_stock`。训练命中 `$PLOTLY_ROOT/_train_cache` 就不再重训。除非加上 `--no-ticket-card`，同一目录里还会写一张 ticket card。这个目录是 `listing` 的配置来源，不是上市以来那份 HTML。

### `ashare-daily listing --as-of 2026-09-24`

按每只股票的上市日画到 2026-09-24。启动前同时要求 `2026-09.signals.csv` 和 `$PLOTLY_ROOT/20260924all_adaptive_stock` 都存在。adaptive 目录缺失时退出并打印该路径。HTML 写到 `$PLOTLY_ROOT/20260924_from_listing_stock`。增量模式会在 `PLOTLY_ROOT` 下找更早的 `*_from_listing_stock`，把中间缺的交易日补上。HTML 写完后，图 7 到图 11 会加上 EWMA 上下轨，含 1 个月那一块。图 12 默认会加，加上 `--no-fig12` 才跳过。

### `ashare-daily hrp`

给聚类股票池画 HRP 树状图。代码里的默认截止日期是 `2026-09-02`，回看 252 天。距离用生产默认 `0.60`。页面是 `$TEMP_DIR/hrp_dendrogram.html`。旁边的代表清单是 `cluster_representatives_20260902.csv`，文件名没有距离后缀。要对齐 2026-09-24，需要自己加上 `--asof-date` 和 `--output`。

### `ashare-daily hrp --dist-t 0.8`

同一张树状图，距离改成更粗的 `0.8`。HTML 默认仍是 `$TEMP_DIR/hrp_dendrogram.html`，所以会覆盖上一条 `hrp` 写出的页面。代表清单单独保存为 `cluster_representatives_20260902_d080.csv`（`0.8` 写成 `_d080`）。不传 `--asof-date` 时，截止日期仍是 `2026-09-02`。

### `ashare-daily triggers --listing-dir "$PLOTLY_ROOT/YYYYMMDD_from_listing_stock"`

扫描上市以来 HTML，把下一交易日的触发价写回这个目录：`next_day_triggers_{下一交易日}.md`、`.csv`，以及 `_hard_touch.csv`。锚是 `SH000300`。上面这一行里的 `YYYYMMDD` 是占位符。对 2026-09-24 这批，要换成 `20260924`，并且先在 shell 里导出 `PLOTLY_ROOT`。`config.env` 只由 `ashare-daily` 读取，shell 不会展开里面的变量，未展开的 `$PLOTLY_ROOT` 会原样传进参数。扫描器还要求 `--as-of`（最后收盘日）和 `--next-day`（下一交易日）。这一批应写成：

```bash
export PLOTLY_ROOT="$(grep -E '^PLOTLY_ROOT=' config.env | cut -d= -f2-)"
ashare-daily triggers \
  --listing-dir "$PLOTLY_ROOT/20260924_from_listing_stock" \
  --as-of 2026-09-24 \
  --next-day 2026-09-25
```

## 测试

```bash
pip install -e .
python -m pytest tests/test_paths.py tests/test_config_env.py tests/test_cli_argv.py tests/test_regime_resume.py tests/test_validation_precheck.py tests/test_listing_worker.py tests/test_as_of.py tests/test_filter_config.py tests/pool
```

`tests/pool` 里的 `test_export_mapping_columns_and_no_diagnostics_on_main_csv` 仍可能因 `KeyError: ma_stack` 失败。这次改目录之前就有这个失败。
